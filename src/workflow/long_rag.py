from typing import Any
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent, Context
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llms import LLM
from src.docs_processing.splitter import split_doc
from src.docs_processing.grouper import get_grouped_docs
from src.retrieval.retriever import LongRAGRetriever
from .events import LoadNodeEvent

class LongRAGWorkflow(Workflow):
    """Long RAG Workflow."""

    @step
    async def ingest(self, ev: StartEvent) -> LoadNodeEvent | None:
        """Ingestion step.

        Args:
            ctx (Context): Context
            ev (StartEvent): start event

        Returns:
            StopEvent | None: stop event with result
        """
        data_dir: str = ev.get("data_dir")
        llm: LLM = ev.get("llm")
        chunk_size: int | None = ev.get("chunk_size")
        similarity_top_k: int = ev.get("similarity_top_k")
        small_chunk_size: int = ev.get("small_chunk_size")
        index: VectorStoreIndex | None = ev.get("index")
        index_kwargs: dict[str, Any] | None = ev.get("index_kwargs")

        if any(
            i is None
            for i in [data_dir, llm, similarity_top_k, small_chunk_size]
        ):
            return None

        if not index:
            docs = SimpleDirectoryReader(data_dir).load_data()
            if chunk_size is not None:
                nodes = split_doc(
                    chunk_size, docs
                )  # split documents into chunks of chunk_size
                grouped_nodes = get_grouped_docs(
                    nodes
                )  # get list of nodes after grouping (groups are combined into one node), these are long retrieval units
            else:
                grouped_nodes = docs

            # split large retrieval units into smaller nodes
            small_nodes = split_doc(small_chunk_size, grouped_nodes)

            index_kwargs = index_kwargs or {}
            index = VectorStoreIndex(small_nodes, **index_kwargs)
        else:
            # get smaller nodes from index and form large retrieval units from these nodes
            small_nodes = index.docstore.docs.values()
            grouped_nodes = get_grouped_docs(small_nodes, None)

        return LoadNodeEvent(
            small_nodes=small_nodes,
            grouped_nodes=grouped_nodes,
            index=index,
            similarity_top_k=similarity_top_k,
            llm=llm,
        )

    @step
    async def make_query_engine(
        self, ctx: Context, ev: LoadNodeEvent
    ) -> StopEvent:
        """Query engine construction step.

        Args:
            ctx (Context): context
            ev (LoadNodeEvent): event

        Returns:
            StopEvent: stop event
        """
        # make retriever and query engine
        retriever = LongRAGRetriever(
            grouped_nodes=ev.grouped_nodes,
            small_toks=ev.small_nodes,
            similarity_top_k=ev.similarity_top_k,
            vector_store=ev.index.vector_store,
        )
        query_eng = RetrieverQueryEngine.from_args(retriever, ev.llm)

        return StopEvent(
            result={
                "retriever": retriever,
                "query_engine": query_eng,
                "index": ev.index,
            }
        )

    @step
    async def query(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Query step.

        Args:
            ctx (Context): context
            ev (StartEvent): start event

        Returns:
            StopEvent | None: stop event with result
        """
        query_str: str | None = ev.get("query_str")
        query_eng = ev.get("query_eng")

        if query_str is None:
            return None

        result = query_eng.query(query_str)
        return StopEvent(result=result)