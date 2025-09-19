from typing import List, Set
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores.simple import BasePydanticVectorStore
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.settings import Settings
from constants import DEFAULT_TOP_K


class LongRAGRetriever(BaseRetriever):
    """Long RAG Retriever."""

    def __init__(
        self,
        grouped_nodes: List[TextNode],
        small_toks: List[TextNode],
        vector_store: BasePydanticVectorStore,
        similarity_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """Constructor.

        Args:
            grouped_nodes (List[TextNode]): Long retrieval units, nodes with docs grouped together based on relationships
            small_toks (List[TextNode]): Smaller tokens
            embed_model (BaseEmbedding, optional): Embed model. Defaults to None.
            similarity_top_k (int, optional): Similarity top k. Defaults to 8.
        """
        self._grouped_nodes = grouped_nodes
        self._grouped_nodes_dict = {node.id_: node for node in grouped_nodes}
        self._small_toks = small_toks
        self._small_toks_dict = {node.id_: node for node in self._small_toks}

        self._similarity_top_k = similarity_top_k
        self._vec_store = vector_store
        self._embed_model = Settings.embed_model

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieves.

        Args:
            query_bundle (QueryBundle): query bundle

        Returns:
            List[NodeWithScore]: nodes with scores
        """
        # make query
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=500
        )

        # query for answer
        query_res = self._vec_store.query(vector_store_query)

        # determine top parents of most similar children (these are long retrieval units)
        top_parents_set: Set[str] = set()
        top_parents: List[NodeWithScore] = []
        for id_, similarity in zip(query_res.ids, query_res.similarities):
            cur_node = self._small_toks_dict[id_]
            parent_id = cur_node.ref_doc_id
            if parent_id not in top_parents_set:
                top_parents_set.add(parent_id)

                parent_node = self._grouped_nodes_dict[parent_id]
                node_with_score = NodeWithScore(
                    node=parent_node, score=similarity
                )
                top_parents.append(node_with_score)

                if len(top_parents_set) >= self._similarity_top_k:
                    break

        assert len(top_parents) == min(
            self._similarity_top_k, len(self._grouped_nodes)
        )

        return top_parents