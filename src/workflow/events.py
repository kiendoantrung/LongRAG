from typing import Iterable
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.workflow import Event
from llama_index.core.schema import TextNode

class LoadNodeEvent(Event):
    """Event for loading nodes."""
    small_nodes: Iterable[TextNode]
    grouped_nodes: list[TextNode]
    index: VectorStoreIndex
    similarity_top_k: int 
    llm: LLM