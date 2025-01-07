from typing import List
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.node_parser import SentenceSplitter

def split_doc(chunk_size: int, documents: List[BaseNode]) -> List[TextNode]:
    """Splits documents into smaller pieces."""
    text_parser = SentenceSplitter(chunk_size=chunk_size)
    return text_parser.get_nodes_from_documents(documents)