# vectorstore_collection.py

import os
from typing import List, Dict, Optional, Tuple, Set, Any
from enum import Enum
from langchain.schema import Document
from langchain_chroma import Chroma
from src.utils.logger_utils import setup_logging
from src.ai_models.ai_models import get_embedding_model
import hashlib

logger = setup_logging(__name__)

class CollectionType(Enum):
    """Enum for different types of vector store collections."""
    NARRATIVE_ARCS = "narrative_arcs"

class VectorStoreCollection:
    """Base class for vector store collections."""

    def __init__(self, collection_name: str, persist_directory: str = os.getenv("PERSIST_DIRECTORY")):
        """
        Initialize the VectorStoreCollection.

        Args:
            collection_name (str): Name of the collection.
            persist_directory (str): Directory to persist the vector store.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = get_embedding_model()
        self.collection = self._initialize_collection()

    def _initialize_collection(self) -> Chroma:
        """Initialize and return a Chroma vector store instance."""
        return Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )
    
    def get_all_embeddings(self) -> List[List[float]]:
        """Get all embeddings from the collection."""
        return self.collection.get(include=["embeddings"])["embeddings"]
    
    def get_all_ids(self) -> List[str]:
        """Get all IDs from the collection."""
        # Assuming that the Chroma vector store does not support fetching IDs directly,
        # you may need to store IDs in a different way or fetch them from the metadata.
        results = self.collection.get(include=["metadatas"])  # Fetch metadata instead
        return [meta.get('id') for meta in results["metadatas"] if 'id' in meta]  # Extract IDs from metadata
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents from the collection."""
        return self.collection.get(include=["documents"])["documents"]
    
    def get_all_metadatas(self) -> List[Dict]:
        """Get all metadatas from the collection."""
        return self.collection.get(include=["metadatas"])["metadatas"]

    @staticmethod
    def generate_content_based_id(text: str) -> str:
        """Generate a unique ID based on the content of the text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def add_documents(self, documents: List[Document], ids: List[str]):
        """Add documents to the vector store."""
        self.collection.add_documents(documents, ids=ids)
        logger.info(f"Added {len(documents)} documents to the vector store.")

    def similarity_search_with_score(self, query: str, k: int = 5, filter: Optional[Dict] = None):
        """Perform a similarity search with cosine distances."""
        return self.collection.similarity_search_with_score(query, k=k, filter=filter)

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by its ID."""
        results = self.collection.get(ids=[doc_id], include=["metadatas", "documents"])
        if results['ids']:
            metadata = results['metadatas'][0]
            content = results['documents'][0]
            return Document(page_content=content, metadata=metadata)
        return None

    def delete_collection(self) -> None:
        """Delete the entire collection from the vector store."""
        self.collection.delete_collection()
        logger.info(f"Deleted collection {self.collection_name}")

    def find_similar_arcs(self, query: str, n_results: int = 5, series: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find similar narrative arcs based on a query."""
        filter_criteria = {"$and": [{"doc_type": "main"}]}

        if series:
            filter_criteria["$and"].append({"series": series})

        logger.debug(f"Query: {query}")
        logger.debug(f"Filter criteria for similarity search: {filter_criteria}")

        try:
            results = self.similarity_search_with_score(
                query,
                k=n_results,
                filter=filter_criteria
            )
            logger.debug(f"Number of results found: {len(results)}")
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

        # Include cosine distance in the returned results
        return [{"metadata": result[0].metadata, "cosine_distance": result[1]} for result in results]

def get_vectorstore_collection(collection_type: CollectionType = CollectionType.NARRATIVE_ARCS) -> VectorStoreCollection:
    """
    Get the appropriate vector store collection based on the collection type.

    Args:
        collection_type (CollectionType): The type of collection to retrieve.

    Returns:
        VectorStoreCollection: An instance of the appropriate vector store collection.

    Raises:
        ValueError: If an unsupported collection type is provided.
    """
    if collection_type == CollectionType.NARRATIVE_ARCS:
        return VectorStoreCollection(CollectionType.NARRATIVE_ARCS.value)
    else:
        raise ValueError(f"Unsupported collection type: {collection_type}")
