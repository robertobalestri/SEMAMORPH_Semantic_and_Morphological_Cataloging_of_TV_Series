# vector_store_service.py

from typing import List, Dict, Optional, Any
from langchain.schema import Document
from langchain_chroma import Chroma
from src.ai_models.ai_models import get_embedding_model
import logging
import os

from src.utils.logger_utils import setup_logging
logger = setup_logging(__name__)

class VectorStoreService:
    """Service to manage vector store interactions."""

    def __init__(self, collection_name: str = "narrative_arcs", persist_directory: Optional[str] = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or os.getenv("PERSIST_DIRECTORY", "./vector_store")
        self.embedding_model = get_embedding_model()
        self.collection = self._initialize_collection()

    def _initialize_collection(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[Document], ids: List[str]):
        try:
            self.collection.add_documents(documents, ids=ids)
            logger.info(f"Added {len(documents)} documents to the vector store.")
        except Exception as e:
            logger.error(f"Failed to add documents to the vector store: {e}")
            raise

    def find_similar_arcs(
        self,
        query: str,
        n_results: int = 5,
        series: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find similar arcs (main documents only) to a query."""
        filter_criteria = {"$and": [{"doc_type": "main"}]}
        if series:
            filter_criteria["$and"].append({"series": series})

        try:
            results = self.collection.similarity_search_with_score(
                query,
                k=n_results,
                filter=filter_criteria
            )
            return [{"metadata": result[0].metadata, "cosine_distance": result[1]} for result in results]
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def delete_documents_by_arc(self, arc_id: str):
        """Delete all documents related to a specific arc."""
        try:
            # Retrieve main document IDs where metadata 'id' matches arc_id
            main_docs_response = self.collection.get(
                where={"id": arc_id},
                include=["metadatas"]
            )
            # Retrieve progression document IDs where metadata 'main_arc_id' matches arc_id
            prog_docs_response = self.collection.get(
                where={"main_arc_id": arc_id},
                include=["metadatas"]
            )

            # Initialize list to collect IDs to delete
            ids_to_delete = []

            # Safely extract 'ids' from main_docs_response
            main_ids = main_docs_response.get('ids', [])
            ids_to_delete.extend(main_ids)

            # Safely extract 'ids' from prog_docs_response
            prog_ids = prog_docs_response.get('ids', [])
            ids_to_delete.extend(prog_ids)

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents for arc ID {arc_id}.")
            else:
                logger.info(f"No documents found for arc ID {arc_id}.")

        except Exception as e:
            logger.error(f"Error deleting documents for arc ID {arc_id}: {e}")
            raise

    def delete_all_documents(self):
        """Delete all documents from the collection."""
        try:
            self.collection.delete()
            logger.info("Deleted all documents from collection")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    def get_all_documents(self, series: str, include_embeddings: bool = False) -> List[Dict]:
        """Get all documents for a series from the vector store."""
        try:
            # Get documents with series filter
            results = self.collection.get(
                where={"series": series},
                include=['metadatas', 'documents', 'embeddings'] if include_embeddings else ['metadatas', 'documents']
            )

            # Format results
            documents = []
            for i in range(len(results['ids'])):
                doc = {
                    'id': results['ids'][i],
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i],
                }
                if include_embeddings and 'embeddings' in results:
                    doc['embedding'] = results['embeddings'][i]
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []

    def find_similar_documents(
        self,
        query: str,
        series: str,
        n_results: int = 10,
        include_embeddings: bool = False
    ) -> List[Dict]:
        """Find similar documents (both arcs and progressions) to a query."""
        try:
            filter_criteria = {"series": series}
            
            # Perform similarity search
            results = self.collection.similarity_search_with_score(
                query,
                k=n_results,
                filter=filter_criteria
            )

            # Format results
            documents = []
            for doc, score in results:
                entry = {
                    'id': doc.metadata.get('id'),
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'distance': score
                }
                documents.append(entry)

            return documents

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
