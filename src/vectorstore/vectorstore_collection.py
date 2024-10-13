import os
from typing import List, Dict, Optional
from enum import Enum
from langchain.schema import Document
from langchain_chroma import Chroma
from src.utils.logger_utils import setup_logging
from src.narrative_classes.narrative_classes import NarrativeArc
from src.ai_models.ai_models import get_embedding_model
import hashlib

logger = setup_logging(__name__)

try:
    import sys
    import pysqlite3 as sqlite3
    sys.modules["sqlite3"] = sqlite3
    logger.debug("Imported pysqlite3 successfully.")
except:
    logger.warning("Failed to import pysqlite3.")
    pass

class CollectionType(Enum):
    """Enum for different types of vector store collections."""
    NARRATIVE_ARCS = "narrative_arcs"

class VectorStoreCollection:
    """Base class for vector store collections."""

    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
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

    @staticmethod
    def generate_content_based_id(text: str) -> str:
        """Generate a unique ID based on the content of the text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_embeddings(self) -> List[List[float]]:
        """Get the embeddings from the vector store."""
        return self.collection.get(include=['embeddings'])['embeddings']
    
    def get_ids(self) -> List[str]:
        """Get the ids from the vector store."""
        return self.collection.get()['ids']
    
    def get_documents(self) -> List[Document]:
        """Get the documents from the vector store."""
        return self.collection.get()['documents']
    
    def get_metadatas(self) -> List[Dict[str, str]]:
        """Get the metadatas from the vector store."""
        return self.collection.get()['metadatas']

    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about the collection."""
        return self.collection.get_collection_stats()

    def delete_collection(self) -> None:
        """Delete the entire collection from the vector store."""
        self.collection.delete_collection()
        logger.info(f"Deleted collection {self.collection_name}")

    def persist(self) -> None:
        """Persist the vector store to disk."""
        self.collection.persist()
        logger.info(f"Vector store persisted to {self.persist_directory}")

class NarrativeArcCollection(VectorStoreCollection):
    """Collection specifically for narrative arcs."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the NarrativeArcCollection."""
        super().__init__(CollectionType.NARRATIVE_ARCS.value, persist_directory)

    def add_narrative_arcs(self, arcs: List[NarrativeArc], series: str, season: str, episode: str) -> None:
        """
        Add a list of narrative arcs to the vector database.

        Args:
            arcs (List[NarrativeArc]): A list of NarrativeArc objects to be added to the database.
            series (str): The series identifier.
            season (str): The season identifier.
            episode (str): The episode identifier.
        """
        if not arcs:
            logger.warning(f"No narrative arcs provided for {series} {season} {episode}. Skipping database update.")
            return

        documents = []
        ids = []
        for arc in arcs:
            # Create or update the main document for the arc
            main_doc = Document(
                page_content=f"{arc.Title}\n\n{arc.Description}",
                metadata={
                    "title": arc.Title,
                    "description": arc.Description,
                    "arc_type": arc.Arc_Type,
                    "duration": arc.Duration,
                    "characters": ",".join(arc.Characters),
                    "doc_type": "main"
                }
            )
            documents.append(main_doc)
            ids.append(self.generate_content_based_id(f"main_{arc.Title}_{series}_{season}_{episode}"))

            # Create a document for the episode's progression
            progression_doc = Document(
                page_content="\n".join(arc.Progression),
                metadata={
                    "title": arc.Title,
                    "arc_type": arc.Arc_Type,
                    "duration": arc.Duration,
                    "characters": ",".join(arc.Characters),
                    "progression": "\n".join(arc.Progression),
                    "doc_type": "progression",
                    "series": series,
                    "season": season,
                    "episode": episode
                }
            )
            documents.append(progression_doc)
            ids.append(self.generate_content_based_id(f"prog_{arc.Title}_{series}_{season}_{episode}"))

        # Add documents with their corresponding IDs
        if documents and ids:
            self.collection.add_documents(documents, ids=ids)
            logger.info(f"Added {len(arcs)} narrative arcs for {series} {season} {episode} to the vector database.")
        else:
            logger.warning(f"No valid documents or IDs generated for {series} {season} {episode}. Skipping database update.")

    def find_similar_arcs(self, query: str, n_results: int = 5) -> List[Dict[str, str]]:
        """
        Find similar narrative arcs in the vector database based on a query.

        Args:
            query (str): The query text to search for similar arcs.
            n_results (int): The number of similar arcs to return. Defaults to 5.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing similar arc information.
        """
        results = self.collection.similarity_search_with_score(
            query, 
            k=n_results,
            filter={"doc_type": "main"}  # Only search in main documents
        )
        
        similar_arcs = []
        for doc, score in results:
            # Fetch all progression documents for this arc
            progression_docs = self.collection.similarity_search(
                query,
                k=100,  # Adjust this number as needed
                filter={"title": doc.metadata["title"], "doc_type": "progression"}
            )
            progressions = [
                {
                    "series": prog_doc.metadata["series"],
                    "season": prog_doc.metadata["season"],
                    "episode": prog_doc.metadata["episode"],
                    "content": prog_doc.page_content.split("\n")
                }
                for prog_doc in progression_docs
            ]
            
            similar_arcs.append({
                "title": doc.metadata["title"],
                "arc_type": doc.metadata["arc_type"],
                "description": doc.metadata["description"],
                "duration": doc.metadata["duration"],
                "characters": doc.metadata["characters"].split(','),
                "progressions": progressions,
                "similarity_score": score
            })
        
        return similar_arcs

    def update_narrative_arc(self, arc: NarrativeArc, series: str, season: str, episode: str) -> None:
        """
        Update an existing narrative arc in the vector database or create a new one if it doesn't exist.

        Args:
            arc (NarrativeArc): The updated NarrativeArc object.
            series (str): The series identifier.
            season (str): The season identifier.
            episode (str): The episode identifier.
        """
        # Update or create the main document
        main_doc_id = self.generate_content_based_id(f"main_{arc.Title}_{series}_{season}_{episode}")
        main_doc = Document(
            page_content=f"{arc.Title}\n\n{arc.Description}",
            metadata={
                "title": arc.Title,
                "description": arc.Description,
                "arc_type": arc.Arc_Type,
                "duration": arc.Duration,
                "characters": ",".join(arc.Characters),
                "doc_type": "main"
            }
        )
        
        # Check if the document exists before updating
        existing_docs = self.collection.get(ids=[main_doc_id])
        if existing_docs['ids']:
            self.collection.update_document(document_id=main_doc_id, document=main_doc)
        else:
            self.collection.add_documents([main_doc], ids=[main_doc_id])

        # Add or update the progression document for this episode
        prog_doc_id = self.generate_content_based_id(f"prog_{arc.Title}_{series}_{season}_{episode}")
        prog_doc = Document(
            page_content="\n".join(arc.Progression),
            metadata={
                "title": arc.Title,
                "arc_type": arc.Arc_Type,
                "duration": arc.Duration,
                "characters": ",".join(arc.Characters),
                "doc_type": "progression",
                "series": series,
                "season": season,
                "episode": episode
            }
        )
        
        # Check if the document exists before updating
        existing_docs = self.collection.get(ids=[prog_doc_id])
        if existing_docs['ids']:
            self.collection.update_document(document_id=prog_doc_id, document=prog_doc)
        else:
            self.collection.add_documents([prog_doc], ids=[prog_doc_id])

        logger.info(f"Updated or added narrative arc: {arc.Title} for {series} {season} {episode}")

    def delete_narrative_arc(self, arc_title: str) -> None:
        """
        Delete a narrative arc from the vector database.

        Args:
            arc_title (str): The title of the narrative arc to delete.
        """
        # Delete the main document
        main_doc_id = self.generate_content_based_id(f"main_{arc_title}")
        self.collection.delete(ids=[main_doc_id])

        # Delete all progression documents for this arc
        progression_docs = self.collection.similarity_search(
            arc_title,
            k=100,  # Adjust this number as needed
            filter={"title": arc_title, "doc_type": "progression"}
        )
        prog_doc_ids = [self.generate_content_based_id(f"prog_{arc_title}_{doc.metadata['series']}_{doc.metadata['season']}_{doc.metadata['episode']}") for doc in progression_docs]
        self.collection.delete(ids=prog_doc_ids)

        logger.info(f"Deleted narrative arc: {arc_title} and all its progressions")

    def get_narrative_arcs_titles(self) -> List[str]:
        """Get the titles of the narrative arcs from the vector store."""
        return list(set([doc.metadata['title'] for doc in self.get_documents() if doc.metadata['doc_type'] == 'main']))

    def update_narrative_arcs(self, existing_arcs: Dict[str, NarrativeArc], new_arcs: List[NarrativeArc], series: str, season: str, episode: str) -> Dict[str, NarrativeArc]:
        """
        Update existing narrative arcs with new arcs.

        Args:
            existing_arcs (Dict[str, NarrativeArc]): A dictionary of existing narrative arcs.
            new_arcs (List[NarrativeArc]): A list of new narrative arcs to be added or updated.
            series (str): The series identifier.
            season (str): The season identifier.
            episode (str): The episode identifier.

        Returns:
            Dict[str, NarrativeArc]: An updated dictionary of narrative arcs.
        """
        updated_arcs = existing_arcs.copy()

        for new_arc in new_arcs:
            if new_arc.Title in updated_arcs:
                # Update existing arc
                existing_arc = updated_arcs[new_arc.Title]
                existing_arc.Description = new_arc.Description
                existing_arc.Arc_Type = new_arc.Arc_Type
                existing_arc.Duration = new_arc.Duration
                existing_arc.Characters = list(set(existing_arc.Characters + new_arc.Characters))
                existing_arc.Progression.extend(new_arc.Progression)
            else:
                # Add new arc
                updated_arcs[new_arc.Title] = new_arc

        # Update the vector store
        for arc in updated_arcs.values():
            self.update_narrative_arc(arc, series, season, episode)

        logger.info(f"Updated {len(new_arcs)} narrative arcs in the vector store.")
        return updated_arcs

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
        return NarrativeArcCollection()
    else:
        raise ValueError(f"Unsupported collection type: {collection_type}")