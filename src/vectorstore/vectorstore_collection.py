# vectorstore_collection.py
import os
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from langchain.schema import Document
from langchain_chroma import Chroma
from src.utils.logger_utils import setup_logging
from src.narrative_classes.narrative_classes import NarrativeArc, ArcProgression
from src.ai_models.ai_models import get_embedding_model, get_llm, LLMType
import hashlib
from langchain.prompts import ChatPromptTemplate
from src.utils.llm_utils import clean_llm_json_response
from pydantic import BaseModel
import traceback

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

class MainDocumentMetadata(BaseModel):
    title: str
    description: str
    arc_type: str
    episodic: bool
    characters: str
    doc_type: str
    series: str
    season: str
    episode: str
    id: str

class ProgressionDocumentMetadata(BaseModel):
    title: str
    arc_type: str
    episodic: bool
    characters: str
    progression: str
    doc_type: str
    series: str
    season: Optional[str] = None
    episode: Optional[str] = None
    id: str
    ordinal_progression_position: int  # New field

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

class NarrativeArcCollection(VectorStoreCollection):
    """Collection specifically for narrative arcs."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the NarrativeArcCollection."""
        super().__init__(CollectionType.NARRATIVE_ARCS.value, persist_directory)
        self.llm_cheap = get_llm(LLMType.CHEAP)  # Initialize cheap LLM here

    def add_or_update_narrative_arcs(self, arcs: List[NarrativeArc], series: str, season: str, episode: str) -> List[Tuple[NarrativeArc, str]]:
        if not arcs:
            logger.warning(f"No narrative arcs provided for {series} {season} {episode}. Skipping database update.")
            return []

        arc_id_map = []
        processed_arcs: Set[str] = set()
        added_arc_ids: Set[str] = set()
        changes: List[Dict] = []  # To keep track of changes for potential rollback

        cosine_distance_threshold = 0.3

        try:
            for arc in arcs:
                #if arc.title in processed_arcs:
                #    continue

                query = f"{arc.title}\n\n{arc.description}\n\n{' '.join([prog.content for prog in arc.progressions])}"
                similar_arcs = self.find_similar_arcs(query, n_results=5, series=series, exclude_ids=added_arc_ids)
                


                merged = False
                for similar_arc in similar_arcs:
                    if similar_arc['cosine_distance'] < cosine_distance_threshold:
                    
                        existing_arc = self.get_arc_by_id(similar_arc['metadata']['id'])
                        
                        if existing_arc and self.decide_arc_merging(arc, existing_arc):

                            if existing_arc.id in added_arc_ids:
                                logger.warning(f"Arc to be merged with same episode merging. Just mantain the existing arc and discard the new one.")
                                #here we should discard the new arc from the lists where we have that
                                arcs.remove(arc)
                                merged = True
                                break



                            logger.info(f"Merging arc '{arc.title}' with existing arc '{existing_arc.title}'")
                            
                            # Store original state for potential rollback
                            original_state = {
                                'arc': existing_arc.model_copy(deep=True),
                                'action': 'update'
                            }
                            
                            # Perform the merge
                            # We don't update title and description because the embeddings and the arc id are calculated on them
                            existing_arc.arc_type = arc.arc_type
                            existing_arc.episodic = arc.episodic
                            existing_arc.characters = list(set(existing_arc.characters + arc.characters))
                            
                            new_progression = arc.progressions[0]
                            new_progression.ordinal_position = len(existing_arc.progressions) + 1
                            new_progression.main_arc_id = existing_arc.id
                            existing_arc.progressions.append(new_progression)
                            
                            self._save_main_arc_document(existing_arc)
                            self._save_progression_document(existing_arc, new_progression)
                            
                            changes.append(original_state)
                            
                            arc_id_map.append((existing_arc, existing_arc.id))
                            added_arc_ids.add(existing_arc.id)
                            merged = True
                            processed_arcs.add(arc.title)
                            break
                
                if not merged:
                    logger.info(f"Adding new arc: '{arc.title}'")
                    new_arc_id = self.generate_content_based_id(f"main_{arc.title}_{series}")
                    arc.id = new_arc_id
                    arc.progressions[0].ordinal_position = 1
                    arc.progressions[0].main_arc_id = new_arc_id
                    self._save_main_arc_document(arc)
                    self._save_progression_document(arc, arc.progressions[0])
                    
                    changes.append({
                        'arc': arc,
                        'action': 'add'
                    })
                    
                    arc_id_map.append((arc, new_arc_id))
                    added_arc_ids.add(new_arc_id)

                    processed_arcs.add(arc.title)

            logger.info(f"Added or updated {len(arc_id_map)} narrative arcs for {series} {season} {episode} in the vector database.")
            for arc, arc_id in arc_id_map:
                logger.debug(f"Added or updated arc: {arc.title}")
            return arc_id_map

        except Exception as e:
            logger.error(f"An error occurred while adding or updating narrative arcs: {str(e)}")
            logger.error(traceback.format_exc())
            self._rollback_changes(changes)
            raise

    def _rollback_changes(self, changes: List[Dict]):
        """Rollback the changes made to the vector store."""
        logger.info("Rolling back changes due to an error...")
        for change in reversed(changes):
            if change['action'] == 'update':
                self._save_main_arc_document(change['arc'])
                for progression in change['arc'].progressions:
                    self._save_progression_document(change['arc'], progression)
            elif change['action'] == 'add':
                self.collection.delete(ids=[change['arc'].id])
                for progression in change['arc'].progressions:
                    prog_id = self.generate_content_based_id(f"prog_{change['arc'].title}_{progression.series}_{progression.season}_{progression.episode}")
                    self.collection.delete(ids=[prog_id])
        logger.info("Rollback completed.")

    def _save_main_arc_document(self, arc: NarrativeArc):
        """Save or update the main arc document in the vector store."""
        metadata = {
            "title": arc.title,
            "arc_type": arc.arc_type,
            "description": arc.description,
            "episodic": arc.episodic,   
            "characters": ','.join(arc.characters),  # Ensure this is a string
            "series": arc.series,
            "doc_type": "main",
            "id": arc.id
        }
        main_doc = Document(
            page_content=f"{arc.title}\n\n{arc.description}",
            metadata=metadata
        )
        self.collection.add_documents([main_doc], ids=[arc.id])

    def _save_progression_document(self, arc: NarrativeArc, progression: ArcProgression):
        """Save a progression document in the vector store."""
        prog_doc_id = self.generate_content_based_id(f"prog_{arc.title}_{progression.series}_{progression.season}_{progression.episode}")
        metadata = {
            "title": arc.title,
            "arc_type": arc.arc_type,
            "episodic": arc.episodic,
            "characters": ','.join(arc.characters),
            "series": arc.series,
            "doc_type": "progression",
            "id": prog_doc_id,
            "main_arc_id": arc.id,
            "season": progression.season,
            "episode": progression.episode,
            "ordinal_position": progression.ordinal_position
        }
        prog_doc = Document(
            page_content=progression.content,
            metadata=metadata
        )
        logger.debug(f"Saving progression document: {prog_doc_id} with content: {progression.content}")
        self.collection.add_documents([prog_doc], ids=[prog_doc_id])

        # Update the progression object with the main_arc_id
        progression.main_arc_id = arc.id

    def get_arc_by_id(self, arc_id: str) -> Optional[NarrativeArc]:
        """Retrieve a narrative arc by its ID."""
        results = self.collection.get(ids=[arc_id], include=["metadatas", "documents"])
        if results['ids']:
            metadata = results['metadatas'][0]
            content = results['documents'][0]
            title, description = content.split("\n\n", 1)
            
            # Convert characters string back to a list
            metadata['characters'] = metadata['characters'].split(',') if metadata['characters'] else []
            
            # Create a dictionary with only the fields that NarrativeArc expects
            arc_data = {
                "title": title,
                "description": description,
                "arc_type": metadata.get('arc_type'),
                "episodic": metadata.get('episodic'),
                "characters": metadata['characters'],
                "series": metadata.get('series'),
                "id": arc_id
            }
            
            # Create the NarrativeArc instance
            arc = NarrativeArc(**arc_data)
            
            # Add progressions using main_arc_id
            arc.progressions = self._get_arc_progressions(arc_id)
            
            return arc
        return None

    def _get_arc_progressions(self, main_arc_id: str) -> List[ArcProgression]:
        """Retrieve all progressions for a given arc using the main_arc_id."""
        results = self.collection.get(
            where={"$and": [{"doc_type": "progression"}, {"main_arc_id": main_arc_id}]},
            include=["metadatas", "documents"]
        )
        progressions = []
        for metadata, content in zip(results['metadatas'], results['documents']):
            progression_data = {
                "content": content,  # Use the document content directly
                "series": metadata.get('series'),
                "season": metadata.get('season'),
                "episode": metadata.get('episode'),
                "ordinal_position": metadata.get('ordinal_position'),
                "main_arc_id": main_arc_id
            }
            progressions.append(ArcProgression(**progression_data))
        return sorted(progressions, key=lambda p: p.ordinal_position)

    def find_similar_arcs(self, query: str, n_results: int = 5, series: str = None, exclude_ids: Set[str] = set()) -> List[Dict[str, str]]:
        """
        Find similar narrative arcs based on a query, excluding specified IDs.

        Args:
            query (str): The query text to search for similar arcs.
            n_results (int): The number of similar arcs to return. Defaults to 5.
            series (str): The series identifier to filter results.
            exclude_ids (Set[str]): Set of arc IDs to exclude from the search.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing similar arc information and cosine distances.
        """
        filter_criteria = {"$and": [{"doc_type": "main"}]}
        
        if series:
            filter_criteria["$and"].append({"series": series})
        
        logger.debug(f"Query: {query}")
        logger.debug(f"Filter criteria for similarity search: {filter_criteria}")
        
        try:
            results = self.collection.similarity_search_with_score(
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

    def decide_arc_merging(self, new_arc: NarrativeArc, existing_arc: NarrativeArc) -> bool:
        """
        Use cheap LLM to decide whether new_arc and existing_arc represent the same narrative arc.

        Args:
            new_arc (NarrativeArc): The new arc to consider.
            existing_arc (NarrativeArc): The existing arc from the vector store.

        Returns:
            bool: True if the arcs represent the same narrative arc, False otherwise.
        """

        if new_arc.title == existing_arc.title:
            return True


        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert in analyzing narrative arcs in TV series.

            Given the following two narrative arcs:

            Arc A:
            title: {title_a}
            description: {description_a}
            characters: {characters_a}
            Arc Type: {arc_type_a}
            Progression: {progression_a}

            Arc B:
            title: {title_b}
            description: {description_b}
            characters: {characters_b}
            Arc Type: {arc_type_b}
            Progression: {progression_b}

            Do these arcs represent the same narrative arc continuing across episodes, or are they distinct arcs?
            Sometimes these arcs are called with different names but they are meant to be the same.
            Answer with "Yes" if they are the same arc, or "No" if they are different arcs.
            The progressions are always different because they are in different episodes, so you should not consider them for deciding if the arcs are the same.
            Also the descriptions can be a little different while the arcs being the same.

            Provide a brief justification for your answer.

            Your answer should be in JSON format:
            {{
                "same_arc": true or false,
                "justification": "Your brief explanation here"
            }}
            """
        )

        response = self.llm_cheap.invoke(prompt.format_messages(
            title_a=existing_arc.title,
            description_a=existing_arc.description,
            characters_a=", ".join(existing_arc.characters),
            arc_type_a=existing_arc.arc_type,
            progression_a=" | ".join([p.content for p in existing_arc.progressions]),
            title_b=new_arc.title,
            description_b=new_arc.description,
            characters_b=", ".join(new_arc.characters),
            arc_type_b=new_arc.arc_type,
            progression_b=" | ".join([p.content for p in new_arc.progressions]),
        ))

        # Parse response
        try:
            response_json = clean_llm_json_response(response.content)
            if isinstance(response_json, dict):
                return response_json.get("same_arc", False)
            elif isinstance(response_json, list):
                return response_json[0].get("same_arc", False)
                
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return False  # Default to not merging if there's an error

    def find_main_document(self, progression_doc: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Find the main document corresponding to a progression document.

        Args:
            progression_doc (Dict[str, str]): The progression document metadata.

        Returns:
            Optional[Dict[str, str]]: The main document metadata if found, None otherwise.
        """
        # Check if 'main_arc_id' is in the progression_doc, if not, return None
        if 'main_arc_id' not in progression_doc:
            logger.warning("No 'main_arc_id' key found in progression document metadata.")
            return None

        main_docs = self.collection.get(
            ids=[progression_doc['main_arc_id']],
            include=["metadatas"]
        )
        
        if main_docs and main_docs['metadatas']:
            return main_docs['metadatas'][0]
        return None

    def get_narrative_arcs_titles(self) -> List[str]:
        """Get the titles of the narrative arcs from the vector store."""
        return list(set([doc.metadata['title'] for doc in self.get_documents() if doc.metadata['doc_type'] == 'main']))

    def inspect_stored_documents(self, n_samples: int = 5):
        """
        Inspect a sample of stored documents to verify their metadata.

        Args:
            n_samples (int): Number of sample documents to inspect. Defaults to 5.
        """
        results = self.collection.get(
            where={"doc_type": "main"},
            include=["metadatas", "documents"],
            limit=n_samples
        )
        
        logger.info(f"Inspecting {len(results['metadatas'])} sample documents:")
        for metadata, document in zip(results['metadatas'], results['documents']):
            logger.info(f"Document metadata: {metadata}")
            logger.info(f"Document content (first 100 chars): {document[:100]}...")
            logger.info("---")

    def get_all_season_arcs(self, series: str, season: str) -> List[NarrativeArc]:
        """
        Retrieve all narrative arcs for a given series and season.

        Args:
            series (str): The series to retrieve arcs for.
            season (str): The season to retrieve arcs for.

        Returns:
            List[NarrativeArc]: A list of NarrativeArc objects for the given series and season.
        """
        logger.info(f"Retrieving all season arcs for {series} season {season}")

        # Query the collection for all main documents of the given series and season
        results = self.collection.get(
            where={"$and": [{"doc_type": "main"}, {"series": series}]},
            include=["metadatas", "documents"]
        )

        arcs = []
        for metadata, content in zip(results['metadatas'], results['documents']):
            title, description = content.split("\n\n", 1)
            arc = NarrativeArc(
                id=metadata['id'],
                title=title,
                arc_type=metadata['arc_type'],
                description=description,
                episodic=metadata['episodic'],
                characters=metadata['characters'].split(',') if metadata['characters'] else [],
                series=series,
                progressions=[]
            )

            # Fetch progressions for this arc
            progressions = self._get_arc_progressions(arc.id)
            
            # Filter progressions for the current season
            arc.progressions = [prog for prog in progressions if prog.season == season]

            arcs.append(arc)

        logger.info(f"Retrieved {len(arcs)} arcs for {series} season {season}")
        return arcs

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
