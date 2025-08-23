# vector_store_service.py

from typing import List, Dict, Optional, Any, Union
from langchain.schema import Document
from langchain_chroma import Chroma
import logging
import os
import numpy as np
from hdbscan import HDBSCAN

# Use absolute imports to avoid relative import issues
try:
    from ai_models.ai_models import get_embedding_model
    from utils.logger_utils import setup_logging
except ImportError:
    # Fallback for when running from different contexts
    import sys
    current_dir = os.path.dirname(__file__)
    backend_src = os.path.dirname(current_dir)
    if backend_src not in sys.path:
        sys.path.insert(0, backend_src)
    from ai_models.ai_models import get_embedding_model
    from utils.logger_utils import setup_logging

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
            # Ensure we have the same number of documents and IDs
            if len(documents) != len(ids):
                raise ValueError(f"Number of documents ({len(documents)}) must match number of IDs ({len(ids)})")
            
            # Check for and handle duplicate IDs
            unique_ids = list(dict.fromkeys(ids))  # Preserves order while removing duplicates
            
            if len(ids) != len(unique_ids):
                logger.info(f"Found {len(ids) - len(unique_ids)} duplicate IDs when adding documents (handling gracefully), using {len(unique_ids)} unique documents")
                # Create a mapping to get the last document for each unique ID
                id_to_doc = {}
                for doc, id_val in zip(documents, ids):
                    id_to_doc[id_val] = doc  # This will keep the last document for duplicate IDs
                
                # Rebuild the lists with unique IDs
                documents = [id_to_doc[id_val] for id_val in unique_ids]
                ids = unique_ids
            
            self.collection.add_documents(documents, ids=ids)
            logger.info(f"Added {len(documents)} documents to the vector store.")
        except Exception as e:
            logger.error(f"Failed to add documents to the vector store: {e}")
            raise

    def find_similar_arcs(
        self,
        query: str,
        n_results: int = 5,
        series: Optional[str] = None,
        exclude_anthology: bool = False
    ) -> List[Dict[str, Any]]:
        """Find similar arcs (main documents only) to a query."""
        filter_criteria = {"$and": [{"doc_type": "main"}]}
        if series:
            filter_criteria["$and"].append({"series": series})
        if exclude_anthology:
            filter_criteria["$and"].append({"arc_type": {"$ne": "Anthology Arc"}})

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

    def find_similar_progressions(
        self,
        query: str,
        n_results: int = 5,
        series: Optional[str] = None,
        season: Optional[str] = None,
        episode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find similar progressions to a query."""
        filter_criteria = {"$and": [{"doc_type": "progression"}]}
        if series:
            filter_criteria["$and"].append({"series": series})
        if season:
            filter_criteria["$and"].append({"season": season})
        if episode:
            filter_criteria["$and"].append({"episode": episode})

        try:
            results = self.collection.similarity_search_with_score(
                query,
                k=n_results,
                filter=filter_criteria
            )
            return [{"metadata": result[0].metadata, "cosine_distance": result[1], "page_content": result[0].page_content} for result in results]
        except Exception as e:
            logger.error(f"Error during progression similarity search: {e}")
            return []

    def find_similar_events(
        self,
        query: str,
        n_results: int = 5,
        series: Optional[str] = None,
        season: Optional[str] = None,
        episode: Optional[str] = None,
        timestamp_range: Optional[tuple] = None,
        min_confidence: Optional[float] = None,
        narrative_arc_ids: Optional[List[str]] = None,
        exclude_episodes: Optional[List[tuple]] = None,  # [(season, episode), ...]
        excluded_event_ids: Optional[List[str]] = None  # Event IDs to exclude
    ) -> List[Dict[str, Any]]:
        """Find similar events to a query with optional temporal, quality, and narrative arc filtering."""
        # Build filter conditions list
        conditions = [{"doc_type": "event"}]
        
        if series:
            conditions.append({"series": series})
        if season:
            conditions.append({"season": season})
        if episode:
            conditions.append({"episode": episode})
        if timestamp_range:
            start_time, end_time = timestamp_range
            conditions.append({"start_timestamp": {"$gte": start_time}})
            conditions.append({"end_timestamp": {"$lte": end_time}})
        if min_confidence:
            conditions.append({"confidence_score": {"$gte": min_confidence}})
        if narrative_arc_ids:
            # Filter by narrative arc IDs - events/progressions use 'main_arc_id', main docs use 'id'
            conditions.append({
                "$or": [
                    {"main_arc_id": {"$in": narrative_arc_ids}},  # For progressions and events
                    {"id": {"$in": narrative_arc_ids}}            # For main arc documents
                ]
            })
        
        # Exclude specific event IDs using $nin operator
        if excluded_event_ids:
            conditions.append({"id": {"$nin": excluded_event_ids}})

        # Build final filter - only use $and if we have multiple conditions
        if len(conditions) == 1:
            filter_criteria = conditions[0]
        else:
            filter_criteria = {"$and": conditions}

        try:
            # Get initial results from vector search
            results = self.collection.similarity_search_with_score(
                query,
                k=n_results * 3 if exclude_episodes else n_results,  # Get more if we need to filter
                filter=filter_criteria
            )
            
            # Apply episode exclusion post-search if needed (ChromaDB doesn't support complex NOT queries)
            if exclude_episodes:
                filtered_results = []
                for result in results:
                    metadata = result[0].metadata
                    episode_season = metadata.get('season')
                    episode_episode = metadata.get('episode')
                    
                    # Check if this episode should be excluded
                    should_exclude = any(
                        episode_season == exclude_season and episode_episode == exclude_episode
                        for exclude_season, exclude_episode in exclude_episodes
                    )
                    
                    if not should_exclude:
                        filtered_results.append(result)
                        
                    # Stop when we have enough results
                    if len(filtered_results) >= n_results:
                        break
                
                results = filtered_results
            
            return [{"metadata": result[0].metadata, "cosine_distance": result[1], "page_content": result[0].page_content} for result in results]
        except Exception as e:
            logger.error(f"Error during event similarity search: {e}")
            return []

    def get_events_by_timestamp_range(
        self,
        series: str,
        season: str,
        episode: str,
        start_time: float,
        end_time: float,
        n_results: int = 50
    ) -> List[Dict[str, Any]]:
        """Get events within a specific timestamp range, ordered by start time."""
        filter_criteria = {
            "$and": [
                {"doc_type": "event"},
                {"series": series},
                {"season": season},
                {"episode": episode},
                {"start_timestamp": {"$gte": start_time}},
                {"end_timestamp": {"$lte": end_time}}
            ]
        }

        try:
            # Get all events in range
            results = self.collection.get(
                where=filter_criteria,
                include=["metadatas", "documents"],
                limit=n_results
            )
            
            # Convert to consistent format and sort by timestamp
            events = []
            if results.get('metadatas') and results.get('documents'):
                for metadata, content in zip(results['metadatas'], results['documents']):
                    events.append({
                        "metadata": metadata,
                        "page_content": content,
                        "cosine_distance": 0.0  # No similarity score for direct retrieval
                    })
                
                # Sort by start_timestamp
                events.sort(key=lambda x: x['metadata'].get('start_timestamp', 0))
            
            return events
        except Exception as e:
            logger.error(f"Error retrieving events by timestamp range: {e}")
            return []

    def delete_documents_by_arc(self, arc_id: str):
        """Delete all documents related to a specific arc (main, progressions, and events)."""
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
            # NEW: Retrieve event document IDs where metadata 'main_arc_id' matches arc_id
            event_docs_response = self.collection.get(
                where={"$and": [{"doc_type": "event"}, {"main_arc_id": arc_id}]},
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

            # NEW: Safely extract 'ids' from event_docs_response
            event_ids = event_docs_response.get('ids', [])
            ids_to_delete.extend(event_ids)

            # Ensure unique IDs to prevent duplicate deletion errors
            unique_ids_to_delete = list(dict.fromkeys(ids_to_delete))  # Preserves order while removing duplicates
            
            if len(ids_to_delete) != len(unique_ids_to_delete):
                logger.info(f"Found {len(ids_to_delete) - len(unique_ids_to_delete)} overlapping IDs when deleting arc {arc_id} (expected due to document metadata overlap), using {len(unique_ids_to_delete)} unique IDs")

            if unique_ids_to_delete:
                self.collection.delete(ids=unique_ids_to_delete)
                logger.info(f"Deleted {len(unique_ids_to_delete)} documents for arc ID {arc_id} (main: {len(main_ids)}, progressions: {len(prog_ids)}, events: {len(event_ids)}).")
            else:
                logger.info(f"No documents found for arc ID {arc_id}.")

        except Exception as e:
            logger.error(f"Error deleting documents for arc ID {arc_id}: {e}")
            raise

    def delete_events_by_progression(self, progression_id: str):
        """Delete all event documents related to a specific progression."""
        try:
            event_docs_response = self.collection.get(
                where={"$and": [{"doc_type": "event"}, {"progression_id": progression_id}]},
                include=["metadatas"]
            )

            event_ids = event_docs_response.get('ids', [])
            if event_ids:
                self.collection.delete(ids=event_ids)
                logger.info(f"Deleted {len(event_ids)} event documents for progression ID {progression_id}.")
            else:
                logger.info(f"No event documents found for progression ID {progression_id}.")

        except Exception as e:
            logger.error(f"Error deleting event documents for progression ID {progression_id}: {e}")
            
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

            # Validate results structure
            if not results or 'ids' not in results:
                logger.warning(f"No results found for series: {series}")
                return []

            # Check array lengths for consistency
            ids_len = len(results['ids'])
            docs_len = len(results.get('documents', []))
            metas_len = len(results.get('metadatas', []))
            
            if docs_len != ids_len or metas_len != ids_len:
                logger.error(f"Array length mismatch: ids={ids_len}, documents={docs_len}, metadatas={metas_len}")
                return []

            # If embeddings requested, check their length too
            if include_embeddings and 'embeddings' in results:
                embeddings_len = len(results['embeddings'])
                if embeddings_len != ids_len:
                    logger.error(f"Embeddings length mismatch: embeddings={embeddings_len}, ids={ids_len}")
                    # Continue without embeddings rather than failing
                    include_embeddings = False
                    logger.warning("Disabling embeddings due to length mismatch")

            # Format results with safe indexing
            documents = []
            for i in range(ids_len):
                try:
                    doc = {
                        'id': results['ids'][i],
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i],
                    }
                    if include_embeddings and 'embeddings' in results and i < len(results['embeddings']):
                        doc['embedding'] = results['embeddings'][i]
                    documents.append(doc)
                except IndexError as ie:
                    logger.error(f"Index error at position {i}: {ie}")
                    break  # Stop processing if we hit an index error

            logger.info(f"Successfully retrieved {len(documents)} documents for series {series}")
            return documents

        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []

    def find_similar_documents(
        self,
        query: str,
        series: str,
        n_results: int = 10
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

    def calculate_arcs_cosine_distances(
        self,
        arc_ids: List[str]
    ) -> Dict[str, Union[str, float]]:
        """Calculate cosine distances between selected arcs."""
        try:
            # Get documents by IDs
            results = self.collection.get(
                ids=arc_ids,
                include=['metadatas', 'embeddings']
            )

            if not results or 'embeddings' not in results or len(results['embeddings']) != 2:
                raise ValueError("Could not find embeddings for both arcs")

            # Get embeddings as numpy arrays
            embedding1 = np.array(results['embeddings'][0])
            embedding2 = np.array(results['embeddings'][1])

            # Calculate dot product
            dot_product = np.dot(embedding1, embedding2)
            
            # Calculate norms
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            # Calculate cosine distance
            if norm1 == 0 or norm2 == 0:
                cosine_distance = 1.0  # Maximum distance for zero vectors
            else:
                cosine_similarity = dot_product / (norm1 * norm2)
                # Ensure the similarity is within [-1, 1] to avoid numerical errors
                cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
                cosine_distance = 1.0 - cosine_similarity

            return {
                "arc1": {
                    "id": results['ids'][0],
                    "title": results['metadatas'][0].get('title', ''),
                    "type": results['metadatas'][0].get('arc_type', '')
                },
                "arc2": {
                    "id": results['ids'][1],
                    "title": results['metadatas'][1].get('title', ''),
                    "type": results['metadatas'][1].get('arc_type', '')
                },
                "distance": float(cosine_distance)  # Convert numpy float to Python float
            }

        except Exception as e:
            logger.error(f"Error calculating cosine distances: {e}")
            raise

    def find_similar_arcs_clusters(
        self,
        series: str,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        cluster_selection_epsilon: float = 0.3
    ) -> List[Dict]:
        """Find clusters of similar arcs using HDBSCAN clustering."""
        try:
            # Get all main arcs with their embeddings
            results = self.collection.get(
                where={"$and": [{"series": series}, {"doc_type": "main"}]},
                include=['metadatas', 'embeddings']
            )

            if not results or 'embeddings' not in results or len(results['embeddings']) < 2:
                return []

            # Convert embeddings to numpy array
            embeddings = np.array(results['embeddings'])
            
            # Option 1: Use embeddings directly (recommended - no warnings)
            # This allows HDBSCAN to work in the original vector space
            try:
                # Normalize embeddings for cosine similarity
                # When embeddings are normalized, euclidean distance = 2 * (1 - cosine_similarity)
                normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                clusterer = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    metric='euclidean',  # Use euclidean on normalized embeddings (equivalent to cosine)
                    core_dist_n_jobs=-1,
                    cluster_selection_method='leaf',
                    prediction_data=True,  # Now this works
                    allow_single_cluster=True
                )
                
                # Fit using the normalized embeddings
                clusterer.fit(normalized_embeddings)
                labels = clusterer.labels_
                probabilities = clusterer.probabilities_
                
            except Exception as e:
                logger.warning(f"Direct embedding clustering failed, falling back to distance matrix: {e}")
                
                # Option 2: Fallback to precomputed distance matrix (original approach)
                # Pre-compute distance matrix using cosine distance
                distance_matrix = np.zeros((len(embeddings), len(embeddings)))
                for i in range(len(embeddings)):
                    for j in range(len(embeddings)):
                        if i != j:
                            # Calculate cosine distance
                            dot_product = np.dot(embeddings[i], embeddings[j])
                            norm_i = np.linalg.norm(embeddings[i])
                            norm_j = np.linalg.norm(embeddings[j])
                            if norm_i == 0 or norm_j == 0:
                                distance_matrix[i, j] = 1.0
                            else:
                                cosine_sim = dot_product / (norm_i * norm_j)
                                # Ensure similarity is within [-1, 1]
                                cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
                                distance_matrix[i, j] = 1.0 - cosine_sim

                # Use precomputed distances (no prediction data available)
                clusterer = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    metric='precomputed',
                    core_dist_n_jobs=-1,
                    cluster_selection_method='leaf',
                    prediction_data=False,  # Must be False for precomputed distances
                    allow_single_cluster=True
                )
                
                # Fit using the distance matrix
                clusterer.fit(distance_matrix)
                labels = clusterer.labels_
                
                # Probabilities are not available when using precomputed distances
                probabilities = [1.0 if label != -1 else 0.0 for label in labels]
                logger.info("Using fallback probabilities as prediction data not available with precomputed distances")

            # Group arcs by cluster
            clusters = {}
            for i, (label, prob) in enumerate(zip(labels, probabilities)):
                if label != -1:  # Ignore noise points
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append({
                        "id": results['ids'][i],
                        "title": results['metadatas'][i].get('title', ''),
                        "type": results['metadatas'][i].get('arc_type', ''),
                        "metadata": results['metadatas'][i],
                        "cluster_probability": float(prob)
                    })

            # Format results
            similar_groups = []
            for cluster_id, arcs in clusters.items():
                cluster_indices = [results['ids'].index(arc['id']) for arc in arcs]
                distances = []
                
                # Calculate average distance between arcs in cluster
                if 'distance_matrix' in locals():
                    # Use pre-computed distance matrix (fallback approach)
                    for i, idx1 in enumerate(cluster_indices):
                        for j, idx2 in enumerate(cluster_indices[i+1:], i+1):
                            distances.append(distance_matrix[idx1, idx2])
                else:
                    # Calculate cosine distances from embeddings directly (preferred approach)
                    for i, idx1 in enumerate(cluster_indices):
                        for j, idx2 in enumerate(cluster_indices[i+1:], i+1):
                            # Calculate cosine distance between embeddings
                            emb1, emb2 = embeddings[idx1], embeddings[idx2]
                            dot_product = np.dot(emb1, emb2)
                            norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
                            if norm1 == 0 or norm2 == 0:
                                distances.append(1.0)
                            else:
                                cosine_sim = dot_product / (norm1 * norm2)
                                cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
                                distances.append(1.0 - cosine_sim)
                
                avg_distance = np.mean(distances) if distances else 0
                avg_probability = np.mean([arc['cluster_probability'] for arc in arcs])
                
                # Handle cluster persistence - may not be available with precomputed distances
                try:
                    cluster_persistence = clusterer.cluster_persistence_[cluster_id]
                except (AttributeError, IndexError):
                    # Fallback: estimate persistence based on cluster size and avg distance
                    cluster_persistence = max(0.1, 1.0 - avg_distance)

                similar_groups.append({
                    "cluster_id": int(cluster_id),
                    "arcs": arcs,
                    "average_distance": float(avg_distance),
                    "size": len(arcs),
                    "average_probability": float(avg_probability),
                    "cluster_persistence": float(cluster_persistence)
                })

            # Sort clusters by size and average probability
            similar_groups.sort(key=lambda x: (x['size'], x['average_probability']), reverse=True)
            
            return similar_groups

        except Exception as e:
            logger.error(f"Error finding similar arcs clusters: {e}")
            raise



class FaceEmbeddingVectorStore:
    """Dedicated vector store service for face embeddings across episodes."""

    def __init__(self, collection_name: str = "face_embeddings", persist_directory: Optional[str] = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or os.getenv("PERSIST_DIRECTORY", "./vector_store")
        self.embedding_model = get_embedding_model()
        self.collection = self._initialize_collection()

    def _initialize_collection(self) -> Chroma:
        """Initialize the face embeddings collection."""
        return Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_face_embeddings(
        self,
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]],
        ids: List[str]
    ) -> None:
        """
        Add face embeddings with metadata to the vector store.
        
        Args:
            embeddings: List of face embedding vectors
            metadata_list: List of metadata dicts containing face info
            ids: List of unique IDs for each face embedding
        """
        try:
            # Convert embeddings to documents (Chroma expects text content)
            documents = []
            for i, metadata in enumerate(metadata_list):
                # Create a text representation of the face metadata for indexing
                doc_text = f"Face from {metadata.get('series', '')} {metadata.get('season', '')} {metadata.get('episode', '')} " \
                          f"at {metadata.get('timestamp', '')} - Speaker: {metadata.get('speaker', 'unknown')} " \
                          f"Confidence: {metadata.get('speaker_confidence', 0)}"
                documents.append(doc_text)
            
            # For face embeddings, we have pre-computed embeddings, so we use the underlying client
            # since LangChain's add_documents would re-compute embeddings
            try:
                # Access the underlying ChromaDB collection
                chroma_collection = self.collection._collection
                
                # Use ChromaDB's native API with pre-computed embeddings
                chroma_collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadata_list,
                    ids=ids
                )
            except AttributeError:
                # Fallback: If the structure changed, try upsert
                try:
                    chroma_collection = self.collection._collection
                    chroma_collection.upsert(
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadata_list,
                        ids=ids
                    )
                except Exception as e:
                    logger.error(f"ChromaDB API error: {e}")
                    # Final fallback: use LangChain method (will re-compute embeddings)
                    from langchain_core.documents import Document
                    langchain_docs = []
                    for i, doc_text in enumerate(documents):
                        doc = Document(
                            page_content=doc_text,
                            metadata=metadata_list[i]
                        )
                        langchain_docs.append(doc)
                    
                    self.collection.add_documents(
                        documents=langchain_docs,
                        ids=ids
                    )
            
            logger.info(f"✅ Added {len(embeddings)} face embeddings to vector store")
            
        except Exception as e:
            logger.error(f"❌ Failed to add face embeddings: {e}")
            raise

    def find_similar_faces(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        series: Optional[str] = None,
        min_speaker_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find similar faces based on embedding similarity.
        
        Args:
            query_embedding: Face embedding to search for
            n_results: Maximum number of results to return
            series: Optional filter by series
            min_speaker_confidence: Minimum speaker confidence threshold
            
        Returns:
            List of similar faces with metadata
        """
        try:
            # Build filter criteria
            filter_criteria = {}
            if series:
                filter_criteria["series"] = series
            if min_speaker_confidence > 0:
                filter_criteria["speaker_confidence"] = {"$gte": min_speaker_confidence}
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_criteria if filter_criteria else None
            )
            
            # Format results
            similar_faces = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    similar_faces.append({
                        "id": results["ids"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if results.get("distances") else None
                    })
            
            logger.info(f"Found {len(similar_faces)} similar faces")
            return similar_faces
            
        except Exception as e:
            logger.error(f"❌ Error finding similar faces: {e}")
            return []

    def get_faces_by_speaker(
        self,
        speaker_name: str,
        series: Optional[str] = None,
        min_confidence: float = 95.0
    ) -> List[Dict[str, Any]]:
        """
        Get all faces associated with a specific speaker.
        
        Args:
            speaker_name: Name of the speaker
            series: Optional filter by series
            min_confidence: Minimum speaker confidence threshold
            
        Returns:
            List of faces for the speaker
        """
        try:
            filter_criteria = {
                "speaker": speaker_name,
                "speaker_confidence": {"$gte": min_confidence}
            }
            if series:
                filter_criteria["series"] = series
            
            results = self.collection.query(
                query_texts=[f"Speaker: {speaker_name}"],
                n_results=1000,  # Get all matches
                where=filter_criteria
            )
            
            speaker_faces = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    speaker_faces.append({
                        "id": results["ids"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if results.get("distances") else None
                    })
            
            logger.info(f"Found {len(speaker_faces)} faces for speaker '{speaker_name}'")
            return speaker_faces
            
        except Exception as e:
            logger.error(f"❌ Error getting faces for speaker '{speaker_name}': {e}")
            return []

    def update_speaker_associations(
        self,
        face_ids: List[str],
        speaker_name: str,
        confidence: float
    ) -> None:
        """
        Update speaker associations for a list of face IDs.
        
        Args:
            face_ids: List of face IDs to update
            speaker_name: New speaker name
            confidence: Confidence score for the association
        """
        try:
            # Get current metadata for these faces
            results = self.collection.get(ids=face_ids, include=["metadatas"])
            
            if not results["metadatas"]:
                logger.warning(f"No faces found for IDs: {face_ids}")
                return
            
            # Update metadata
            updated_metadatas = []
            for metadata in results["metadatas"]:
                metadata["speaker"] = speaker_name
                metadata["speaker_confidence"] = confidence
                updated_metadatas.append(metadata)
            
            # Update in vector store
            self.collection.update(
                ids=face_ids,
                metadatas=updated_metadatas
            )
            
            logger.info(f"✅ Updated speaker associations for {len(face_ids)} faces")
            
        except Exception as e:
            logger.error(f"❌ Error updating speaker associations: {e}")
            raise

    def delete_episode_faces(self, series: str, season: str, episode: str) -> None:
        """Delete all faces from a specific episode."""
        try:
            filter_criteria = {
                "series": series,
                "season": season,
                "episode": episode
            }
            
            # Get all faces for this episode
            results = self.collection.query(
                query_texts=[""],
                n_results=10000,
                where=filter_criteria
            )
            
            if results["ids"] and results["ids"][0]:
                self.collection.delete(ids=results["ids"][0])
                logger.info(f"🗑️ Deleted {len(results['ids'][0])} faces from {series} {season} {episode}")
            else:
                logger.info(f"ℹ️ No faces found to delete for {series} {season} {episode}")
                
        except Exception as e:
            logger.error(f"❌ Error deleting episode faces: {e}")
            raise
        