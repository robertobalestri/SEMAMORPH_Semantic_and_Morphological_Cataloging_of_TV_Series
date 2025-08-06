"""
Face embedding vector store using ChromaDB for SEMAMORPH speaker identification.
Stores face embeddings with metadata for similarity search and clustering.
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

from ..utils.logger_utils import setup_logging

logger = setup_logging(__name__)

class FaceEmbeddingVectorStore:
    """ChromaDB-based vector store for face embeddings."""
    
    def __init__(self, path_handler):
        self.path_handler = path_handler
        self._client = None
        self._default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    def get_chroma_client(self):
        """Initialize and return a persistent ChromaDB client."""
        if self._client is None:
            chroma_db_path = self.path_handler.get_chroma_db_path()
            logger.info(f"💾 Initializing ChromaDB client at: {chroma_db_path}")
            os.makedirs(chroma_db_path, exist_ok=True)
            try:
                self._client = chromadb.PersistentClient(path=chroma_db_path)
            except Exception as e:
                logger.error(f"❌ Failed to initialize ChromaDB PersistentClient at {chroma_db_path}: {e}")
                logger.warning("❌ Falling back to in-memory client (data will NOT be persistent across runs!)")
                try:
                    self._client = chromadb.Client()  # In-memory fallback
                except Exception as fallback_e:
                    logger.error(f"❌ Failed to initialize even in-memory ChromaDB client: {fallback_e}")
                    raise RuntimeError("Could not initialize ChromaDB") from fallback_e
        return self._client

    def get_face_collection(self, collection_name="dialogue_faces", embedding_model_name="Facenet512"):
        """Get or create the ChromaDB collection for face embeddings."""
        client = self.get_chroma_client()
        full_collection_name = f"{collection_name}_{embedding_model_name.replace('-', '_').lower()}"
        logger.debug(f"📂 Accessing ChromaDB collection: {full_collection_name}")
        try:
            collection = client.get_or_create_collection(
                name=full_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            return collection
        except Exception as e:
            logger.error(f"❌ Failed to get or create ChromaDB collection '{full_collection_name}': {e}")
            raise

    def save_embeddings_to_vector_store(
        self, 
        df_faces: pd.DataFrame, 
        embedding_model_name: str = "Facenet512"
    ) -> None:
        """
        Save face embeddings to ChromaDB vector store.
        
        Args:
            df_faces: DataFrame with face data and embeddings
            embedding_model_name: Name of the embedding model used
        """
        if df_faces.empty or 'embedding' not in df_faces.columns:
            logger.warning("⚠️ No face embeddings to save to vector store")
            return
        
        logger.info(f"💾 Saving {len(df_faces)} face embeddings to vector store")
        
        try:
            collection = self.get_face_collection(embedding_model_name=embedding_model_name)
            
            # Get context from path_handler
            series = self.path_handler.get_series()
            season = self.path_handler.get_season()
            episode = self.path_handler.get_episode()
            episode_code = self.path_handler.get_episode_code()
            
            # Filter valid embeddings
            valid_mask = df_faces['embedding'].apply(
                lambda emb: isinstance(emb, np.ndarray) and emb.ndim == 1 and emb.size > 0
            )
            df_valid = df_faces[valid_mask].copy()
            
            if df_valid.empty:
                logger.warning("⚠️ No valid embeddings found to save")
                return

            # Check for existing embeddings
            image_paths = df_valid['image_path'].tolist()
            try:
                existing_in_db = collection.get(ids=image_paths, include=[])
                existing_ids_set = set(existing_in_db['ids'])
                logger.debug(f"Found {len(existing_ids_set)} existing embeddings in vector store")
            except Exception as e:
                logger.warning(f"⚠️ Error checking existing IDs in ChromaDB: {e}")
                existing_ids_set = set()
            
            # Filter to new embeddings only
            new_faces_df = df_valid[~df_valid['image_path'].isin(existing_ids_set)]
            
            if new_faces_df.empty:
                logger.info("✅ All embeddings already exist in vector store")
                return
            
            logger.info(f"Adding {len(new_faces_df)} new embeddings to vector store")
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            
            for _, row in new_faces_df.iterrows():
                # Normalize embedding
                emb = row['embedding']
                norm = np.linalg.norm(emb)
                if norm > 1e-9:
                    normalized_emb = (emb / norm).astype(np.float32)
                else:
                    continue  # Skip zero embeddings
                
                # Prepare metadata
                metadata = {
                    'series': series,
                    'season': season,
                    'episode': episode,
                    'episode_code': episode_code,
                    'dialogue_index': int(row.get('dialogue_index', -1)),
                    'face_index': int(row.get('face_index', 0)),
                    'timestamp_seconds': float(row.get('timestamp_seconds', 0.0)),
                    'speaker': str(row.get('speaker', '')),
                    'speaker_confidence': float(row.get('speaker_confidence', 0.0)),
                    'detection_confidence': float(row.get('detection_confidence', 0.0)),
                    'blur_score': float(row.get('blur_score', 0.0))
                }
                
                # Add cluster info if available
                if 'face_id' in row:
                    metadata['face_id'] = int(row['face_id'])
                if 'cluster_id' in row:
                    metadata['cluster_id'] = int(row['cluster_id'])
                
                ids.append(str(row['image_path']))
                embeddings.append(normalized_emb.tolist())
                metadatas.append(metadata)
            
            if ids:
                # Batch add to ChromaDB
                batch_size = 100
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i+batch_size]
                    batch_embeddings = embeddings[i:i+batch_size]
                    batch_metadatas = metadatas[i:i+batch_size]
                    
                    collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas
                    )
                
                logger.info(f"✅ Successfully added {len(ids)} embeddings to vector store")
            
        except Exception as e:
            logger.error(f"❌ Error saving embeddings to vector store: {e}")
    
    def find_similar_faces(
        self, 
        query_embedding: List[float], 
        n_results: int = 10,
        series: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find similar faces in the vector store.
        
        Args:
            query_embedding: Query face embedding as list
            n_results: Number of results to return
            series: Filter by series (optional)
            min_confidence: Minimum detection confidence
            
        Returns:
            List of similar face records with similarity scores
        """
        try:
            collection = self.get_face_collection()
            
            # Build where clause properly for ChromaDB
            conditions = []
            if series:
                conditions.append({'series': series})
            if min_confidence > 0:
                conditions.append({'detection_confidence': {"$gte": min_confidence}})
            
            # ChromaDB requires explicit $and for multiple conditions
            where_clause = None
            if len(conditions) == 1:
                where_clause = conditions[0]
            elif len(conditions) > 1:
                where_clause = {"$and": conditions}
            
            # Query ChromaDB
            if where_clause:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_clause,
                    include=['metadatas', 'distances']
                )
            else:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=['metadatas', 'distances']
                )
            
            # Format results
            similar_faces = []
            if results and results['ids'] and results['ids'][0]:
                for i, (face_id, metadata, distance) in enumerate(zip(
                    results['ids'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    similarity = 1.0 - distance  # Convert distance to similarity
                    face_record = {
                        'id': face_id,
                        'similarity': similarity,
                        'metadata': metadata
                    }
                    similar_faces.append(face_record)
            
            return similar_faces
            
        except Exception as e:
            logger.error(f"❌ Error finding similar faces: {e}")
            return []
    
    def get_faces_by_speaker(
        self, 
        speaker_name: str, 
        series: Optional[str] = None,
        min_confidence: float = 90.0,
        n_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get faces associated with a specific speaker.
        
        Args:
            speaker_name: Name of the speaker
            series: Filter by series (optional)  
            min_confidence: Minimum speaker confidence
            n_results: Maximum number of results
            
        Returns:
            List of face records for the speaker
        """
        try:
            collection = self.get_face_collection()
            
            # Build where clause properly for ChromaDB
            conditions = [
                {'speaker': speaker_name},
                {'speaker_confidence': {"$gte": min_confidence}}
            ]
            if series:
                conditions.append({'series': series})
            
            # ChromaDB requires explicit $and for multiple conditions
            where_clause = {"$and": conditions} if len(conditions) > 1 else conditions[0]
            
            # Get faces for this speaker
            results = collection.get(
                where=where_clause,
                limit=n_results,
                include=['metadatas', 'embeddings']
            )
            
            # Format results
            speaker_faces = []
            if results and results['ids']:
                for i, (face_id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
                    face_record = {
                        'id': face_id,
                        'metadata': metadata
                    }
                    if results.get('embeddings') and i < len(results['embeddings']):
                        face_record['embedding'] = results['embeddings'][i]
                    speaker_faces.append(face_record)
            
            return speaker_faces
            
        except Exception as e:
            logger.error(f"❌ Error getting faces for speaker {speaker_name}: {e}")
            return []
    
    def get_collection_stats(self, embedding_model_name: str = "Facenet512") -> Dict[str, Any]:
        """Get statistics about the face collection."""
        try:
            collection = self.get_face_collection(embedding_model_name=embedding_model_name)
            count = collection.count()
            
            stats = {
                'total_faces': count,
                'collection_name': collection.name
            }
            
            if count > 0:
                # Get sample to analyze metadata
                sample = collection.get(limit=min(100, count), include=['metadatas'])
                if sample and sample['metadatas']:
                    series_set = set()
                    speakers_set = set()
                    episodes_set = set()
                    
                    for metadata in sample['metadatas']:
                        if metadata:
                            series_set.add(metadata.get('series', ''))
                            speakers_set.add(metadata.get('speaker', ''))
                            episodes_set.add(metadata.get('episode_code', ''))
                    
                    stats.update({
                        'unique_series': len(series_set),
                        'unique_speakers': len(speakers_set),
                        'unique_episodes': len(episodes_set)
                    })
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {e}")
            return {'error': str(e)}

    def reset_collection(self, embedding_model_name: str = "Facenet512") -> bool:
        """Reset (delete and recreate) the face collection."""
        try:
            client = self.get_chroma_client()
            collection_name = f"dialogue_faces_{embedding_model_name.replace('-', '_').lower()}"
            
            # Delete existing collection
            try:
                client.delete_collection(name=collection_name)
                logger.info(f"🗑️ Deleted existing collection: {collection_name}")
            except Exception:
                logger.debug(f"Collection {collection_name} didn't exist or couldn't be deleted")
            
            # Create new collection
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✅ Created new collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error resetting collection: {e}")
            return False
