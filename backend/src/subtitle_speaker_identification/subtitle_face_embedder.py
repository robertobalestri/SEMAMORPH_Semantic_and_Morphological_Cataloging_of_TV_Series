"""
Face embedding generation for subtitle-based face tracking.
Generates embeddings for face crops and manages persistence.
"""
import os
import pickle
import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path

# Configure TensorFlow before importing DeepFace
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

# Configure GPU memory growth to prevent CUDA out-of-memory errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")

from deepface import DeepFace
from tqdm import tqdm

from ..utils.logger_utils import setup_logging

logger = setup_logging(__name__)

class SubtitleFaceEmbedder:
    """Generates and manages face embeddings for subtitle-based face tracking."""
    
    def __init__(self, path_handler):
        self.path_handler = path_handler
    
    def generate_embeddings(
        self,
        df_faces: pd.DataFrame,
        model: str = "Facenet512",
        force_regenerate: bool = False
    ) -> pd.DataFrame:
        """
        Generate embeddings for face crops.
        
        Args:
            df_faces: DataFrame with face metadata
            model: DeepFace embedding model
            force_regenerate: If True, regenerate existing embeddings
            
        Returns:
            DataFrame with embeddings added
        """
        if df_faces.empty:
            logger.warning("⚠️ No faces provided for embedding generation")
            return df_faces
        
        logger.info(f"🧠 Generating embeddings for {len(df_faces)} faces using {model}")
        
        # Setup directories
        embeddings_dir = self.path_handler.get_dialogue_embeddings_dir()
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Pre-load model
        try:
            _ = DeepFace.represent(" ", model_name=model, enforce_detection=False, detector_backend='skip')
            logger.info(f"✅ Pre-loaded embedding model: {model}")
        except Exception as e:
            logger.warning(f"⚠️ Could not pre-load model {model}: {e}")
        
        embeddings_list = []
        embedding_paths_list = []
        failed_count = 0
        
        for idx, row in tqdm(df_faces.iterrows(), total=len(df_faces), desc="Generating embeddings"):
            image_path = row['image_path']
            dialogue_idx = row['dialogue_index']
            face_idx = row['face_index']
            
            # Create embedding file path
            embedding_filename = f"dialogue_{dialogue_idx:04d}_face_{face_idx:02d}_embedding.pkl"
            embedding_path = os.path.join(embeddings_dir, embedding_filename)
            
            # Check if embedding already exists
            if not force_regenerate and os.path.exists(embedding_path):
                try:
                    embedding = self._load_embedding_from_pickle(embedding_path)
                    if embedding is not None:
                        embeddings_list.append(embedding)
                        embedding_paths_list.append(embedding_path)
                        continue
                except Exception as e:
                    logger.debug(f"Error loading existing embedding {embedding_path}: {e}")
            
            # Generate new embedding
            try:
                if not os.path.exists(image_path):
                    logger.debug(f"⚠️ Image not found: {image_path}")
                    embeddings_list.append(None)
                    embedding_paths_list.append(None)
                    failed_count += 1
                    continue
                
                # Generate embedding
                embedding_result = DeepFace.represent(
                    img_path=image_path,
                    model_name=model,
                    enforce_detection=False,
                    detector_backend='skip'  # Skip detection since face is already cropped
                )
                
                if embedding_result and len(embedding_result) > 0:
                    embedding = np.array(embedding_result[0]['embedding'], dtype=np.float32)
                    
                    # Save embedding as pickle
                    with open(embedding_path, 'wb') as f:
                        pickle.dump(embedding, f)
                    
                    embeddings_list.append(embedding)
                    embedding_paths_list.append(embedding_path)
                    
                else:
                    logger.debug(f"⚠️ No embedding generated for {image_path}")
                    embeddings_list.append(None)
                    embedding_paths_list.append(None)
                    failed_count += 1
                
            except Exception as e:
                logger.debug(f"⚠️ Error generating embedding for {image_path}: {e}")
                embeddings_list.append(None)
                embedding_paths_list.append(None)
                failed_count += 1
        
        # Add embeddings to DataFrame
        df_faces = df_faces.copy()
        df_faces['embedding'] = embeddings_list
        df_faces['embedding_path'] = embedding_paths_list
        
        # Filter out failed embeddings
        valid_mask = df_faces['embedding'].notna()
        df_valid = df_faces[valid_mask].copy()
        
        success_count = len(df_valid)
        logger.info(f"✅ Embedding generation complete: {success_count} successful, {failed_count} failed")
        
        return df_valid
    
    def _load_embedding_from_pickle(self, pickle_path: str) -> Optional[np.ndarray]:
        """Load embedding from pickle file."""
        try:
            with open(pickle_path, 'rb') as f:
                embedding = pickle.load(f)
                if isinstance(embedding, np.ndarray) and embedding.ndim == 1:
                    return embedding.astype(np.float32)
                else:
                    logger.debug(f"Invalid embedding format in {pickle_path}")
                    return None
        except Exception as e:
            logger.debug(f"Error loading pickle {pickle_path}: {e}")
            return None
    
    def save_embeddings_to_vector_store(
        self,
        df_faces: pd.DataFrame,
        face_vector_store
    ) -> None:
        """
        Save face embeddings to vector store.
        
        Args:
            df_faces: DataFrame with face data and embeddings
            face_vector_store: FaceEmbeddingVectorStore instance
        """
        if df_faces.empty or 'embedding' not in df_faces.columns:
            logger.warning("⚠️ No valid embeddings to save to vector store")
            return
        
        # Filter valid embeddings
        valid_mask = df_faces['embedding'].notna()
        df_valid = df_faces[valid_mask].copy()
        
        if df_valid.empty:
            logger.warning("⚠️ No valid embeddings found")
            return
        
        logger.info(f"💾 Saving {len(df_valid)} embeddings to vector store")
        
        # Prepare data for vector store
        embeddings = []
        metadata_list = []
        ids = []
        
        for idx, row in df_valid.iterrows():
            embedding = row['embedding']
            
            # Convert numpy array to list for JSON serialization
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            else:
                logger.warning(f"Skipping invalid embedding at index {idx}")
                continue
            
            # Create unique ID
            face_id = f"{row['episode_code']}_dialogue_{row['dialogue_index']:04d}_face_{row['face_index']:02d}"
            
            # Create metadata - handle missing columns gracefully
            metadata = {
                'series': str(row.get('series', '')),
                'season': str(row.get('season', '')), 
                'episode': str(row.get('episode', '')),
                'episode_code': str(row.get('episode_code', '')),
                'dialogue_index': int(row['dialogue_index']) if pd.notna(row.get('dialogue_index')) else 0,
                'face_index': int(row['face_index']) if pd.notna(row.get('face_index')) else 0,
                'timestamp_seconds': float(row['timestamp_seconds']) if row.get('timestamp_seconds') is not None else 0.0,
                'dialogue_text': str(row.get('dialogue_text', '')),
                'speaker': str(row.get('speaker', '')),
                'speaker_confidence': float(row.get('speaker_confidence', 0.0)) if row.get('speaker_confidence') is not None else 0.0,
                'scene_number': int(row.get('scene_number', 0)) if row.get('scene_number') is not None and pd.notna(row.get('scene_number')) else 0,
                'detection_confidence': float(row.get('detection_confidence', 0.0)) if row.get('detection_confidence') is not None else 0.0,
                'blur_score': float(row.get('blur_score', 0.0)) if row.get('blur_score') is not None else 0.0,
                'image_path': str(row.get('image_path', '')),
                'frame_path': str(row.get('frame_path', '')),
                'embedding_path': str(row.get('embedding_path', ''))
            }
            
            embeddings.append(embedding_list)
            metadata_list.append(metadata)
            ids.append(face_id)
        
        # Save to vector store
        try:
            face_vector_store.add_face_embeddings(
                embeddings=embeddings,
                metadata_list=metadata_list,
                ids=ids
            )
            logger.info(f"✅ Successfully saved {len(embeddings)} embeddings to vector store")
        except Exception as e:
            logger.error(f"❌ Error saving embeddings to vector store: {e}")
            raise
