"""
Face clustering system for SEMAMORPH speaker identification.
Implements cluster-based speaker assignment using facial embeddings.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging

from ..utils.logger_utils import setup_logging
from ..config import config
from .sex_validator import SexValidator  # NEW: Import sex validator

logger = setup_logging(__name__)

class FaceClusteringSystem:
    """
    Face clustering system for speaker identification.
    
    Implements the algorithm:
    1. Extract face embeddings from subtitles
    2. Cluster similar faces
    3. Associate characters with clusters using high-confidence dialogues  
    4. Assign uncertain dialogues based on cluster similarity
    """
    
    def __init__(self, path_handler, face_vector_store):
        """Initialize the face clustering system."""
        self.path_handler = path_handler
        self.face_vector_store = face_vector_store
        # Boolean confidence approach - no threshold needed
        self.clusters = {}  # cluster_id -> cluster_info
        self.cluster_medians = {}  # cluster_id -> median_embedding (Level 1: within-cluster)
        self.character_medians = {}  # character_name -> median_embedding (Level 2: cross-cluster)
        self.validated_clusters = {}  # Loaded from validated clusters file
        self.validated_character_medians = {}  # Loaded from validated character medians file
        
        # NEW: Initialize sex validator
        self.sex_validator = SexValidator(path_handler)
        
        # Load validated clusters from previous episodes in the same series
        self._initialize_with_validated_clusters()
        
        # Load character medians from previous episodes for cross-episode matching
        self._load_validated_character_medians()
        
    def _safe_get_numeric(self, data: dict, key: str, default: float = 0.0) -> float:
        """
        Safely get a numeric value from a dictionary, handling None values.
        
        Args:
            data: Dictionary to get value from
            key: Key to retrieve
            default: Default value if key doesn't exist or is None
            
        Returns:
            Float value, never None
        """
        value = data.get(key, default)
        return default if value is None else float(value)
        
    def _initialize_with_validated_clusters(self) -> None:
        """
        Load validated clusters from previous episodes in the same series.
        These will be used for cross-episode character matching.
        """
        try:
            self.validated_series_clusters = self.load_validated_clusters_from_series()
            
            if self.validated_series_clusters:
                character_count = len(self.validated_series_clusters)
                total_clusters = sum(len(clusters) for clusters in self.validated_series_clusters.values())
                logger.info(f"📚 Loaded {total_clusters} validated clusters for {character_count} characters from series '{self.path_handler.get_series()}'")
                
                # Log character summary
                for character, clusters in self.validated_series_clusters.items():
                    best_confidence = max(c['cluster_confidence'] for c in clusters) if clusters else 0
                    total_faces = sum(c['face_count'] for c in clusters)
                    logger.debug(f"   📊 {character}: {len(clusters)} clusters, {total_faces} faces, best confidence: {best_confidence:.1f}%")
            else:
                logger.info(f"📚 No validated clusters found for series '{self.path_handler.get_series()}' - first episode or fresh start")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load validated clusters from series: {e}")
            self.validated_series_clusters = {}
    
    def _load_validated_character_medians(self) -> None:
        """
        Load validated character medians from previous episodes in the same series.
        These will be used for cross-episode character matching of new clusters.
        """
        try:
            character_medians = self.load_validated_character_medians_from_series()
            
            if character_medians:
                self.validated_character_medians = character_medians
                logger.info(f"🎭 Loaded character medians for {len(character_medians)} characters from series '{self.path_handler.get_series()}'")
                
                # Log character median summary
                for character, median_info in character_medians.items():
                    episode_code = median_info.get('episode_code', 'unknown')
                    cluster_count = median_info.get('cluster_count', 0)
                    face_count = median_info.get('face_count', 0)
                    logger.debug(f"   🎭 {character}: from {episode_code}, {cluster_count} clusters, {face_count} faces")
            else:
                logger.info(f"🎭 No validated character medians found for series '{self.path_handler.get_series()}'")
                self.validated_character_medians = {}
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load validated character medians from series: {e}")
            self.validated_character_medians = {}
        
    def run_face_clustering_pipeline(
        self, 
        df_faces: pd.DataFrame,
        dialogue_lines: List,
        cosine_similarity_threshold: Optional[float] = None,
        min_cluster_size: Optional[int] = None,
        merge_threshold: Optional[float] = None,
        expected_embedding_dim: Optional[int] = None
    ) -> Tuple[List, Dict, Dict, pd.DataFrame]:
        """
        Run the complete face clustering pipeline with consistent data handling.
        
        CRITICAL: All files must be saved AFTER all cluster modifications are complete
        to ensure data consistency across files.
        """
        logger.info("🎭 Starting Face Clustering Pipeline")
        
        # Set default thresholds from config if not provided
        if cosine_similarity_threshold is None:
            cosine_similarity_threshold = config.cosine_similarity_threshold
        if min_cluster_size is None:
            min_cluster_size = config.min_cluster_size_final
        if merge_threshold is None:
            merge_threshold = config.centroid_merge_threshold
        if expected_embedding_dim is None:
            expected_embedding_dim = config.face_embedding_dimension
            
        logger.info(f"📊 Configuration: similarity={cosine_similarity_threshold}, min_size={min_cluster_size}, merge={merge_threshold}")
        
        # Step 1: Perform initial clustering using ChromaDB
        logger.info("🔗 Step 1: Clustering faces using ChromaDB similarity")
        clustered_faces = self._cluster_faces_chromadb(
            df_faces, 
            cosine_similarity_threshold=cosine_similarity_threshold,
            min_cluster_size=min_cluster_size,
            expected_embedding_dim=expected_embedding_dim
        )
        
        # Step 2: Refine clusters if merge threshold provided
        if merge_threshold > 0:
            logger.info("🔄 Step 2: Refining clusters via merging")
            clustered_faces = self._refine_clusters_chromadb(
                clustered_faces,
                merge_threshold=merge_threshold,
                embedding_col='embedding',
                face_id_col='face_id'
            )
        
        # Step 3: Associate characters with clusters using high-confidence dialogues
        logger.info("🔗 Step 3: Associating characters with clusters")
        self._associate_characters_with_clusters(clustered_faces, min_cluster_size)
        
        # Step 4: Apply cluster assignment enhancements BEFORE calculating medians
        # CRITICAL: This must happen before any file saving to ensure consistency
        if config.enable_parity_detection or config.enable_spatial_outlier_removal:
            logger.info("🔍 Step 4: Applying cluster assignment enhancements...")
            self._enhance_cluster_assignments(clustered_faces, 'face_id')
            
            # CRITICAL: Re-normalize cluster IDs after enhancements to ensure consistency
            logger.info("🔧 Step 4.1: Normalizing cluster IDs for consistency")
            clustered_faces = self._normalize_cluster_ids(clustered_faces)
        
        # Step 5: Generate cluster median embeddings (AFTER all modifications)
        logger.info("🧮 Step 5: Generating cluster median embeddings")
        self._generate_cluster_medians(clustered_faces)
        
        # Step 6: Generate character median embeddings (Level 2)
        logger.info("🎭 Step 6: Generating character median embeddings")
        self._generate_character_medians()
        
        # Step 6.5: Sex-based cluster validation (NEW)
        logger.info("🧬 Step 6.5: Sex-based cluster validation")
        self._perform_sex_validation(clustered_faces, dialogue_lines)
        
        # Step 7: Assign uncertain dialogues using cluster similarity
        logger.info("🎯 Step 7: Assigning uncertain dialogues")
        updated_dialogue = self._assign_uncertain_dialogues(dialogue_lines, clustered_faces)
        
        # Step 8: SAVE ALL FILES AT ONCE (CRITICAL for consistency)
        logger.info("💾 Step 8: Saving all cluster information consistently")
        self._save_all_cluster_files_consistently(clustered_faces)
        
        # Return results
        cluster_info = self.clusters
        character_clusters = self.get_character_clusters()
        
        logger.info(f"✅ Face clustering pipeline completed: {len(cluster_info)} clusters, {len(character_clusters)} character assignments")
        
        return updated_dialogue, cluster_info, character_clusters, clustered_faces
    
    def _cluster_faces_chromadb(
        self, 
        df_faces: pd.DataFrame,
        cosine_similarity_threshold: float = 0.65,
        min_cluster_size: int = 5,
        expected_embedding_dim: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Cluster face embeddings using ChromaDB for similarity search and persistence.
        Includes series, season, episode, and episode_code in metadata.
        """
        # Use config dimension if not provided
        if expected_embedding_dim is None:
            expected_embedding_dim = config.face_embedding_dimension
            
        logger.info(f"🔍 ChromaDB Similarity Clustering (Threshold: {cosine_similarity_threshold}, Min Size: {min_cluster_size})")

        # Input Validation & Normalization
        if df_faces.empty or "embedding" not in df_faces.columns or "image_path" not in df_faces.columns:
            logger.warning("⚠️ Input DataFrame empty or lacks 'embedding'/'image_path' columns.")
            return df_faces.assign(face_id=-1)
        
        initial_rows = len(df_faces)
        logger.info(f"   Validating {initial_rows} input embeddings (expecting dim {expected_embedding_dim})...")
        
        valid_mask = df_faces['embedding'].apply(
            lambda emb: isinstance(emb, np.ndarray) and emb.ndim == 1 and emb.shape[0] == expected_embedding_dim and np.linalg.norm(emb) > 1e-9
        )
        df_valid = df_faces[valid_mask].copy()
        num_invalid = initial_rows - len(df_valid)
        
        if num_invalid > 0:
            logger.warning(f"⚠️ Excluded {num_invalid} rows with invalid embeddings.")
        if df_valid.empty:
            logger.error("❌ No valid embeddings left after filtering.")
            return df_valid.assign(face_id=-1)
        
        logger.info("   Normalizing embeddings...")
        embeddings_list = [(emb / np.linalg.norm(emb)).astype(np.float32) for emb in df_valid["embedding"]]
        df_valid['normalized_embedding'] = embeddings_list
        logger.info(f"   Processed {len(df_valid)} valid & normalized embeddings.")

        # Get Context from PathHandler
        series = self.path_handler.get_series()
        season = self.path_handler.get_season()
        episode = self.path_handler.get_episode()
        episode_code = self.path_handler.get_episode_code()

        # ChromaDB Interaction
        try:
            collection = self.face_vector_store.get_face_collection()
        except Exception as e:
            logger.error(f"❌ Cannot proceed without ChromaDB collection. Error: {e}")
            return df_valid.assign(face_id=-1)

        # Identify Existing vs. New Faces
        all_image_paths = df_valid['image_path'].tolist()
        logger.info(f"   Checking {len(all_image_paths)} faces against ChromaDB collection '{collection.name}'...")
        
        try:
            existing_in_db = collection.get(ids=all_image_paths, include=['metadatas'])
            existing_ids_set = set(existing_in_db['ids'])
            logger.info(f"   Found {len(existing_ids_set)} faces already present in ChromaDB.")
            
            # Update DataFrame with existing face_id assignments from ChromaDB
            if existing_in_db['metadatas']:
                existing_assignments = {}
                for id_, meta in zip(existing_in_db['ids'], existing_in_db['metadatas']):
                    if meta and 'face_id' in meta:
                        existing_assignments[id_] = meta['face_id']
                
                # Map existing assignments to DataFrame
                df_valid['face_id'] = df_valid['image_path'].map(lambda x: existing_assignments.get(x, -99))
                logger.info(f"   Updated DataFrame with existing face_id assignments for {len(existing_assignments)} faces.")
        except Exception as e:
            logger.warning(f"⚠️ Error checking existing IDs in ChromaDB: {e}. Assuming all faces are new for adding.")
            existing_ids_set = set()
            df_valid['face_id'] = -99  # Mark all as unprocessed

        new_faces_df = df_valid[~df_valid['image_path'].isin(existing_ids_set)]
        num_to_add = len(new_faces_df)

        # Add New Faces to ChromaDB (with initial face_id = -99 and context)
        if num_to_add > 0:
            logger.info(f"   Adding {num_to_add} new faces to ChromaDB collection with initial face_id=-99...")
            try:
                # Build metadata list including context
                metadatas_to_add = []
                for _, row in new_faces_df.iterrows():
                    metadatas_to_add.append({
                        'face_id': -99,  # Marker for 'unprocessed'
                        'dialogue_index': int(row.get('dialogue_index', -1)),
                        'face_index': int(row.get('face_index', 0)),
                        'timestamp_seconds': float(row.get('timestamp_seconds', 0.0)),
                        'speaker': str(row.get('speaker', '')),
                        'speaker_confidence': float(row.get('speaker_confidence', 0.0)),
                        'detection_confidence': float(row.get('detection_confidence', 0.0)),
                        'blur_score': float(row.get('blur_score', 0.0)),
                        'series': str(series),
                        'season': str(season),
                        'episode': str(episode),
                        'episode_code': str(episode_code)
                    })

                collection.add(
                    ids=new_faces_df['image_path'].tolist(),
                    embeddings=new_faces_df['normalized_embedding'].tolist(),
                    metadatas=metadatas_to_add
                )
                logger.info(f"   ✅ Successfully added {num_to_add} new faces with context metadata.")
                
                # Update DataFrame for new faces - mark them as unprocessed
                df_valid.loc[df_valid['image_path'].isin(new_faces_df['image_path']), 'face_id'] = -99
                logger.info(f"   📝 Marked {num_to_add} new faces as unprocessed (face_id = -99) in DataFrame.")
                
            except Exception as e:
                logger.error(f"❌ Error adding new faces to ChromaDB: {e}")
        else:
            logger.info("   No new faces to add to ChromaDB.")
            
        # Ensure all faces have face_id set (either from existing assignments or as unprocessed)
        if 'face_id' not in df_valid.columns:
            df_valid['face_id'] = -99
        df_valid['face_id'] = df_valid['face_id'].fillna(-99).astype(int)
        logger.info(f"   📊 DataFrame face_id distribution: {df_valid['face_id'].value_counts().to_dict()}")

        # Assign Clusters for faces that need processing
        unprocessed_faces_df = df_valid[df_valid['face_id'] == -99]
        logger.info(f"   Assigning cluster IDs to {len(unprocessed_faces_df)} unprocessed faces...")

        if unprocessed_faces_df.empty:
            logger.info("   No faces found needing cluster assignment.")
        else:
            logger.info(f"   Found {len(unprocessed_faces_df)} faces to assign...")
            next_new_cluster_id = 0

            # Find the maximum existing cluster ID globally across the collection
            try:
                all_metas = collection.get(include=['metadatas'])
                max_existing_id = -1
                if all_metas and all_metas['metadatas']:
                    existing_face_ids = [
                        m['face_id'] for m in all_metas['metadatas']
                        if m and isinstance(m.get('face_id'), int) and m.get('face_id') >= 0
                    ]
                    if existing_face_ids:
                        max_existing_id = max(existing_face_ids)
                next_new_cluster_id = max_existing_id + 1
                logger.info(f"   Determined next available global cluster ID: {next_new_cluster_id}")
            except Exception as e:
                logger.warning(f"⚠️ Error determining max existing global cluster ID: {e}. Starting new IDs from 0.")
                next_new_cluster_id = 0

            num_assigned = 0
            num_new_clusters = 0
            
            for idx, (_, row) in enumerate(tqdm(unprocessed_faces_df.iterrows(), desc="Assigning Clusters", total=len(unprocessed_faces_df))):
                face_id_to_process = row['image_path']
                current_embedding = row['normalized_embedding']
                assigned_cluster_id = -1

                # Query ChromaDB for nearest *already assigned* neighbors (globally)
                try:
                    norm = np.linalg.norm(current_embedding)
                    if norm < 1e-9:
                        continue
                    query_embedding_normalized = (current_embedding / norm).astype(np.float32).tolist()

                    results = collection.query(
                        query_embeddings=[query_embedding_normalized],
                        n_results=1,
                        where={"face_id": {"$gte": 0}},  # Only search assigned clusters
                        include=['metadatas', 'distances']
                    )

                    if results and results['ids'] and results['ids'][0]:
                        best_match_meta = results['metadatas'][0][0]
                        best_match_distance = results['distances'][0][0]
                        similarity = 1.0 - best_match_distance

                        if similarity >= cosine_similarity_threshold:
                            assigned_cluster_id = best_match_meta['face_id']

                except Exception as e:
                    if "No datapoints found" in str(e) or "query resulted in empty results" in str(e):
                        pass
                    else:
                        logger.warning(f"⚠️ Error querying ChromaDB for face {face_id_to_process}: {e}")
                    assigned_cluster_id = -1

                # Assign new cluster ID if no match found
                if assigned_cluster_id == -1:
                    assigned_cluster_id = next_new_cluster_id
                    next_new_cluster_id += 1
                    num_new_clusters += 1

                # Update the face's metadata in ChromaDB
                try:
                    # Get the current metadata to preserve existing fields
                    current_data = collection.get(ids=[face_id_to_process], include=['metadatas'])
                    if current_data and current_data['metadatas'] and current_data['metadatas'][0]:
                        # Preserve existing metadata and update face_id
                        updated_meta = current_data['metadatas'][0].copy()
                        updated_meta['face_id'] = int(assigned_cluster_id)
                    else:
                        # Create new metadata if none exists
                        updated_meta = {
                            'face_id': int(assigned_cluster_id),
                            'dialogue_index': int(row.get('dialogue_index', -1)),
                            'face_index': int(row.get('face_index', 0)),
                            'timestamp_seconds': float(row.get('timestamp_seconds', 0.0)),
                            'speaker': str(row.get('speaker', '')),
                            'speaker_confidence': float(row.get('speaker_confidence', 0.0)),
                            'detection_confidence': float(row.get('detection_confidence', 0.0)),
                            'blur_score': float(row.get('blur_score', 0.0)),
                            'series': str(series),
                            'season': str(season),
                            'episode': str(episode),
                            'episode_code': str(episode_code)
                        }
                    
                    collection.update(
                        ids=[face_id_to_process],
                        metadatas=[updated_meta]
                    )
                    
                    # Update DataFrame
                    df_valid.loc[df_valid['image_path'] == face_id_to_process, 'face_id'] = assigned_cluster_id
                    num_assigned += 1
                except Exception as e:
                    logger.error(f"❌ Error updating face_id for {face_id_to_process} in ChromaDB: {e}")

            logger.info(f"   ✅ Assignment loop finished. Assigned IDs to {num_assigned} faces. Created {num_new_clusters} new global clusters.")

        # Post-Processing: Min Cluster Size
        logger.info(f"🧹 Applying min_cluster_size ({min_cluster_size}) cleanup via ChromaDB updates...")
        try:
            all_data = collection.get(include=['metadatas'])
            if all_data and all_data['ids']:
                current_assignments = {}
                for id_, meta in zip(all_data['ids'], all_data['metadatas']):
                    if meta and 'face_id' in meta and meta['face_id'] is not None:
                        current_assignments[id_] = int(meta['face_id'])
                    else:
                        current_assignments[id_] = -1

                face_ids_series = pd.Series(current_assignments)
                id_counts = face_ids_series[face_ids_series >= 0].value_counts()
                small_clusters = id_counts[id_counts < min_cluster_size].index.tolist()

                if small_clusters:
                    ids_to_update = face_ids_series[face_ids_series.isin(small_clusters)].index.tolist()
                    num_reassigned = len(ids_to_update)
                    logger.info(f"   Found {len(small_clusters)} clusters < {min_cluster_size}. Reassigning {num_reassigned} faces to noise (-1).")

                    if ids_to_update:
                        batch_size = 500
                        for i in range(0, len(ids_to_update), batch_size):
                            batch_ids = ids_to_update[i:i + batch_size]
                            
                            # Get current metadata to preserve existing fields
                            current_batch_data = collection.get(ids=batch_ids, include=['metadatas'])
                            updated_metadatas = []
                            
                            for j, batch_id in enumerate(batch_ids):
                                if (current_batch_data and current_batch_data['metadatas'] and 
                                    j < len(current_batch_data['metadatas']) and 
                                    current_batch_data['metadatas'][j]):
                                    # Preserve existing metadata and update face_id
                                    updated_meta = current_batch_data['metadatas'][j].copy()
                                    updated_meta['face_id'] = -1
                                else:
                                    # Create minimal metadata if none exists
                                    updated_meta = {'face_id': -1}
                                updated_metadatas.append(updated_meta)
                            
                            collection.update(
                                ids=batch_ids,
                                metadatas=updated_metadatas
                            )
                        logger.info("   ✅ Min cluster size cleanup complete.")
                else:
                    logger.info("   No clusters smaller than min_cluster_size.")
            else:
                logger.info("   No data found in collection for cleanup check.")

        except Exception as e:
            logger.error(f"❌ Error during min cluster size cleanup: {e}")

        # Final DataFrame preparation
        logger.info("   Preparing final DataFrame with cluster assignments...")
        
        # Ensure face_id column exists and is properly typed
        if 'face_id' not in df_valid.columns:
            df_valid['face_id'] = -1
        df_valid['face_id'] = df_valid['face_id'].astype(int)
        
        # Map face_id to cluster_id for compatibility with existing code
        df_valid['cluster_id'] = df_valid['face_id']

        n_clusters_final = len(df_valid[df_valid['face_id'] >= 0]['face_id'].unique())
        n_noise_final = (df_valid['face_id'] == -1).sum()
        logger.info(f"📊 Final Counts in DataFrame: {n_clusters_final} cluster(s), {n_noise_final} noise points.")

        return df_valid
    
    def _calculate_normalized_centroids(self, df_clustered: pd.DataFrame, embedding_col='embedding', face_id_col='face_id'):
        """Calculate normalized centroids for clusters."""
        centroids = {}
        if embedding_col not in df_clustered.columns:
            logger.warning(f"⚠️ Embedding column '{embedding_col}' not found for centroid calculation.")
            if 'normalized_embedding' in df_clustered.columns:
                logger.info("   Using 'normalized_embedding' column instead.")
                embedding_col = 'normalized_embedding'
            else:
                return centroids
        
        df_valid = df_clustered[(df_clustered[face_id_col] >= 0) & df_clustered[embedding_col].notna()].copy()
        if df_valid.empty:
            return centroids
        
        df_valid = df_valid[df_valid[embedding_col].apply(lambda x: isinstance(x, np.ndarray) and x.ndim == 1 and x.size > 0)]
        if df_valid.empty:
            return centroids

        grouped = df_valid.groupby(face_id_col)
        for face_id, group in tqdm(grouped, desc="Calculating Centroids", leave=False):
            try:
                embeddings = np.vstack(group[embedding_col].tolist())
                if embeddings.ndim != 2 or embeddings.shape[0] == 0:
                    continue
                centroid = np.mean(embeddings, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-9:
                    normalized_centroid = (centroid / norm).astype(np.float32)
                    centroids[face_id] = normalized_centroid
            except ValueError as e:
                logger.warning(f"⚠️ ValueError calculating centroid for cluster {face_id}: {e}. Check embedding shapes.")
                continue
            except Exception as e:
                logger.warning(f"⚠️ Unexpected error calculating centroid for cluster {face_id}: {e}")
                continue
        return centroids
    
    def _refine_clusters_chromadb(
        self,
        df_clustered: pd.DataFrame,
        merge_threshold: float,
        embedding_col='embedding',
        face_id_col='face_id'
    ) -> pd.DataFrame:
        """Refine clusters via ChromaDB merging based on centroid similarity."""
        logger.info(f"🔄 Refining Clusters via ChromaDB (Threshold: {merge_threshold})")
        
        try:
            collection = self.face_vector_store.get_face_collection()
        except Exception as e:
            logger.error(f"❌ Cannot refine clusters without ChromaDB collection. Error: {e}")
            return df_clustered

        iteration = 0
        max_iterations = 200  # Increased for more aggressive clustering

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"  Refinement Iteration {iteration}...")
            merged_in_iteration = False

            logger.info(f"    Calculating centroids from {len(df_clustered)} faces in DataFrame...")
            centroids_dict = self._calculate_normalized_centroids(df_clustered, embedding_col, face_id_col)
            cluster_ids = sorted([cid for cid in centroids_dict.keys() if cid >= 0])
            num_clusters = len(cluster_ids)

            if num_clusters < 2:
                logger.info("  No more pairs of clusters (>=0) to compare. Stopping refinement.")
                break

            logger.info(f"    Comparing {num_clusters} current centroids...")
            if not centroids_dict or not cluster_ids:
                logger.info("    No valid centroids calculated. Stopping refinement.")
                break

            valid_cluster_ids = [cid for cid in cluster_ids if cid in centroids_dict]
            if len(valid_cluster_ids) < num_clusters:
                logger.info(f"    Warning: Mismatch between cluster IDs ({num_clusters}) and calculated centroids ({len(valid_cluster_ids)}). Using valid subset.")
                cluster_ids = valid_cluster_ids
                num_clusters = len(cluster_ids)
                if num_clusters < 2:
                    logger.info("    Not enough valid centroids remaining to compare. Stopping refinement.")
                    break
            
            centroid_matrix = np.array([centroids_dict[fid] for fid in cluster_ids])

            try:
                similarity_matrix = cosine_similarity(centroid_matrix)
            except Exception as e:
                logger.info(f"    ❌ Error calculating similarity matrix: {e}. Stopping refinement.")
                break

            max_similarity = -1.0
            merge_pair = None
            for i in range(num_clusters):
                for j in range(i + 1, num_clusters):
                    similarity = similarity_matrix[i, j]
                    if similarity >= merge_threshold:
                        if similarity > max_similarity:
                            max_similarity = similarity
                            id1 = cluster_ids[i]
                            id2 = cluster_ids[j]
                            merge_pair = (min(id1, id2), max(id1, id2))

            if merge_pair:
                id_to_keep, id_to_merge = merge_pair
                logger.info(f"    Found merge candidate: Cluster {id_to_merge} into Cluster {id_to_keep} (Similarity: {max_similarity:.4f})")
                try:
                    docs_to_update = collection.get(where={face_id_col: id_to_merge}, include=[])
                    ids_to_update = docs_to_update['ids']
                    num_affected = len(ids_to_update)
                    if num_affected > 0:
                        logger.info(f"    Updating {num_affected} faces in ChromaDB from face_id {id_to_merge} to {id_to_keep}...")
                        batch_size = 500
                        for i in range(0, len(ids_to_update), batch_size):
                            batch_ids = ids_to_update[i:i+batch_size]
                            
                            # Get current metadata to preserve existing fields
                            current_batch_data = collection.get(ids=batch_ids, include=['metadatas'])
                            updated_metadatas = []
                            
                            for j, batch_id in enumerate(batch_ids):
                                if (current_batch_data and current_batch_data['metadatas'] and 
                                    j < len(current_batch_data['metadatas']) and 
                                    current_batch_data['metadatas'][j]):
                                    # Preserve existing metadata and update face_id
                                    updated_meta = current_batch_data['metadatas'][j].copy()
                                    updated_meta[face_id_col] = int(id_to_keep)
                                else:
                                    # Create minimal metadata if none exists
                                    updated_meta = {face_id_col: int(id_to_keep)}
                                updated_metadatas.append(updated_meta)
                            
                            collection.update(
                                ids=batch_ids,
                                metadatas=updated_metadatas
                            )
                        logger.info("      ✅ ChromaDB update successful.")
                        df_clustered.loc[df_clustered[face_id_col] == id_to_merge, face_id_col] = id_to_keep
                        # Also update cluster_id for compatibility
                        df_clustered.loc[df_clustered['cluster_id'] == id_to_merge, 'cluster_id'] = id_to_keep
                        merged_in_iteration = True
                    else:
                        logger.info(f"    No faces found in ChromaDB with face_id {id_to_merge}. Skipping update.")
                except Exception as e:
                    logger.info(f"    ❌ Error updating ChromaDB during merge: {e}. Stopping refinement.")
                    break
            else:
                logger.info("    No cluster pairs found above threshold in this iteration.")

            if not merged_in_iteration:
                logger.info("  No merges performed in this iteration. Refinement complete.")
                break
            elif iteration == max_iterations:
                logger.info("  Reached maximum refinement iterations. Stopping.")

        final_cluster_count = len(df_clustered[df_clustered[face_id_col] >= 0][face_id_col].unique())
        logger.info(f"--- Refinement Finished. Final cluster count (>=0) in DataFrame: {final_cluster_count} ---")
        return df_clustered
    
    def _associate_characters_with_clusters(self, df_faces: pd.DataFrame, min_cluster_size: int) -> None:
        """
        Associate characters with clusters based on high-confidence dialogues.
        
        Enhanced with multi-face probability distribution when enabled.
        For each character with confidence >= threshold, find the predominant cluster.
        Uses face_id as the cluster identifier from ChromaDB clustering.
        Only assigns characters to clusters that meet the minimum size requirement.
        """
        if df_faces.empty:
            return
        
        # Check if multi-face processing is enabled
        enable_multiface = config.enable_multiface_processing
        logger.info(f"🔧 Multi-face processing: {'ENABLED' if enable_multiface else 'DISABLED'}")
        
        # Step 2.5: Cross-episode character matching using validated clusters
        logger.info("🔗 Step 2.5: Cross-episode character matching")
        self._match_clusters_with_validated_characters(df_faces)
        
        if enable_multiface:
            self._associate_characters_with_clusters_multiface(df_faces, min_cluster_size)
        else:
            self._associate_characters_with_clusters_singleface(df_faces, min_cluster_size)
    
    def _associate_characters_with_clusters_singleface(self, df_faces: pd.DataFrame, min_cluster_size: int) -> None:
        """
        Original single-face cluster association algorithm.
        """
        # Use face_id as cluster identifier (from ChromaDB clustering)
        cluster_col = 'face_id' if 'face_id' in df_faces.columns else 'cluster_id'
        
        # PRE-FILTER: Remove clusters that don't meet minimum size requirement
        logger.info(f"🔍 Pre-filtering clusters by minimum size ({min_cluster_size})")
        cluster_sizes = df_faces[df_faces[cluster_col] >= 0][cluster_col].value_counts()
        valid_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()
        
        logger.info(f"📊 Found {len(valid_clusters)} clusters meeting size requirement out of {len(cluster_sizes)} total clusters")
        
        # Filter DataFrame to only include valid clusters
        df_faces_filtered = df_faces[
            (df_faces[cluster_col].isin(valid_clusters)) | (df_faces[cluster_col] == -1)  # Keep noise for completeness
        ].copy()
        
        if df_faces_filtered.empty:
            logger.warning("⚠️ No valid clusters found after size filtering")
            return
        
        # Filter confident speaker assignments (using filtered data)
        confident_mask = (
            (df_faces_filtered['is_llm_confident'] == True) &
            (df_faces_filtered['speaker'].notna()) &
            (df_faces_filtered['speaker'] != '') &
            (df_faces_filtered[cluster_col] != -1)  # Valid clusters only
        )
        
        df_confident = df_faces_filtered[confident_mask]
        
        if df_confident.empty:
            logger.warning(f"⚠️ No confident speaker assignments found")
            return
        
        logger.info(f"📊 [SINGLE-FACE] Analyzing {len(df_confident)} confident face-speaker pairs from valid clusters")
        
        # Group by cluster and count speaker occurrences (FIXED LOGIC)
        cluster_speaker_counts = defaultdict(lambda: defaultdict(int))
        
        for _, row in df_confident.iterrows():
            speaker = row['speaker']
            cluster_id = row[cluster_col]
            cluster_speaker_counts[cluster_id][speaker] += 1
        
        # Assign each cluster to its predominant character (FIXED LOGIC)
        for cluster_id, speaker_counts in cluster_speaker_counts.items():
            # Find the speaker with the most occurrences in this cluster
            predominant_speaker = max(speaker_counts.items(), key=lambda x: x[1])
            speaker, count = predominant_speaker
            
            # Calculate confidence (percentage of cluster faces belonging to this speaker)
            total_faces_in_cluster = sum(speaker_counts.values())
            cluster_confidence = (count / total_faces_in_cluster) * 100
            
            # Always assign to the majority speaker (no arbitrary threshold)
            logger.debug(f"🔍 Assigning cluster {cluster_id} to speaker: {speaker}")
            # Note: Character assignments now tracked in self.clusters only for consistency
            
            # Store cluster info
            logger.debug(f"🔍 Storing cluster info - cluster_id: {cluster_id}, speaker: {speaker}")
            
            if cluster_id not in self.clusters:
                    self.clusters[cluster_id] = {
                        'cluster_id': int(cluster_id),  # Ensure int conversion
                        'character_name': speaker,
                        'face_count': int(count),       # Count of predominant speaker's faces
                        'total_cluster_faces': int(total_faces_in_cluster),  # Total faces in cluster
                        'cluster_confidence': float(cluster_confidence),  # Ensure float conversion
                        'faces': [],
                        'assignment_method': 'single_face_count',
                        'speaker_breakdown': dict(speaker_counts),  # Show all speakers in cluster
                        'cluster_status': 'VALID'  # Default status for successfully assigned clusters
                }
            else:
                # Update existing cluster (in case of conflicts, keep highest confidence)
                if cluster_confidence > self.clusters[cluster_id].get('cluster_confidence', 0):
                    self.clusters[cluster_id]['character_name'] = speaker
                    self.clusters[cluster_id]['face_count'] = int(count)
                    self.clusters[cluster_id]['total_cluster_faces'] = int(total_faces_in_cluster)
                    self.clusters[cluster_id]['cluster_confidence'] = float(cluster_confidence)
                    self.clusters[cluster_id]['assignment_method'] = 'single_face_count'
                    self.clusters[cluster_id]['speaker_breakdown'] = dict(speaker_counts)
            
            logger.info(f"✅ [SINGLE-FACE] Assigned cluster {cluster_id} to '{speaker}' ({cluster_confidence:.1f}% confidence, {count}/{total_faces_in_cluster} faces)")
            logger.debug(f"   📊 Speaker breakdown: {dict(speaker_counts)}")
        
        # Apply cluster assignment enhancements
        if config.enable_parity_detection or config.enable_spatial_outlier_removal:
            logger.info("🔍 [ENHANCEMENT] Applying cluster assignment enhancements...")
            self._enhance_cluster_assignments(df_faces_filtered, cluster_col)
    
    def _associate_characters_with_clusters_multiface(self, df_faces: pd.DataFrame, min_cluster_size: int) -> None:
        """
        Enhanced multi-face cluster association algorithm with probability distribution.
        """
        # Use face_id as cluster identifier (from ChromaDB clustering)
        cluster_col = 'face_id' if 'face_id' in df_faces.columns else 'cluster_id'
        
        # PRE-FILTER: Remove clusters that don't meet minimum size requirement
        logger.info(f"🔍 [MULTI-FACE] Pre-filtering clusters by minimum size ({min_cluster_size})")
        cluster_sizes = df_faces[df_faces[cluster_col] >= 0][cluster_col].value_counts()
        valid_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()
        
        logger.info(f"📊 Found {len(valid_clusters)} clusters meeting size requirement out of {len(cluster_sizes)} total clusters")
        
        # Filter DataFrame to only include valid clusters
        df_faces_filtered = df_faces[
            (df_faces[cluster_col].isin(valid_clusters)) | (df_faces[cluster_col] == -1)  # Keep noise for completeness
        ].copy()
        
        if df_faces_filtered.empty:
            logger.warning("⚠️ No valid clusters found after size filtering")
            return
        
        # Get confident dialogue indices
        confident_mask = (
            (df_faces_filtered['is_llm_confident'] == True) &
            (df_faces_filtered['speaker'].notna()) &
            (df_faces_filtered['speaker'] != '') &
            (df_faces_filtered[cluster_col] != -1)  # Valid clusters only
        )
        
        df_confident = df_faces_filtered[confident_mask]
        
        if df_confident.empty:
            logger.warning(f"⚠️ No confident speaker assignments found")
            return
        
        # Get unique dialogue indices with confident assignments
        confident_dialogue_indices = df_confident['dialogue_index'].unique()
        logger.info(f"📊 [MULTI-FACE] Processing {len(confident_dialogue_indices)} confident dialogues")
        
        # Enhanced character-cluster probability accumulation
        character_cluster_probabilities = defaultdict(lambda: defaultdict(float))
        character_dialogue_counts = defaultdict(lambda: defaultdict(int))
        total_faces_processed = 0
        total_probability_distributed = 0.0
        
        # Process each confident dialogue
        for dialogue_idx in confident_dialogue_indices:
            dialogue_faces = df_faces_filtered[df_faces_filtered['dialogue_index'] == dialogue_idx]
            
            if len(dialogue_faces) == 0:
                continue
                
            # All faces in the same dialogue have the same speaker (from LLM assignment)
            speaker = dialogue_faces.iloc[0]['speaker']
            valid_dialogue_faces = dialogue_faces[dialogue_faces[cluster_col] != -1]
            
            if len(valid_dialogue_faces) > 0:
                # Apply max faces limit for performance
                max_faces = config.multiface_max_faces_per_dialogue
                if len(valid_dialogue_faces) > max_faces:
                    # Sort by face quality and take top N
                    valid_dialogue_faces = valid_dialogue_faces.nlargest(max_faces, 'detection_confidence')
                    logger.debug(f"🔧 Limited dialogue {dialogue_idx} to top {max_faces} faces (from {len(dialogue_faces)})")
                
                # Equal probability distribution among all valid faces
                if config.multiface_equal_probability_distribution:
                    probability_per_face = 1.0 / len(valid_dialogue_faces)
                else:
                    # Alternative: Weight by detection confidence
                    total_confidence = valid_dialogue_faces['detection_confidence'].sum()
                    confidence_weights = valid_dialogue_faces['detection_confidence'] / total_confidence
                
                logger.debug(f"🎯 Dialogue {dialogue_idx}: {len(valid_dialogue_faces)} faces, {probability_per_face:.3f} probability each")
                
                for _, face_row in valid_dialogue_faces.iterrows():
                    cluster_id = face_row[cluster_col]
                    
                    if config.multiface_equal_probability_distribution:
                        prob_to_add = probability_per_face
                    else:
                        prob_to_add = confidence_weights.loc[face_row.name]
                    
                    character_cluster_probabilities[speaker][cluster_id] += prob_to_add
                    character_dialogue_counts[speaker][cluster_id] += 1
                    total_faces_processed += 1
                    total_probability_distributed += prob_to_add
        
        logger.info(f"📊 [MULTI-FACE] Processed {total_faces_processed} faces, distributed {total_probability_distributed:.2f} total probability")
        
        # FIXED LOGIC: Assign clusters to characters based on highest probability per cluster
        cluster_assignments = {}
        cluster_character_probabilities = defaultdict(dict)
        
        # Reorganize data: cluster_id -> {speaker: probability}
        for speaker, cluster_probs in character_cluster_probabilities.items():
            for cluster_id, probability in cluster_probs.items():
                cluster_character_probabilities[cluster_id][speaker] = probability
        
        # Assign each cluster to its predominant character
        for cluster_id, speaker_probs in cluster_character_probabilities.items():
            if speaker_probs:
                # Find speaker with highest probability in this cluster
                best_speaker = max(speaker_probs.items(), key=lambda x: x[1])
                speaker, speaker_probability = best_speaker
                
                # VALIDATION: Check if top character has sufficient occurrences (configurable minimum)
                minimum_occurrences = config.cluster_minimum_occurrences
                if speaker_probability < minimum_occurrences:
                    logger.info(f"❌ [CLUSTER-VALIDATION] Cluster {cluster_id} rejected: top character '{speaker}' has only {speaker_probability:.1f} occurrences (< {minimum_occurrences})")
                    
                    # Mark cluster as invalid due to insufficient evidence
                    self.clusters[cluster_id] = {
                        'cluster_id': int(cluster_id),
                        'character_name': None,
                        'face_count': 0,
                        'cluster_confidence': 0.0,
                        'cluster_status': 'INSUFFICIENT_EVIDENCE',
                        'faces': [],
                        'assignment_method': 'multi_face_probability',
                        'rejection_reason': f'Top character has only {speaker_probability:.1f} occurrences (< {minimum_occurrences})',
                        'speaker_breakdown': {s: float(p) for s, p in speaker_probs.items()}
                    }
                    continue
                
                # Calculate confidence as percentage of cluster's total probability mass
                total_cluster_probability = sum(speaker_probs.values())
                confidence = (speaker_probability / total_cluster_probability) * 100
                
                # Always assign to the majority speaker (no arbitrary threshold)
                # Note: Character assignments now tracked in self.clusters only for consistency
                
                cluster_assignments[cluster_id] = {
                    'assigned_character': speaker,
                    'confidence': float(confidence),
                    'total_occurrences': float(speaker_probability),
                    'total_cluster_probability': float(total_cluster_probability),
                    'dialogue_count': int(character_dialogue_counts[speaker][cluster_id]),
                    'assignment_method': 'multi_face_probability',
                    'speaker_breakdown': {s: float(p) for s, p in speaker_probs.items()},  # Probability values
                    'face_count_breakdown': {s: int(character_dialogue_counts[s][cluster_id]) for s in speaker_probs.keys()},  # Actual face counts
                    'total_actual_faces': int(sum(character_dialogue_counts[s][cluster_id] for s in speaker_probs.keys()))  # Total faces in cluster
                }
                
                # Store in clusters dict
                if cluster_id not in self.clusters:
                    self.clusters[cluster_id] = {
                        'cluster_id': int(cluster_id),
                        'character_name': speaker,
                        'face_count': int(character_dialogue_counts[speaker][cluster_id]),
                        'total_cluster_faces': int(sum(character_dialogue_counts[s][cluster_id] for s in speaker_probs.keys())),
                        'cluster_confidence': float(confidence),
                        'faces': [],
                        'assignment_method': 'multi_face_probability',
                        'total_probability': float(speaker_probability),
                        'total_cluster_probability': float(total_cluster_probability),
                        'speaker_breakdown': {s: float(p) for s, p in speaker_probs.items()},
                        'cluster_status': 'VALID'  # Default status for successfully assigned clusters
                    }
                else:
                    # Update existing cluster (in case of conflicts, keep highest confidence)
                    if confidence > self.clusters[cluster_id].get('cluster_confidence', 0):
                        self.clusters[cluster_id]['character_name'] = speaker
                        self.clusters[cluster_id]['face_count'] = int(character_dialogue_counts[speaker][cluster_id])
                        self.clusters[cluster_id]['total_cluster_faces'] = int(sum(character_dialogue_counts[s][cluster_id] for s in speaker_probs.keys()))
                        self.clusters[cluster_id]['cluster_confidence'] = float(confidence)
                        self.clusters[cluster_id]['assignment_method'] = 'multi_face_probability'
                        self.clusters[cluster_id]['total_probability'] = float(speaker_probability)
                        self.clusters[cluster_id]['total_cluster_probability'] = float(total_cluster_probability)
                        self.clusters[cluster_id]['speaker_breakdown'] = {s: float(p) for s, p in speaker_probs.items()}
                
                logger.info(f"✅ [MULTI-FACE] Assigned cluster {cluster_id} to '{speaker}' ({confidence:.1f}% confidence, {speaker_probability:.2f}/{total_cluster_probability:.2f} probability)")
                logger.debug(f"   📊 Speaker breakdown: {dict(speaker_probs)}")
        
        # Save cluster assignments to JSON file
        self._save_cluster_assignments_json(cluster_assignments)
        
        # Apply cluster assignment enhancements
        if config.enable_parity_detection or config.enable_spatial_outlier_removal:
            logger.info("🔍 [ENHANCEMENT] Applying cluster assignment enhancements...")
            self._enhance_cluster_assignments(df_faces_filtered, cluster_col)
        
        logger.info(f"✅ [MULTI-FACE] Character-cluster assignment completed:")
        logger.info(f"   📊 {len(cluster_assignments)} clusters assigned to characters")
        for cluster_id, assignment in sorted(cluster_assignments.items(), key=lambda x: x[1]['confidence'], reverse=True):
            character = assignment['assigned_character']
            confidence = assignment['confidence']
            logger.info(f"   🎭 Cluster {cluster_id} → '{character}' ({confidence:.1f}% confidence)")
        
        return df_faces_filtered
    
    def _generate_cluster_medians(self, df_faces: pd.DataFrame) -> None:
        """
        Generate median embeddings for each cluster.
        Uses face_id as the cluster identifier from ChromaDB clustering.
        """
        if df_faces.empty:
            return
        
        # Use face_id as cluster identifier (from ChromaDB clustering)
        cluster_col = 'face_id' if 'face_id' in df_faces.columns else 'cluster_id'
        
        # Group faces by cluster
        valid_clusters = df_faces[df_faces[cluster_col] != -1]
        
        for cluster_id in valid_clusters[cluster_col].unique():
            cluster_faces = valid_clusters[valid_clusters[cluster_col] == cluster_id]
            
            # VALIDATION: Only generate medians for valid clusters
            cluster_info = self.clusters.get(cluster_id, {})
            cluster_status = cluster_info.get('cluster_status')
            character_name = cluster_info.get('character_name')
            
            # Skip invalid clusters during median calculation
            if cluster_status in ['SPATIAL_OUTLIER', 'SPATIAL_OUTLIER_FROM_CHARACTER_MEDIAN', 'AMBIGUOUS', 'INSUFFICIENT_EVIDENCE'] or not character_name:
                logger.debug(f"🚫 Skipping median calculation for cluster {cluster_id}: status={cluster_status}, character={character_name}")
                continue
            
            # Extract embeddings for this cluster
            embeddings = []
            face_info = []
            
            for _, row in cluster_faces.iterrows():
                # Use normalized_embedding if available, otherwise use embedding
                embedding_col = 'normalized_embedding' if 'normalized_embedding' in row and isinstance(row['normalized_embedding'], np.ndarray) else 'embedding'
                
                if isinstance(row[embedding_col], np.ndarray) and row[embedding_col].size > 0:
                    embeddings.append(row[embedding_col])
                    face_info.append({
                        'dialogue_index': row.get('dialogue_index', -1),
                        'speaker': row.get('speaker', ''),
                        'speaker_confidence': row.get('speaker_confidence', 0.0),
                        'image_path': row.get('image_path', '')
                    })
            
            if embeddings:
                # Calculate median embedding
                embeddings_matrix = np.vstack(embeddings)
                median_embedding = np.median(embeddings_matrix, axis=0)
                
                # Normalize the median embedding
                norm = np.linalg.norm(median_embedding)
                if norm > 1e-9:
                    median_embedding = median_embedding / norm
                
                logger.debug(f"🔍 Storing cluster median - cluster_id type: {type(cluster_id)}, value: {cluster_id}")
                self.cluster_medians[cluster_id] = median_embedding
                
                # Update cluster info
                if cluster_id in self.clusters:
                    self.clusters[cluster_id]['faces'] = face_info
                    # Don't save median_embedding to avoid large JSON files
                    self.clusters[cluster_id]['embedding_dimension'] = int(len(median_embedding))  # Ensure int
                else:
                    # Create cluster info if not exists
                    self.clusters[cluster_id] = {
                        'cluster_id': int(cluster_id),  # Ensure int conversion
                        'character_name': None,
                        'face_count': int(len(embeddings)),  # Ensure int conversion
                        'faces': face_info,
                        # Don't save median_embedding to avoid large JSON files
                        'embedding_dimension': int(len(median_embedding))  # Ensure int conversion
                    }
                
                character_name = self.clusters[cluster_id].get('character_name', 'Unknown')
                logger.debug(f"🧮 Generated median embedding for cluster {cluster_id} ({character_name}): {len(embeddings)} faces")
    
    def _generate_character_medians(self) -> None:
        """
        Generate character-level medians from validated cluster medians.
        
        This creates Level 2 medians: median of cluster medians for each character.
        Should be called AFTER _generate_cluster_medians().
        """
        logger.info(f"🔍 [CHAR-MEDIAN-GEN] Starting character median generation...")
        logger.info(f"🔍 [CHAR-MEDIAN-GEN] Cluster medians available: {len(self.cluster_medians) if hasattr(self, 'cluster_medians') else 'No cluster_medians attr'}")
        character_clusters = self.get_character_clusters()
        logger.info(f"🔍 [CHAR-MEDIAN-GEN] Character clusters available: {len(character_clusters)}")
        logger.info(f"🔍 [CHAR-MEDIAN-GEN] Clusters available: {len(self.clusters) if hasattr(self, 'clusters') else 'No clusters attr'}")
        
        if not self.cluster_medians:
            logger.warning("⚠️ Cannot generate character medians: no cluster medians available")
            return
        
        self.character_medians.clear()
        
        # Group clusters by character - IMPROVED METHOD
        # Instead of relying on character_clusters (which only stores one cluster per character),
        # scan ALL clusters to find which ones are assigned to each character
        character_to_clusters = defaultdict(list)
        
        logger.info(f"🔍 [CHAR-MEDIAN-GEN] Scanning all clusters for character assignments...")
        
        # Scan all clusters for character assignments
        for cluster_id, cluster_info in self.clusters.items():
            character_name = cluster_info.get('character_name')
            if character_name:
                cluster_status = cluster_info.get('cluster_status', 'NO_STATUS')
                logger.debug(f"🔍 [CHAR-MEDIAN-GEN] Cluster {cluster_id}: character='{character_name}', status={cluster_status}")
                
                # Only include valid clusters
                if cluster_status not in ['SPATIAL_OUTLIER', 'SPATIAL_OUTLIER_FROM_CHARACTER_MEDIAN', 'AMBIGUOUS', 'INSUFFICIENT_EVIDENCE']:
                    if cluster_id in self.cluster_medians:
                        character_to_clusters[character_name].append(cluster_id)
                        logger.debug(f"✅ [CHAR-MEDIAN-GEN] Added cluster {cluster_id} to character '{character_name}'")
                    else:
                        logger.warning(f"⚠️ [CHAR-MEDIAN-GEN] Cluster {cluster_id} for '{character_name}' missing from cluster_medians")
                else:
                    logger.debug(f"❌ [CHAR-MEDIAN-GEN] Skipped cluster {cluster_id} for '{character_name}' due to status: {cluster_status}")
        
        logger.info(f"🔍 [CHAR-MEDIAN-GEN] Character to clusters mapping: {dict(character_to_clusters)}")
        logger.info(f"🔍 [CHAR-MEDIAN-GEN] Found {len(character_to_clusters)} characters with valid clusters")
        
        # Calculate character-level medians with face-count weighting
        for character, cluster_ids in character_to_clusters.items():
            if len(cluster_ids) >= 1:  # Need at least 1 valid cluster
                
                if len(cluster_ids) == 1:
                    # Single cluster: use cluster median directly
                    cluster_median = self.cluster_medians[cluster_ids[0]]
                    character_median = cluster_median.copy()
                    logger.debug(f"🎭 '{character}': Single cluster {cluster_ids[0]} → direct median")
                else:
                    # Multiple clusters: calculate FACE-COUNT WEIGHTED median
                    logger.debug(f"🎭 '{character}': Multiple clusters {cluster_ids} → face-count weighted median")
                    
                    # Collect cluster medians with their face counts for weighting
                    weighted_embeddings = []
                    total_faces = 0
                    
                    for cluster_id in cluster_ids:
                        cluster_median = self.cluster_medians[cluster_id]
                        cluster_info = self.clusters.get(cluster_id, {})
                        face_count = cluster_info.get('face_count', 1)
                        
                        # Repeat the cluster median 'face_count' times for weighted calculation
                        # This gives more influence to clusters with more faces
                        for _ in range(face_count):
                            weighted_embeddings.append(cluster_median)
                        
                        total_faces += face_count
                        logger.debug(f"      Cluster {cluster_id}: {face_count} faces (weight: {face_count})")
                    
                    # Calculate weighted median (clusters with more faces have more influence)
                    character_median = np.median(np.vstack(weighted_embeddings), axis=0)
                    logger.debug(f"      Result: Weighted median from {len(cluster_ids)} clusters, {total_faces} total faces")
                
                # Normalize character median
                norm = np.linalg.norm(character_median)
                if norm > 1e-9:
                    character_median = character_median / norm
                
                self.character_medians[character] = character_median
                logger.debug(f"🎭 Generated face-count weighted character median for '{character}': {len(cluster_ids)} clusters → Level 2 median")
        
        logger.info(f"🎭 Generated {len(self.character_medians)} character-level medians from {len(self.cluster_medians)} cluster medians")
        
        # Log detailed character median information
        for character_name, character_median in self.character_medians.items():
            cluster_count = len(character_to_clusters.get(character_name, []))
            logger.info(f"   🎭 '{character_name}': {cluster_count} clusters → Level 2 median generated")
        
        # Check for any characters that should have medians but don't
        all_characters_in_clusters = set()
        for cluster_info in self.clusters.values():
            char_name = cluster_info.get('character_name')
            if char_name:
                all_characters_in_clusters.add(char_name)
        
        missing_characters = all_characters_in_clusters - set(self.character_medians.keys())
        if missing_characters:
            logger.warning(f"⚠️ Characters found in clusters but missing character medians: {missing_characters}")
            for missing_char in missing_characters:
                clusters_for_char = [cid for cid, cinfo in self.clusters.items() 
                                   if cinfo.get('character_name') == missing_char]
                logger.warning(f"   Missing '{missing_char}' has clusters: {clusters_for_char}")
        else:
            logger.info(f"✅ All characters with valid clusters have character medians generated")
    
    def _generate_clustering_tracking_report(self) -> None:
        """
        Generate comprehensive clustering tracking report for debugging and analysis.
        """
        try:
            episode_code = self.path_handler.get_episode_code()
            timestamp = pd.Timestamp.now().isoformat()
            
            # Collect all clustering decisions and transformations
            tracking_data = {
                "episode_code": episode_code,
                "timestamp": timestamp,
                "processing_pipeline": {
                    "step_1_initial_clustering": {
                        "total_faces_processed": sum(c.get('face_count', 0) for c in self.clusters.values()),
                        "initial_clusters_formed": len(self.clusters),
                        "similarity_threshold_used": config.cosine_similarity_threshold,
                        "min_cluster_size": config.min_cluster_size_final
                    },
                    "step_2_character_assignment": {
                        "assignment_method": "multi_face_probability",
            
                        "minimum_occurrences": config.cluster_minimum_occurrences,
                        "characters_assigned": len(set(c.get('character_name') for c in self.clusters.values() if c.get('character_name')))
                    },
                    "step_3_quality_control": {
                        "spatial_outlier_detection": config.enable_spatial_outlier_removal,
                        "parity_detection": config.enable_parity_detection,
                        "ambiguous_resolution": config.enable_ambiguous_resolution,
                        "outlier_cluster_detection": config.enable_outlier_cluster_detection
                    }
                },
                "cluster_decisions": {},
                "character_assignments": {},
                "spatial_outlier_analysis": {},
                "protection_analysis": {},
                "final_statistics": {}
            }
            
            # Track each cluster's journey through the pipeline
            for cluster_id, cluster_info in self.clusters.items():
                cluster_decision = {
                    "cluster_id": cluster_id,
                    "initial_face_count": cluster_info.get('face_count', 0),
                    "total_faces_in_cluster": cluster_info.get('total_cluster_faces', 0),
                    "character_assigned": cluster_info.get('character_name'),
                    "assignment_confidence": cluster_info.get('cluster_confidence', 0),
                    "final_status": cluster_info.get('cluster_status', 'UNKNOWN'),
                    "assignment_method": cluster_info.get('assignment_method', 'unknown'),
                    "speaker_breakdown": cluster_info.get('speaker_breakdown', {}),
                    "total_occurrences": cluster_info.get('total_occurrences', 0),
                    "total_cluster_probability": cluster_info.get('total_cluster_probability', 0),
                    "character_percentage": 0,
                    "protection_analysis": {},
                    "outlier_reason": cluster_info.get('outlier_reason'),
                    "original_character": cluster_info.get('original_character')
                }
                
                # Calculate character percentage
                if cluster_info.get('total_cluster_probability', 0) > 0:
                    cluster_decision["character_percentage"] = (
                        cluster_info.get('total_occurrences', 0) / 
                        cluster_info.get('total_cluster_probability', 1) * 100
                    )
                
                # Analyze protection eligibility
                char_percentage = cluster_decision["character_percentage"]
                face_count = cluster_decision["initial_face_count"]
                protection_percentage = config.cluster_protection_percentage_threshold
                protection_min_faces = config.cluster_protection_min_faces
                
                cluster_decision["protection_analysis"] = {
                    "character_percentage": char_percentage,
                    "face_count": face_count,
                    "protection_threshold_percentage": protection_percentage,
                    "protection_threshold_faces": protection_min_faces,
                    "meets_percentage_threshold": char_percentage > protection_percentage,
                    "meets_face_threshold": face_count >= protection_min_faces,
                    "should_be_protected": (char_percentage > protection_percentage and face_count >= protection_min_faces),
                    "actual_status": cluster_decision["final_status"],
                    "protection_applied": (
                        cluster_decision["final_status"] != "SPATIAL_OUTLIER" and 
                        char_percentage > protection_percentage and 
                        face_count >= protection_min_faces
                    ),
                    "protection_failed": (
                        cluster_decision["final_status"] == "SPATIAL_OUTLIER" and 
                        char_percentage > protection_percentage and 
                        face_count >= protection_min_faces
                    )
                }
                
                tracking_data["cluster_decisions"][str(cluster_id)] = cluster_decision
            
            # Character assignment analysis
            for character_name in set(c.get('character_name') for c in self.clusters.values() if c.get('character_name')):
                character_clusters = [
                    cid for cid, cinfo in self.clusters.items() 
                    if cinfo.get('character_name') == character_name
                ]
                
                character_data = {
                    "character_name": character_name,
                    "assigned_clusters": character_clusters,
                    "total_clusters": len(character_clusters),
                    "valid_clusters": [
                        cid for cid in character_clusters 
                        if self.clusters[cid].get('cluster_status') == 'VALID'
                    ],
                    "outlier_clusters": [
                        cid for cid in character_clusters 
                        if self.clusters[cid].get('cluster_status') == 'SPATIAL_OUTLIER'
                    ],
                    "ambiguous_clusters": [
                        cid for cid in character_clusters 
                        if self.clusters[cid].get('cluster_status') == 'AMBIGUOUS'
                    ],
                    "total_faces": sum(
                        self.clusters[cid].get('face_count', 0) for cid in character_clusters
                    ),
                    "has_character_median": character_name in self.character_medians,
                    "lost_faces_to_outliers": sum(
                        self.clusters[cid].get('face_count', 0) for cid in character_clusters
                        if self.clusters[cid].get('cluster_status') == 'SPATIAL_OUTLIER'
                    )
                }
                
                tracking_data["character_assignments"][character_name] = character_data
            
            # Final statistics
            tracking_data["final_statistics"] = {
                "total_clusters": len(self.clusters),
                "valid_clusters": len([c for c in self.clusters.values() if c.get('cluster_status') == 'VALID']),
                "spatial_outliers": len([c for c in self.clusters.values() if c.get('cluster_status') == 'SPATIAL_OUTLIER']),
                "ambiguous_clusters": len([c for c in self.clusters.values() if c.get('cluster_status') == 'AMBIGUOUS']),
                "character_medians_generated": len(self.character_medians),
                "protection_failures": len([
                    c for c in tracking_data["cluster_decisions"].values() 
                    if c["protection_analysis"]["protection_failed"]
                ]),
                "successfully_protected": len([
                    c for c in tracking_data["cluster_decisions"].values() 
                    if c["protection_analysis"]["should_be_protected"] and not c["protection_analysis"]["protection_failed"]
                ])
            }
            
            # Save tracking report
            tracking_path = self.path_handler.get_face_processing_summary_path().replace('_summary.json', '_clustering_tracking.json')
            with open(tracking_path, 'w', encoding='utf-8') as f:
                json.dump(tracking_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Clustering tracking report saved: {tracking_path}")
            
            # Log critical issues
            protection_failures = tracking_data["final_statistics"]["protection_failures"]
            if protection_failures > 0:
                logger.warning(f"🚨 PROTECTION FAILURES: {protection_failures} clusters should have been protected but were marked as outliers")
                for cluster_id, cluster_data in tracking_data["cluster_decisions"].items():
                    if cluster_data["protection_analysis"]["protection_failed"]:
                        logger.warning(f"   ⚠️ Cluster {cluster_id}: {cluster_data['character_assigned']} ({cluster_data['character_percentage']:.1f}%, {cluster_data['initial_face_count']} faces) - SHOULD BE PROTECTED")
            
        except Exception as e:
            logger.error(f"❌ Error generating clustering tracking report: {e}")
            import traceback
            traceback.print_exc()
    
    def _assign_uncertain_dialogues(
        self, 
        dialogue_lines: List, 
        df_faces: pd.DataFrame
    ) -> List:
        """
        Assign uncertain dialogues based on similarity to cluster medians.
        Enhanced with multi-face LLM disambiguation when enabled.
        """
        if not self.cluster_medians:
            logger.warning("⚠️ No cluster medians available for uncertain dialogue assignment")
            return dialogue_lines
        
        # Check if multi-face processing is enabled
        enable_multiface = config.enable_multiface_processing
        logger.info(f"🔧 Multi-face uncertain assignment: {'ENABLED' if enable_multiface else 'DISABLED'}")
        
        if enable_multiface:
            return self._assign_uncertain_dialogues_multiface(dialogue_lines, df_faces)
        else:
            return self._assign_uncertain_dialogues_singleface(dialogue_lines, df_faces)
    
    def _assign_uncertain_dialogues_singleface(
        self, 
        dialogue_lines: List, 
        df_faces: pd.DataFrame
    ) -> List:
        """
        Original single-face uncertain dialogue assignment algorithm.
        """
        # Create mapping from dialogue index to face data
        face_by_dialogue = {}
        for _, row in df_faces.iterrows():
            dialogue_idx = row.get('dialogue_index', -1)
            if dialogue_idx != -1:
                if dialogue_idx not in face_by_dialogue:
                    face_by_dialogue[dialogue_idx] = []
                face_by_dialogue[dialogue_idx].append(row)
        
        assigned_count = 0
        total_uncertain = 0
        
        for dialogue in dialogue_lines:
            # Check if dialogue needs assignment
            is_uncertain = not dialogue.is_llm_confident
            
            if is_uncertain:
                total_uncertain += 1
                
                # Get face data for this dialogue
                dialogue_faces = face_by_dialogue.get(dialogue.index, [])
                
                if dialogue_faces:
                    # Select the primary face (largest, best quality)
                    primary_face = self._select_primary_face(dialogue_faces)
                    
                    if primary_face is not None and isinstance(primary_face.get('embedding'), np.ndarray):
                        # Find best matching cluster
                        best_character, best_similarity = self._find_best_character_match(
                            primary_face['embedding']
                        )
                        
                        if best_character and best_similarity >= 0.65:  # 65% similarity threshold
                            # Update dialogue assignment
                            if dialogue.original_llm_speaker is None:
                                dialogue.original_llm_speaker = dialogue.speaker
                                dialogue.original_llm_is_confident = dialogue.is_llm_confident
                            
                            dialogue.speaker = best_character
                            dialogue.is_llm_confident = True  # Now confident after face assignment
                            dialogue.resolution_method = "face_clustering_single"
                            
                            assigned_count += 1
                            logger.debug(f"🎯 [SINGLE-FACE] Assigned dialogue {dialogue.index} to '{best_character}' (similarity: {best_similarity:.3f})")
        
        if total_uncertain > 0:
            success_rate = (assigned_count / total_uncertain) * 100
            logger.info(f"📊 [SINGLE-FACE] Uncertain dialogue assignment: {assigned_count}/{total_uncertain} ({success_rate:.1f}%) successfully assigned")
        
        return dialogue_lines
    
    def _assign_uncertain_dialogues_multiface(
        self, 
        dialogue_lines: List, 
        df_faces: pd.DataFrame
    ) -> List:
        """
        Enhanced multi-face uncertain dialogue assignment using direct character median comparison.
        Removed LLM disambiguation in favor of direct vector store character median matching.
        """
        # Initialize enhanced SRT logger if available
        enhanced_srt_logger = None
        try:
            from .enhanced_srt_logger import EnhancedSRTLogger
            series = self.path_handler.get_series()
            season = self.path_handler.get_season()
            episode = self.path_handler.get_episode()
            enhanced_srt_logger = EnhancedSRTLogger(series, season, episode)
            enhanced_srt_logger.log_enhanced_srt_generation_start(len(dialogue_lines))
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize enhanced SRT logger: {e}")
        
        # Create mapping from dialogue index to face data
        face_by_dialogue = {}
        for _, row in df_faces.iterrows():
            dialogue_idx = row.get('dialogue_index', -1)
            if dialogue_idx != -1:
                if dialogue_idx not in face_by_dialogue:
                    face_by_dialogue[dialogue_idx] = []
                face_by_dialogue[dialogue_idx].append(row)
        
        assigned_count = 0
        total_uncertain = 0
        character_median_assignments = 0
        
        # Load character medians from vector store for comparison
        vector_store_character_medians = self._load_all_character_medians_from_vector_store()
        
        if not vector_store_character_medians:
            logger.warning("⚠️ No character medians loaded from vector store - using current episode medians only")
            vector_store_character_medians = self.character_medians.copy()
        
        logger.info(f"🎭 [VECTOR-STORE] Using {len(vector_store_character_medians)} character medians for face comparison: {list(vector_store_character_medians.keys())}")
        
        for dialogue in dialogue_lines:
            # Check if dialogue needs assignment
            is_uncertain = not dialogue.is_llm_confident
            
            if is_uncertain:
                total_uncertain += 1
                
                # Get face data for this dialogue
                dialogue_faces = face_by_dialogue.get(dialogue.index, [])
                
                if dialogue_faces:
                    logger.debug(f"🎭 [UNCERTAIN] Processing dialogue {dialogue.index} with {len(dialogue_faces)} faces")
                
                if dialogue_faces and enhanced_srt_logger:
                    # Log detailed face analysis
                    import time
                    start_time = time.time()
                    
                    # Analyze all faces in this dialogue
                    face_candidates = self._analyze_dialogue_faces_direct(
                        dialogue_faces, 
                        vector_store_character_medians,
                        dialogue.speaker  # Pass original LLM speaker as candidate
                    )
                    
                    # Store candidates for enhanced SRT display
                    dialogue.candidate_speakers = face_candidates.get('qualified_candidates', [])
                    dialogue.face_similarities = face_candidates.get('qualified_similarities', [])
                    dialogue.face_cluster_ids = face_candidates.get('qualified_cluster_ids', [])
                    
                    # Store ALL candidates for enhanced SRT display
                    dialogue.all_candidate_speakers = face_candidates.get('all_candidates', [])
                    dialogue.all_face_similarities = face_candidates.get('all_similarities', []) 
                    dialogue.all_face_cluster_ids = face_candidates.get('all_cluster_ids', [])
                    
                    # Log detailed analysis
                    processing_time_ms = (time.time() - start_time) * 1000
                    enhanced_srt_logger.log_dialogue_analysis(
                        dialogue.index, 
                        dialogue_faces, 
                        face_candidates, 
                        dialogue, 
                        dialogue, 
                        processing_time_ms
                    )
                    
                    logger.debug(f"🎭 [DIRECT-MEDIAN] Dialogue {dialogue.index}: {len(face_candidates.get('all_candidates', []))} total faces analyzed, {len(face_candidates.get('qualified_candidates', []))} qualified candidates")
                    
                    # Only assign if we have a valid assignment above threshold
                    if face_candidates.get('best_assignment'):
                        
                        # Store original LLM assignment
                        if dialogue.original_llm_speaker is None:
                            dialogue.original_llm_speaker = dialogue.speaker
                            dialogue.original_llm_is_confident = dialogue.is_llm_confident
                        
                        # Get the best assignment or create one from the top candidate
                        best = face_candidates.get('best_assignment')
                        if not best and face_candidates.get('qualified_candidates'):
                            # Create assignment from top qualified candidate
                            top_char = face_candidates['qualified_candidates'][0]
                            top_sim = face_candidates['qualified_similarities'][0] if face_candidates.get('qualified_similarities') else 0.0
                            best = {
                                'character': top_char,
                                'confidence': top_sim * 100,
                                'method': 'character_median_direct',
                                'similarity': top_sim
                            }
                        elif not best and face_candidates.get('all_candidates'):
                            # Create assignment from top all candidate (fallback)
                            top_char = face_candidates['all_candidates'][0]
                            top_sim = face_candidates['all_similarities'][0] if face_candidates.get('all_similarities') else 0.0
                            best = {
                                'character': top_char,
                                'confidence': top_sim * 100,
                                'method': 'character_median_direct',
                                'similarity': top_sim
                            }
                        
                        if best:
                            # Assign the character
                            dialogue.speaker = best['character']
                            dialogue.is_llm_confident = True  # Now confident after face assignment
                            dialogue.resolution_method = best['method']
                            
                            assigned_count += 1
                            if best['method'] == "character_median_direct":
                                character_median_assignments += 1
                            
                            distance = 1.0 - best.get('similarity', 0.0)
                            logger.debug(f"🎯 [DIRECT-MEDIAN] Assigned dialogue {dialogue.index} to '{best['character']}' (confidence: {best['confidence']:.1f}%, distance: {distance:.3f}, method: {best['method']})")
                        
                        # Handle multiple candidates for enhanced SRT display
                        if len(face_candidates.get('qualified_candidates', [])) > 1:
                            dialogue.resolution_method = "face_clustering_multi_unresolved"
                            logger.debug(f"🔍 [DIRECT-MEDIAN] Multiple candidates for dialogue {dialogue.index}: {face_candidates['qualified_candidates']}")
                    else:
                        # No suitable face candidates found
                        logger.debug(f"🔍 [DIRECT-MEDIAN] No suitable face candidates for dialogue {dialogue.index}")
                else:
                    # No faces detected for this dialogue
                    logger.debug(f"📷 [DIRECT-MEDIAN] No faces detected for dialogue {dialogue.index}")
        
        if total_uncertain > 0:
            success_rate = (assigned_count / total_uncertain) * 100
            median_rate = (character_median_assignments / assigned_count) * 100 if assigned_count > 0 else 0
            logger.info(f"📊 [DIRECT-MEDIAN] Uncertain dialogue assignment: {assigned_count}/{total_uncertain} ({success_rate:.1f}%) successfully assigned")
            logger.info(f"🎭 [DIRECT-MEDIAN] Character median assignments: {character_median_assignments}/{assigned_count} ({median_rate:.1f}%) used direct character median matching")
        
        # Log completion if logger is available
        if enhanced_srt_logger:
            enhanced_srt_logger.log_enhanced_srt_generation_complete("enhanced_srt_generated")
        
        return dialogue_lines
    
    def _get_episode_plot(self) -> str:
        """Get the episode plot for LLM context."""
        try:
            # Try different plot file paths in order of preference
            plot_paths = [
                self.path_handler.get_entity_normalized_plot_file_path(),
                self.path_handler.get_entity_substituted_plot_file_path(),
                self.path_handler.get_named_plot_file_path(),
                self.path_handler.get_simplified_plot_file_path(),
                self.path_handler.get_raw_plot_file_path()
            ]
            
            for plot_path in plot_paths:
                if os.path.exists(plot_path):
                    with open(plot_path, 'r', encoding='utf-8') as f:
                        plot_content = f.read().strip()
                        if plot_content:  # Make sure it's not empty
                            logger.debug(f"📖 Using plot from: {plot_path}")
                            return plot_content
                            
        except Exception as e:
            logger.warning(f"⚠️ Could not load episode plot: {e}")
        
        logger.warning("⚠️ No episode plot file found")
        return ""
    
    def _get_scene_subtitles(self, target_dialogue, all_dialogue_lines, context_window=5):
        """Get scene subtitles around the target dialogue for LLM context."""
        target_index = target_dialogue.index
        
        # Find dialogues in the same scene or nearby
        scene_lines = []
        for dialogue in all_dialogue_lines:
            if abs(dialogue.index - target_index) <= context_window:
                scene_lines.append(f"[{dialogue.index}] {dialogue.text}")
        
        return scene_lines
    
    def _select_primary_face(self, face_list: List[Dict]) -> Optional[Dict]:
        """
        Select the primary face from multiple faces in a dialogue.
        
        Prioritizes by:
        1. Face area (larger faces preferred)
        2. Detection confidence
        3. Image quality (lower blur score)
        """
        if not face_list:
            return None
        
        # Convert to list of dicts if needed
        faces = []
        for face in face_list:
            if isinstance(face, dict):
                faces.append(face)
            else:
                # Convert pandas Series to dict
                faces.append(face.to_dict())
        
        # Calculate face areas and composite scores
        for face in faces:
            # Calculate face area
            width = face.get('face_width', 0)
            height = face.get('face_height', 0)
            face['face_area'] = width * height
            
            # Composite score: area (50%) + detection confidence (30%) + quality (20%)
            area_score = face['face_area'] / 10000  # Normalize roughly
            conf_score = face.get('detection_confidence', 0) / 100
            blur_score = max(0, 1 - (face.get('blur_score', 50) / 100))  # Lower blur is better
            
            face['composite_score'] = (
                0.5 * min(area_score, 1.0) + 
                0.3 * conf_score + 
                0.2 * blur_score
            )
        
        # Select face with highest composite score
        primary_face = max(faces, key=lambda f: f.get('composite_score', 0))
        
        return primary_face

    def _load_all_character_medians_from_vector_store(self) -> Dict[str, np.ndarray]:
        """
        Load all character medians from vector store for direct comparison.
        Includes both current episode and cross-episode character medians.
        """
        try:
            collection = self.face_vector_store.get_face_collection()
            series = self.path_handler.get_series()
            
            # Get all character medians for this series
            character_median_results = collection.get(
                where={
                    "$and": [
                        {"type": "character_median"},
                        {"series": series}
                    ]
                },
                include=["metadatas", "embeddings"]
            )
            
            character_medians = {}
            
            if character_median_results["ids"]:
                for i, (id_, metadata, embedding) in enumerate(zip(
                    character_median_results["ids"], 
                    character_median_results["metadatas"], 
                    character_median_results["embeddings"]
                )):
                    character_name = metadata.get("character_name", "")
                    if character_name:
                        # Use most recent/highest confidence median for each character
                        if (character_name not in character_medians or 
                            metadata.get('avg_cluster_confidence', 0) > character_medians[character_name].get('confidence', 0)):
                            
                            character_medians[character_name] = {
                                'embedding': np.array(embedding),
                                'confidence': metadata.get('avg_cluster_confidence', 0.0),
                                'episode_code': metadata.get('episode_code', ''),
                                'face_count': metadata.get('face_count', 0)
                            }
                
                logger.info(f"🎭 [VECTOR-STORE] Loaded character medians for {len(character_medians)} characters from vector store")
                
                # Combine with current episode character medians if available
                if self.character_medians:
                    for char_name, char_median in self.character_medians.items():
                        character_medians[char_name] = {
                            'embedding': char_median,
                            'confidence': 100.0,  # Current episode has highest priority
                            'episode_code': self.path_handler.get_episode_code(),
                            'face_count': 0  # Will be updated if needed
                        }
                    logger.debug(f"🎭 [VECTOR-STORE] Combined with {len(self.character_medians)} current episode character medians")
            
            return {name: data['embedding'] for name, data in character_medians.items()}
            
        except Exception as e:
            logger.error(f"❌ Error loading character medians from vector store: {e}")
            # Fallback to current episode medians only
            if self.character_medians:
                logger.info(f"🔄 [VECTOR-STORE] Falling back to current episode character medians: {len(self.character_medians)} characters")
                return self.character_medians.copy()
            return {}

    def _analyze_dialogue_faces_direct(
        self, 
        dialogue_faces: List, 
        vector_store_character_medians: Dict[str, np.ndarray],
        original_llm_speaker: Optional[str] = None
    ) -> Dict:
        """
        Analyze faces in dialogue by comparing directly with character medians from vector store.
        
        Args:
            dialogue_faces: List of face data for this dialogue
            vector_store_character_medians: Character medians loaded from vector store
            original_llm_speaker: Original LLM speaker assignment to include as candidate
            
        Returns:
            Dictionary with candidate analysis results
        """
        if not vector_store_character_medians:
            logger.warning("⚠️ No character medians available for direct comparison")
            return {
                'qualified_candidates': [],
                'qualified_similarities': [],
                'qualified_cluster_ids': [],
                'all_candidates': [],
                'all_similarities': [],
                'all_cluster_ids': [],
                'best_assignment': None
            }
        
        logger.debug(f"🔍 [DIRECT-MEDIAN] Analyzing {len(dialogue_faces)} faces against {len(vector_store_character_medians)} character medians")
        
        # Configuration thresholds
        qualification_threshold = config.similarity_to_character_median_threshold_for_speaker_assignment
        assignment_threshold = config.similarity_to_character_median_threshold_for_speaker_assignment
        
        logger.debug(f"🔍 [DIRECT-MEDIAN] Using qualification threshold: {qualification_threshold}, assignment threshold: {assignment_threshold}")
        
        # Collect all face analyses
        all_face_results = []
        qualified_face_results = []
        cluster_assigned_faces = []
        
        for face_data in dialogue_faces:
            if not isinstance(face_data.get('embedding'), np.ndarray):
                continue
                
            face_embedding = face_data['embedding']
            cluster_id = face_data.get('face_id', face_data.get('cluster_id', -1))
            
            # Check if this face is already assigned to a cluster with a character
            if cluster_id >= 0 and cluster_id in self.clusters:
                cluster_info = self.clusters[cluster_id]
                cluster_character = cluster_info.get('character_name')
                cluster_status = cluster_info.get('cluster_status')
                
                # If face is in a valid cluster, consider it already assigned
                if (cluster_character and 
                    cluster_status not in ['SPATIAL_OUTLIER', 'SPATIAL_OUTLIER_FROM_CHARACTER_MEDIAN', 'AMBIGUOUS', 'INSUFFICIENT_EVIDENCE']):
                    
                    cluster_assigned_faces.append({
                        'character': cluster_character,
                        'confidence': cluster_info.get('cluster_confidence', 90.0),
                        'cluster_id': cluster_id,
                        'method': 'cluster_assigned',
                        'similarity': 1.0  # Already assigned, high confidence
                    })
                    
                    logger.debug(f"🔗 [DIRECT-MEDIAN] Face already assigned to cluster {cluster_id} → '{cluster_character}'")
                    continue
            
            # Compare with all character medians from vector store
            face_norm = np.linalg.norm(face_embedding)
            if face_norm <= 1e-9:
                continue
                
            normalized_face = face_embedding / face_norm
            
            best_character = None
            best_similarity = 0.0
            
            for character_name, character_median in vector_store_character_medians.items():
                # Calculate cosine similarity
                similarity = np.dot(normalized_face, character_median)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_character = character_name
            
            # Create a result for the nearest character, but only qualify if above threshold
            if best_character:
                face_result = {
                    'character': best_character,
                    'similarity': best_similarity,
                    'confidence': best_similarity * 100,
                    'cluster_id': cluster_id,
                    'method': 'character_median_direct'
                }
                
                all_face_results.append(face_result)
                
                # Only qualify faces that meet the similarity threshold
                if best_similarity >= qualification_threshold:
                    qualified_face_results.append(face_result)
                    distance = 1.0 - best_similarity
                    logger.debug(f"✅ [DIRECT-MEDIAN] Face qualifies: '{best_character}' (similarity: {best_similarity:.3f}, distance: {distance:.3f})")
                else:
                    distance = 1.0 - best_similarity
                    logger.debug(f"❌ [DIRECT-MEDIAN] Face below threshold: '{best_character}' (similarity: {best_similarity:.3f}, distance: {distance:.3f}, threshold: {qualification_threshold:.3f})")
        
        # Combine cluster-assigned faces with vector store matches
        all_candidates = cluster_assigned_faces + all_face_results
        qualified_candidates = cluster_assigned_faces + qualified_face_results
        
        # Add original LLM speaker as candidate if provided and not already present
        if original_llm_speaker and original_llm_speaker not in [c['character'] for c in qualified_candidates]:
            qualified_candidates.append({
                'character': original_llm_speaker,
                'similarity': 0.5,  # Medium similarity as fallback
                'confidence': 50.0,
                'cluster_id': -1,
                'method': 'llm_original'
            })
            logger.debug(f"➕ [DIRECT-MEDIAN] Added original LLM speaker as candidate: '{original_llm_speaker}'")
        
        # Remove duplicates and aggregate by character
        unique_candidates = {}
        for candidate in qualified_candidates:
            char_name = candidate['character']
            if char_name not in unique_candidates or candidate['confidence'] > unique_candidates[char_name]['confidence']:
                unique_candidates[char_name] = candidate
        
        unique_all_candidates = {}
        for candidate in all_candidates:
            char_name = candidate['character']
            if char_name not in unique_all_candidates or candidate['confidence'] > unique_all_candidates[char_name]['confidence']:
                unique_all_candidates[char_name] = candidate
        
        # Sort by confidence
        sorted_qualified = sorted(unique_candidates.values(), key=lambda x: x['confidence'], reverse=True)
        sorted_all = sorted(unique_all_candidates.values(), key=lambda x: x['confidence'], reverse=True)
        
        # Determine best assignment - only assign if above threshold
        best_assignment = None
        if sorted_qualified:
            top_candidate = sorted_qualified[0]
            # Only assign if above assignment threshold
            if top_candidate['similarity'] >= assignment_threshold:
                best_assignment = top_candidate
                distance = 1.0 - top_candidate['similarity']  # Convert similarity to distance
                logger.debug(f"🎯 [DIRECT-MEDIAN] Best assignment: '{top_candidate['character']}' (similarity: {top_candidate['similarity']:.3f}, distance: {distance:.3f})")
            else:
                distance = 1.0 - top_candidate['similarity']
                logger.debug(f"❌ [DIRECT-MEDIAN] Top candidate below assignment threshold: '{top_candidate['character']}' (similarity: {top_candidate['similarity']:.3f}, distance: {distance:.3f}, threshold: {assignment_threshold:.3f})")
        elif sorted_all:
            # If no qualified candidates, check if best from all candidates meets threshold
            top_candidate = sorted_all[0]
            if top_candidate['similarity'] >= assignment_threshold:
                best_assignment = top_candidate
                distance = 1.0 - top_candidate['similarity']
                logger.debug(f"🎯 [DIRECT-MEDIAN] Fallback assignment: '{top_candidate['character']}' (similarity: {top_candidate['similarity']:.3f}, distance: {distance:.3f})")
            else:
                distance = 1.0 - top_candidate['similarity']
                logger.debug(f"❌ [DIRECT-MEDIAN] Fallback candidate below assignment threshold: '{top_candidate['character']}' (similarity: {top_candidate['similarity']:.3f}, distance: {distance:.3f}, threshold: {assignment_threshold:.3f})")
        
        return {
            'qualified_candidates': [c['character'] for c in sorted_qualified],
            'qualified_similarities': [c['similarity'] for c in sorted_qualified],
            'qualified_cluster_ids': [c['cluster_id'] for c in sorted_qualified],
            'all_candidates': [c['character'] for c in sorted_all],
            'all_similarities': [c['similarity'] for c in sorted_all],
            'all_cluster_ids': [c['cluster_id'] for c in sorted_all],
            'best_assignment': best_assignment
        }
    
    def _find_best_cluster_match(self, query_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Find the best matching cluster for a face embedding.
        
        Compares against cluster medians only (not individual faces).
        """
        if not self.cluster_medians:
            return None, 0.0
        
        best_character = None
        best_similarity = 0.0
        
        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 1e-9:
            query_embedding = query_embedding / norm
        else:
            return None, 0.0
        
        # Compare against each cluster median
        for cluster_id, median_embedding in self.cluster_medians.items():
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, median_embedding)
            
            if similarity > best_similarity:
                # Get character name for this cluster
                cluster_info = self.clusters.get(cluster_id, {})
                character_name = cluster_info.get('character_name')
                cluster_status = cluster_info.get('cluster_status')
                
                # Only consider clusters with assigned characters AND valid status
                if character_name and cluster_status not in ['INSUFFICIENT_EVIDENCE', 'SPATIAL_OUTLIER', 'AMBIGUOUS']:
                    best_similarity = similarity
                    best_character = character_name
        
        return best_character, best_similarity
    
    def _find_best_character_match(self, query_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Find the best matching CHARACTER for a face embedding using character-level medians.
        
        This uses Level 2 medians (character medians) for robust inference when 
        characters have multiple clusters. Falls back to cluster-level if unavailable.
        """
        if not self.character_medians:
            logger.debug("⚠️ No character medians available, falling back to cluster matching")
            return self._find_best_cluster_match(query_embedding)
        
        best_character = None
        best_similarity = 0.0
        
        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 1e-9:
            query_embedding = query_embedding / norm
        else:
            return None, 0.0
        
        # Compare against each character median (Level 2)
        for character, character_median in self.character_medians.items():
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, character_median)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_character = character
        
        logger.debug(f"🎭 [CHARACTER-MATCH] Best character match: '{best_character}' (similarity: {best_similarity:.3f})")
        return best_character, best_similarity
    

    
    def _save_cluster_assignments_json(self, cluster_assignments: Dict) -> None:
        """
        Store cluster assignments for later consistent saving.
        
        CRITICAL: This method no longer saves files immediately. Instead, it stores
        the assignments to be saved later by _save_all_cluster_files_consistently()
        to ensure perfect data consistency across all files.
        
        Args:
            cluster_assignments: Dictionary of cluster assignments to store
        """
        # Store assignments for later consistent saving
        self._last_cluster_assignments = cluster_assignments
        
        logger.info(f"💾 [MULTI-FACE] Stored {len(cluster_assignments)} cluster assignments for consistent saving")
        
        # Log assignment summary for debugging
        for cluster_id, assignment in cluster_assignments.items():
            character = assignment.get('assigned_character', 'Unknown')
            confidence = assignment.get('confidence', 0)
            logger.debug(f"   📝 Stored: Cluster {cluster_id} → '{character}' ({confidence:.1f}%)")
    
    def _actually_save_cluster_assignments_json(self, cluster_assignments: Dict) -> None:
        """
        Actually save cluster assignments to JSON file.
        
        This method is called by _save_all_cluster_files_consistently()
        to ensure consistent timing with other file saves.
        
        Args:
            cluster_assignments: Dictionary of cluster assignments to save
        """
        cluster_assignments_path = self.path_handler.get_face_clusters_summary_path().replace('.json', '_assignments.json')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(cluster_assignments_path), exist_ok=True)
        
        assignment_data = {
            'episode_code': self.path_handler.get_episode_code(),
            'timestamp': pd.Timestamp.now().isoformat(),
            'assignment_method': 'multi_face_probability',
            'configuration': {
                'enable_multiface_processing': config.enable_multiface_processing,
                'multiface_equal_probability_distribution': config.multiface_equal_probability_distribution,
                'multiface_max_faces_per_dialogue': config.multiface_max_faces_per_dialogue
            },
            'cluster_assignments': cluster_assignments,
            'assignment_statistics': {
                'total_clusters_assigned': len(cluster_assignments),
                'assignment_method': 'multi_face_probability'
            }
        }
        
        try:
            with open(cluster_assignments_path, 'w', encoding='utf-8') as f:
                json.dump(assignment_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved cluster assignments to: {cluster_assignments_path}")
        except Exception as e:
            logger.error(f"❌ Error saving cluster assignments: {e}")
    
    def _save_cluster_info(self, df_faces: pd.DataFrame) -> None:
        """
        Save cluster information and character associations to file.
        """
        cluster_info_path = self.path_handler.get_face_clusters_summary_path()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(cluster_info_path), exist_ok=True)
        
        # Convert numpy/pandas types to native Python types for JSON serialization
        def convert_for_json(obj):
            """Convert numpy/pandas types to JSON-serializable Python types."""
            if isinstance(obj, dict):
                # Convert both keys AND values
                converted_dict = {}
                for k, v in obj.items():
                    # Convert key - handle numpy/pandas types in keys
                    if hasattr(k, 'item'):  # numpy scalar
                        converted_key = k.item()
                    elif isinstance(k, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                        converted_key = int(k)
                    elif isinstance(k, (np.floating, np.float16, np.float32, np.float64)):
                        converted_key = float(k)
                    elif isinstance(k, np.bool_):
                        converted_key = bool(k)
                    else:
                        converted_key = k
                    
                    # Convert value recursively
                    converted_value = convert_for_json(v)
                    converted_dict[converted_key] = converted_value
                return converted_dict
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif pd.isna(obj):  # pandas NaN/None
                return None
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif str(type(obj)).startswith('<class \'numpy.'):  # numpy types
                logger.debug(f"🔍 Converting numpy type {type(obj)}: {obj}")
                return obj.item() if hasattr(obj, 'item') else obj
            elif str(type(obj)).startswith('<class \'pandas.'):  # pandas types
                logger.debug(f"🔍 Converting pandas type {type(obj)}: {obj}")
                return obj.item() if hasattr(obj, 'item') else obj
            elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):  # numpy integers
                logger.debug(f"🔍 Converting numpy integer {type(obj)}: {obj}")
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):  # numpy floats
                logger.debug(f"🔍 Converting numpy float {type(obj)}: {obj}")
                return float(obj)
            elif isinstance(obj, np.bool_):  # numpy bool
                logger.debug(f"🔍 Converting numpy bool {type(obj)}: {obj}")
                return bool(obj)
            else:
                # Log potentially problematic types
                obj_type = type(obj)
                if 'int64' in str(obj_type) or 'numpy' in str(obj_type) or 'pandas' in str(obj_type):
                    logger.warning(f"⚠️ Potentially problematic type not converted: {obj_type} = {obj}")
                return obj
        
        # Prepare cluster summary data
        logger.info("🔍 Preparing cluster summary data for JSON serialization...")
        logger.info(f"   self.clusters type: {type(self.clusters)}, keys: {list(self.clusters.keys()) if self.clusters else 'None'}")
        character_clusters = self.get_character_clusters()
        logger.info(f"   character_clusters type: {type(character_clusters)}, keys: {list(character_clusters.keys()) if character_clusters else 'None'}")
        
        # Log sample cluster data to understand structure
        if self.clusters:
            sample_cluster_key = list(self.clusters.keys())[0]
            sample_cluster_value = self.clusters[sample_cluster_key]
            logger.info(f"   Sample cluster key type: {type(sample_cluster_key)}, value type: {type(sample_cluster_value)}")
            logger.info(f"   Sample cluster content: {sample_cluster_value}")
            
        if character_clusters:
            sample_char_key = list(character_clusters.keys())[0]
            sample_char_value = character_clusters[sample_char_key]
            logger.info(f"   Sample character key type: {type(sample_char_key)}, value type: {type(sample_char_value)}")
            logger.info(f"   Sample character content: {sample_char_value}")
        
        cluster_summary = {
            'episode_code': self.path_handler.get_episode_code(),
            'total_faces': int(len(df_faces)),
            'clustered_faces': int(len(df_faces[df_faces['cluster_id'] != -1])),
            'total_clusters': int(len(self.clusters)),
            'character_assignments': int(len(self.get_character_clusters())),
            'clusters': convert_for_json(self.clusters),
            'character_clusters': convert_for_json(self.get_character_clusters()),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        try:
            logger.info("🔍 Attempting JSON serialization...")
            with open(cluster_info_path, 'w', encoding='utf-8') as f:
                json.dump(cluster_summary, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved cluster information to: {cluster_info_path}")
        except Exception as e:
            logger.error(f"❌ Error saving cluster information: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            
            # Try to identify the problematic key-value pairs
            logger.info("🔍 Analyzing cluster_summary contents for problematic types...")
            for key, value in cluster_summary.items():
                try:
                    json.dumps({key: value})
                    logger.debug(f"   ✅ {key}: {type(value)} - OK")
                except Exception as field_error:
                    logger.error(f"   ❌ {key}: {type(value)} - ERROR: {field_error}")
                    if hasattr(value, '__dict__'):
                        logger.error(f"      Object attributes: {vars(value)}")
                    elif isinstance(value, dict):
                        logger.error(f"      Dict keys and types: {[(k, type(v)) for k, v in value.items()]}")
                    elif isinstance(value, list) and value:
                        logger.error(f"      List sample types: {[type(item) for item in value[:3]]}")
            
            # Also try to save a minimal version to isolate the issue
            try:
                minimal_summary = {
                    'episode_code': str(self.path_handler.get_episode_code()),
                    'total_clusters': int(len(self.clusters)),
                    'character_assignments': int(len(self.get_character_clusters()))
                }
                with open(cluster_info_path.replace('.json', '_minimal.json'), 'w') as f:
                    json.dump(minimal_summary, f, indent=2)
                logger.info("✅ Saved minimal cluster info successfully")
            except Exception as minimal_error:
                logger.error(f"❌ Even minimal save failed: {minimal_error}")
        
        # Also save cluster medians to vector store for future use
        self._save_cluster_medians_to_vector_store()
        
        # Also save character medians to vector store for cross-episode matching
        self._save_character_medians_to_vector_store()
        
        # Save character medians summary JSON for easy access
        self._save_character_medians_summary()
        
        # Save streamlined processing summary instead of verbose cluster details
        self._save_processing_summary()
    
    def _save_cluster_medians_to_vector_store(self) -> None:
        """
        Save cluster median embeddings to the vector store for future similarity searches.
        Initial clusters are marked as has_been_validated=False and will be validated later.
        Only validated clusters (has_been_validated=True) are kept for cross-episode matching.
        """
        if not self.cluster_medians:
            return
        
        try:
            # Get context from path_handler
            series = self.path_handler.get_series()
            season = self.path_handler.get_season()
            episode = self.path_handler.get_episode()
            episode_code = self.path_handler.get_episode_code()
            
            collection = self.face_vector_store.get_face_collection()
            
            # First, cleanup any unvalidated clusters from previous runs of this episode
            self._cleanup_unvalidated_clusters(series, episode_code)
            
            # Prepare cluster median data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            
            for cluster_id, median_embedding in self.cluster_medians.items():
                cluster_info = self.clusters.get(cluster_id, {})
                character_name = cluster_info.get('character_name', '')
                cluster_status = cluster_info.get('cluster_status', 'VALID')
                
                # Skip clusters that are not VALID (spatial outliers, ambiguous, etc.)
                if cluster_status != 'VALID':
                    logger.debug(f"🚫 Skipping cluster {cluster_id} with status {cluster_status}")
                    continue
                
                # Only save clusters with assigned characters
                if not character_name:
                    logger.debug(f"🚫 Skipping cluster {cluster_id} without character assignment")
                    continue
                
                # Create unique ID for cluster median
                cluster_median_id = f"{episode_code}_cluster_{cluster_id}_median"
                
                # Determine validation status - initially False, will be set to True after spatial validation
                has_been_validated = self._is_cluster_validated(cluster_id, cluster_info)
                
                # Metadata for cluster median - ensure all values are properly typed and non-None
                metadata = {
                    'type': 'cluster_median',
                    'series': str(series),
                    'season': str(season), 
                    'episode': str(episode),
                    'episode_code': str(episode_code),
                    'cluster_id': int(cluster_id),
                    'entity_name': str(character_name) if character_name else '',
                    'character_name': str(character_name) if character_name else '',
                    'face_count': int(cluster_info.get('face_count', 0)),
                    'cluster_confidence': float(cluster_info.get('cluster_confidence', 0.0)),
                    'has_been_validated': bool(has_been_validated),  # NEW: Validation status
                    'cluster_status': str(cluster_status),  # Track cluster validation status
                    'created_timestamp': pd.Timestamp.now().isoformat(),
                    'validation_timestamp': pd.Timestamp.now().isoformat() if has_been_validated else ''
                }
                
                ids.append(cluster_median_id)
                embeddings.append(median_embedding.tolist())
                metadatas.append(metadata)
            
            if ids:
                # Add cluster medians to ChromaDB
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                logger.info(f"💾 Saved {len(ids)} cluster medians to vector store")
        
        except Exception as e:
            logger.error(f"❌ Error saving cluster medians to vector store: {e}")
    
    def _save_character_medians_to_vector_store(self) -> None:
        """
        Save character median embeddings to the vector store for cross-episode character matching.
        Character medians represent the "typical" embedding for each character across all their clusters.
        """
        if not self.character_medians:
            logger.debug("📊 No character medians to save")
            return
        
        try:
            # Get context from path_handler
            series = self.path_handler.get_series()
            season = self.path_handler.get_season()
            episode = self.path_handler.get_episode()
            episode_code = self.path_handler.get_episode_code()
            
            collection = self.face_vector_store.get_face_collection()
            
            # First, cleanup any previous character medians from this episode
            self._cleanup_character_medians(series, episode_code)
            
            # Prepare character median data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            
            for character_name, character_median in self.character_medians.items():
                # Count valid clusters and total faces for this character
                character_clusters = [
                    cluster_info for cluster_id, cluster_info in self.clusters.items()
                    if cluster_info.get('character_name') == character_name 
                    and cluster_info.get('cluster_status') not in ['SPATIAL_OUTLIER', 'SPATIAL_OUTLIER_FROM_CHARACTER_MEDIAN', 'AMBIGUOUS', 'INSUFFICIENT_EVIDENCE']
                ]
                
                if not character_clusters:
                    continue
                
                total_faces = sum(self._safe_get_numeric(c, 'face_count', 0) for c in character_clusters)
                avg_confidence = sum(self._safe_get_numeric(c, 'cluster_confidence', 0) for c in character_clusters) / len(character_clusters)
                
                # Create unique ID for character median
                character_median_id = f"{episode_code}_character_{character_name.replace(' ', '_')}_median"
                
                # Metadata for character median
                metadata = {
                    'type': 'character_median',  # NEW: Character-level median
                    'series': str(series),
                    'season': str(season),
                    'episode': str(episode),
                    'episode_code': str(episode_code),
                    'entity_name': str(character_name),
                    'character_name': str(character_name),
                    'cluster_count': int(len(character_clusters)),  # Number of clusters for this character
                    'face_count': int(total_faces),  # Total faces across all clusters
                    'avg_cluster_confidence': float(avg_confidence),
                    'has_been_validated': True,  # Character medians are considered validated
                    'created_timestamp': pd.Timestamp.now().isoformat(),
                    'median_level': 'character'  # Label to distinguish from cluster medians
                }
                
                ids.append(character_median_id)
                embeddings.append(character_median.tolist())
                metadatas.append(metadata)
            
            if ids:
                # Add character medians to ChromaDB
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                logger.info(f"🎭 Saved {len(ids)} character medians to vector store for cross-episode matching")
        
        except Exception as e:
            logger.error(f"❌ Error saving character medians to vector store: {e}")
    
    def _cleanup_character_medians(self, series: str, episode_code: str) -> None:
        """
        Remove any existing character medians from this episode to avoid duplicates.
        """
        try:
            collection = self.face_vector_store.get_face_collection()
            
            # Find existing character medians for this episode
            existing_results = collection.get(
                where={
                    "$and": [
                        {"type": "character_median"},
                        {"series": series},
                        {"episode_code": episode_code}
                    ]
                },
                include=["metadatas"]
            )
            
            if existing_results["ids"]:
                collection.delete(ids=existing_results["ids"])
                logger.debug(f"🧹 Cleaned up {len(existing_results['ids'])} existing character medians for {episode_code}")
        
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning up character medians: {e}")
    
    def _save_character_medians_summary(self) -> None:
        """
        Save character medians summary JSON with validated clusters information.
        This is the key file for cross-episode character recognition.
        """
        logger.info(f"🔍 [CHARACTER-MEDIANS-DEBUG] Checking if character medians exist...")
        logger.info(f"🔍 [CHARACTER-MEDIANS-DEBUG] Character medians count: {len(self.character_medians) if self.character_medians else 0}")
        logger.info(f"🔍 [CHARACTER-MEDIANS-DEBUG] Character medians keys: {list(self.character_medians.keys()) if self.character_medians else []}")
        logger.info(f"🔍 [CHARACTER-MEDIANS-DEBUG] Cluster medians count: {len(self.cluster_medians) if hasattr(self, 'cluster_medians') else 'No cluster_medians attr'}")
        logger.info(f"🔍 [CHARACTER-MEDIANS-DEBUG] Clusters count: {len(self.clusters) if hasattr(self, 'clusters') else 'No clusters attr'}")
        
        if not self.character_medians:
            logger.warning("🎭 No character medians to save in summary - attempting to generate them now")
            
            # Try to generate character medians if they don't exist
            try:
                self._generate_character_medians()
                logger.info(f"🔍 [CHARACTER-MEDIANS-DEBUG] After generation attempt: {len(self.character_medians)} character medians")
                
                if not self.character_medians:
                    logger.error("❌ Still no character medians after generation attempt")
                    return
            except Exception as e:
                logger.error(f"❌ Failed to generate character medians: {e}")
                return
        
        try:
            episode_code = self.path_handler.get_episode_code()
            
            # Count validated clusters
            validated_clusters = [
                cluster_info for cluster_info in self.clusters.values()
                if cluster_info.get('cluster_status') in ['VALID', 'RESOLVED_AMBIGUOUS']
            ]
            
            logger.debug(f"🔍 [CHARACTER-MEDIANS] Total clusters: {len(self.clusters)}")
            logger.debug(f"🔍 [CHARACTER-MEDIANS] Validated clusters: {len(validated_clusters)}")
            logger.debug(f"🔍 [CHARACTER-MEDIANS] Character medians available: {len(self.character_medians)}")
            
            # Debug cluster statuses
            status_counts = {}
            for cluster_info in self.clusters.values():
                status = cluster_info.get('cluster_status', 'NO_STATUS')
                status_counts[status] = status_counts.get(status, 0) + 1
            logger.debug(f"🔍 [CHARACTER-MEDIANS] Cluster status breakdown: {status_counts}")
            
            summary = {
                "episode_code": episode_code,
                "timestamp": pd.Timestamp.now().isoformat(),
                "total_characters": len(self.character_medians),
                "total_validated_clusters": len(validated_clusters),
                "characters": {}
            }
            
            # Build character summary
            for character_name, character_median in self.character_medians.items():
                logger.debug(f"🔍 [CHARACTER-MEDIANS] Processing character: {character_name}")
                
                # Get all validated clusters for this character
                character_clusters = [
                    (cluster_id, cluster_info) for cluster_id, cluster_info in self.clusters.items()
                    if (cluster_info.get('character_name') == character_name and 
                        cluster_info.get('cluster_status') in ['VALID', 'RESOLVED_AMBIGUOUS'])
                ]
                
                logger.debug(f"🔍 [CHARACTER-MEDIANS] {character_name}: {len(character_clusters)} validated clusters found")
                
                if not character_clusters:
                    # Fallback: Look for ANY clusters assigned to this character (not just validated ones)
                    all_character_clusters = [
                        (cluster_id, cluster_info) for cluster_id, cluster_info in self.clusters.items()
                        if cluster_info.get('character_name') == character_name
                    ]
                    logger.warning(f"⚠️ [CHARACTER-MEDIANS] No validated clusters for {character_name}, but found {len(all_character_clusters)} total clusters")
                    
                    if all_character_clusters:
                        # Use all clusters for this character, marking them as validated for this purpose
                        character_clusters = all_character_clusters
                        logger.info(f"✅ [CHARACTER-MEDIANS] Using all {len(character_clusters)} clusters for {character_name}")
                    else:
                        logger.warning(f"⚠️ [CHARACTER-MEDIANS] No clusters found at all for {character_name}, skipping")
                        continue
                
                total_faces = sum(self._safe_get_numeric(cluster_info, 'face_count', 0) for _, cluster_info in character_clusters)
                avg_confidence = sum(self._safe_get_numeric(cluster_info, 'cluster_confidence', 0) for _, cluster_info in character_clusters) / len(character_clusters)
                
                # Build validated clusters details
                validated_clusters_details = []
                for cluster_id, cluster_info in character_clusters:
                    cluster_detail = {
                        "cluster_id": int(cluster_id),
                        "status": cluster_info.get('cluster_status', 'VALID'),  # Default to VALID if no status
                        "confidence": round(self._safe_get_numeric(cluster_info, 'cluster_confidence', 0.0), 2),
                        "face_count": int(self._safe_get_numeric(cluster_info, 'face_count', 0)),
                        "assignment_method": cluster_info.get('assignment_method', 'unknown')
                    }
                    
                    # Calculate distance from cluster median to character median
                    if cluster_id in self.cluster_medians and character_name in self.character_medians:
                        try:
                            cluster_median_embedding = self.cluster_medians[cluster_id]
                            character_median_embedding = self.character_medians[character_name]
                            
                            # Calculate cosine similarity and convert to distance
                            similarity = cosine_similarity([cluster_median_embedding], [character_median_embedding])[0][0]
                            distance = 1.0 - similarity  # Convert similarity to distance
                            
                            cluster_detail["distance_to_character_median"] = round(float(distance) if distance is not None else 0.0, 4)
                            cluster_detail["similarity_to_character_median"] = round(float(similarity) if similarity is not None else 0.0, 4)
                            
                        except Exception as e:
                            logger.warning(f"⚠️ Could not calculate distance for cluster {cluster_id} to character {character_name}: {e}")
                            cluster_detail["distance_to_character_median"] = None
                            cluster_detail["similarity_to_character_median"] = None
                    else:
                        logger.debug(f"🔍 Missing median data for cluster {cluster_id} or character {character_name}")
                        cluster_detail["distance_to_character_median"] = None
                        cluster_detail["similarity_to_character_median"] = None
                    
                    # Add resolution details for ambiguous clusters
                    if cluster_info.get('cluster_status') == 'RESOLVED_AMBIGUOUS':
                        cluster_detail.update({
                            "resolution_similarity": round(self._safe_get_numeric(cluster_info, 'resolution_similarity', 0.0), 3),
                            "original_ambiguous_characters": cluster_info.get('original_ambiguous_characters', [])
                        })
                    
                    validated_clusters_details.append(cluster_detail)
                
                # Determine median quality based on cluster count and faces
                median_quality = "high" if len(character_clusters) >= 3 and total_faces >= 20 else \
                               "medium" if len(character_clusters) >= 2 and total_faces >= 10 else "low"
                
                # Calculate distance statistics for this character
                distances = [detail["distance_to_character_median"] for detail in validated_clusters_details 
                           if detail["distance_to_character_median"] is not None]
                similarities = [detail["similarity_to_character_median"] for detail in validated_clusters_details 
                              if detail["similarity_to_character_median"] is not None]
                
                # OUTLIER CLUSTER DETECTION
                outlier_detection_threshold = config.outlier_distance_threshold
                outlier_score_threshold = config.outlier_score_threshold
                enable_outlier_detection = config.enable_outlier_cluster_detection
                
                outlier_clusters = []
                suspicious_assignment = False
                
                if enable_outlier_detection and distances:  # Check individual clusters regardless of average
                    avg_distance = sum(distances) / len(distances) if distances else 0.0
                    
                    # Check individual clusters against threshold (don't require average to exceed threshold)
                    logger.debug(f"🔍 OUTLIER CHECK: '{character_name}' avg_distance={avg_distance:.4f}, threshold={outlier_detection_threshold}")
                    
                    # Always check individual clusters for outliers
                    for detail in validated_clusters_details:
                        distance = detail.get("distance_to_character_median")
                        if distance is not None:
                            cluster_id = detail["cluster_id"]
                            face_count = detail["face_count"]
                            confidence = detail["confidence"]
                            
                            # Simple outlier detection: only check distance from character median
                            if distance > outlier_detection_threshold:
                                # Mark this character as having suspicious assignments
                                suspicious_assignment = True
                                
                                outlier_clusters.append({
                                    "cluster_id": cluster_id,
                                    "distance": distance,
                                    "face_count": face_count,
                                    "confidence": confidence,
                                    "recommendation": "LIKELY_WRONG_ASSIGNMENT"
                                })
                                
                                # ACTUALLY mark the cluster as spatial outlier from character median
                                original_character = self.clusters[cluster_id].get('character_name')
                                self.clusters[cluster_id].update({
                                    'cluster_status': 'SPATIAL_OUTLIER_FROM_CHARACTER_MEDIAN',
                                    'original_character': original_character,
                                    'outlier_reason': f'high_distance_from_character_median({distance:.4f})',
                                    'assignment_method': 'character_median_outlier_removed',
                                    'character_name': None  # Remove character assignment
                                })
                                
                                logger.warning(f"🚫 Cluster {cluster_id} MARKED AS OUTLIER: distance={distance:.4f} > {outlier_detection_threshold}")
                                
                                # Update cluster detail with outlier information
                                detail["outlier_detected"] = True
                                detail["recommendation"] = "LIKELY_WRONG_ASSIGNMENT"
                                detail["status"] = "SPATIAL_OUTLIER_FROM_CHARACTER_MEDIAN"
                            else:
                                detail["outlier_detected"] = False
                    
                    # Log suspicious assignment warning after processing all clusters
                    if suspicious_assignment:
                        logger.warning(f"🚨 SUSPICIOUS ASSIGNMENT: '{character_name}' has outlier clusters (avg_distance={avg_distance:.4f})")
                
                distance_stats = {}
                if distances:
                    distance_stats = {
                        "avg_distance_to_character_median": round(sum(distances) / len(distances), 4),
                        "max_distance_to_character_median": round(max(distances), 4),
                        "min_distance_to_character_median": round(min(distances), 4),
                        "avg_similarity_to_character_median": round(sum(similarities) / len(similarities), 4),
                        "cluster_cohesion": "high" if sum(similarities) / len(similarities) > 0.8 else 
                                          "medium" if sum(similarities) / len(similarities) > 0.6 else "low"
                    }
                else:
                    distance_stats = {
                        "avg_distance_to_character_median": None,
                        "max_distance_to_character_median": None,
                        "min_distance_to_character_median": None,
                        "avg_similarity_to_character_median": None,
                        "cluster_cohesion": "unknown"
                    }
                
                summary["characters"][character_name] = {
                    "character_median_id": f"{episode_code}_character_{character_name.replace(' ', '_')}_median",
                    "cluster_count": len(character_clusters),
                    "face_count": total_faces,
                    "avg_cluster_confidence": round(avg_confidence, 2),
                    "validated_clusters": validated_clusters_details,
                    "cross_episode_ready": True,
                    "median_quality": median_quality,
                    "suspicious_assignment": suspicious_assignment,
                    "outlier_clusters_detected": len(outlier_clusters),
                    "outlier_details": outlier_clusters if outlier_clusters else None,
                    **distance_stats  # Add distance statistics
                }
            
            # Save to file
            output_path = self.path_handler.get_character_medians_summary_path()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"🎭 Character medians summary saved: {output_path}")
            logger.info(f"   📊 {len(summary['characters'])} characters, {summary['total_validated_clusters']} validated clusters")
            
            # Log outlier detection summary
            total_suspicious = sum(1 for char_data in summary['characters'].values() 
                                 if char_data.get('suspicious_assignment', False))
            total_outlier_clusters = sum(char_data.get('outlier_clusters_detected', 0) 
                                       for char_data in summary['characters'].values())
            
            if total_suspicious > 0:
                logger.warning(f"🚨 OUTLIER DETECTION SUMMARY:")
                logger.warning(f"   📊 {total_suspicious}/{len(summary['characters'])} characters with suspicious assignments")
                logger.warning(f"   🎯 {total_outlier_clusters} clusters flagged as likely wrong assignments")
                
                for char_name, char_data in summary['characters'].items():
                    if char_data.get('suspicious_assignment', False):
                        outlier_count = char_data.get('outlier_clusters_detected', 0)
                        avg_dist = char_data.get('avg_distance_to_character_median', 0)
                        logger.warning(f"   ⚠️ '{char_name}': {outlier_count} outlier clusters, avg_distance={avg_dist}")
            else:
                logger.info(f"✅ No suspicious cluster assignments detected")
            
            # RECALCULATE CHARACTER MEDIANS after marking outliers
            if total_outlier_clusters > 0:
                logger.info(f"🔄 RECALCULATING CHARACTER MEDIANS: {total_outlier_clusters} outliers marked, recalculating without them")
                self._recalculate_character_medians_after_outlier_removal()
                logger.info(f"✅ Character medians recalculated without outlier clusters")
                
                # CRITICAL: Update vector store with clean character medians
                logger.info(f"💾 Updating vector store with clean character medians")
                self._save_character_medians_to_vector_store()
                
                # Save updated character medians summary
                logger.info(f"💾 Saving updated character medians summary after outlier removal")
                self._save_character_medians_summary_final()
            
        except Exception as e:
            logger.error(f"❌ Error saving character medians summary: {e}")
            import traceback
            traceback.print_exc()
    
    def _recalculate_character_medians_after_outlier_removal(self) -> None:
        """
        Recalculate character medians after marking outliers as SPATIAL_OUTLIER_FROM_CHARACTER_MEDIAN.
        This ensures character medians are not contaminated by outlier clusters.
        """
        try:
            logger.info("🔄 [RECALC-CHAR-MEDIANS] Recalculating character medians without outlier clusters")
            
            # Clear existing character medians
            self.character_medians.clear()
            
            # Collect valid clusters for each character (excluding all outlier types)
            character_to_clusters = defaultdict(list)
            
            for cluster_id, cluster_info in self.clusters.items():
                character_name = cluster_info.get('character_name')
                if character_name:
                    cluster_status = cluster_info.get('cluster_status', 'NO_STATUS')
                    
                    # Exclude ALL outlier types including the new one
                    if cluster_status not in ['SPATIAL_OUTLIER', 'SPATIAL_OUTLIER_FROM_CHARACTER_MEDIAN', 'AMBIGUOUS', 'INSUFFICIENT_EVIDENCE']:
                        if cluster_id in self.cluster_medians:
                            character_to_clusters[character_name].append(cluster_id)
                            logger.debug(f"✅ [RECALC] Including cluster {cluster_id} for character '{character_name}'")
                        else:
                            logger.warning(f"⚠️ [RECALC] Cluster {cluster_id} for '{character_name}' missing from cluster_medians")
                    else:
                        logger.debug(f"❌ [RECALC] Excluding cluster {cluster_id} for '{character_name}' due to status: {cluster_status}")
            
            logger.info(f"🔍 [RECALC] Character to clusters mapping after outlier removal: {dict(character_to_clusters)}")
            
            # Recalculate character medians using the same logic as before
            for character, cluster_ids in character_to_clusters.items():
                if len(cluster_ids) >= 1:
                    
                    if len(cluster_ids) == 1:
                        # Single cluster: use cluster median directly
                        cluster_median = self.cluster_medians[cluster_ids[0]]
                        character_median = cluster_median.copy()
                        logger.debug(f"🎭 [RECALC] '{character}': Single cluster {cluster_ids[0]} → direct median")
                    else:
                        # Multiple clusters: calculate face-count weighted median
                        logger.debug(f"🎭 [RECALC] '{character}': Multiple clusters {cluster_ids} → face-count weighted median")
                        
                        weighted_embeddings = []
                        total_faces = 0
                        
                        for cluster_id in cluster_ids:
                            cluster_median = self.cluster_medians[cluster_id]
                            cluster_info = self.clusters.get(cluster_id, {})
                            face_count = max(cluster_info.get('face_count', 1), 1)
                            
                            weighted_embeddings.extend([cluster_median] * face_count)
                            total_faces += face_count
                        
                        # Calculate weighted median
                        weighted_embeddings = np.array(weighted_embeddings)
                        character_median = np.median(weighted_embeddings, axis=0)
                        
                        logger.debug(f"🎭 [RECALC] '{character}': Weighted median from {len(cluster_ids)} clusters, {total_faces} total faces")
                    
                    # Store the recalculated character median
                    self.character_medians[character] = character_median
                    logger.info(f"✅ [RECALC] Recalculated median for '{character}' using {len(cluster_ids)} valid clusters")
                else:
                    logger.warning(f"⚠️ [RECALC] Character '{character}' has no valid clusters after outlier removal")
            
            logger.info(f"✅ [RECALC] Character median recalculation complete: {len(self.character_medians)} characters")
            
        except Exception as e:
            logger.error(f"❌ [RECALC] Error recalculating character medians: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_character_medians_summary_final(self) -> None:
        """
        Save final character medians summary after outlier removal.
        This generates a clean summary without the outlier clusters.
        """
        try:
            logger.info("💾 [FINAL-SUMMARY] Saving final character medians summary")
            
            # Just call the regular summary method again - it will now exclude the marked outliers
            # But first, let's rename the current summary file to preserve the before/after comparison
            
            # Move current summary to "_before_outlier_removal" version for comparison
            current_summary_path = self.path_handler.get_character_medians_summary_path()
            backup_path = current_summary_path.replace('.json', '_before_outlier_removal.json')
            
            if os.path.exists(current_summary_path):
                import shutil
                shutil.move(current_summary_path, backup_path)
                logger.info(f"📋 Backed up original summary to: {backup_path}")
            
            # Generate new summary (this will exclude outlier clusters automatically)
            self._generate_character_medians_summary_only()
            
            logger.info(f"✅ [FINAL-SUMMARY] Final character medians summary saved")
            
        except Exception as e:
            logger.error(f"❌ [FINAL-SUMMARY] Error saving final summary: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_character_medians_summary_only(self) -> None:
        """
        Generate character medians summary without outlier detection.
        Used for the final clean summary after outlier removal.
        """
        try:
            episode_code = self.path_handler.get_episode_code()
            timestamp = datetime.now().isoformat()
            
            summary = {
                "episode_code": episode_code,
                "timestamp": timestamp,
                "total_characters": 0,
                "total_validated_clusters": 0,
                "characters": {}
            }
            
            # Collect character data without doing outlier detection
            for character_name, character_median in self.character_medians.items():
                character_clusters = [
                    cid for cid, cinfo in self.clusters.items() 
                    if cinfo.get('character_name') == character_name and 
                       cinfo.get('cluster_status') not in ['SPATIAL_OUTLIER', 'SPATIAL_OUTLIER_FROM_CHARACTER_MEDIAN', 'AMBIGUOUS', 'INSUFFICIENT_EVIDENCE']
                ]
                
                if not character_clusters:
                    continue
                
                # Calculate basic statistics
                total_faces = sum(self.clusters[cid].get('face_count', 0) for cid in character_clusters)
                avg_confidence = sum(self.clusters[cid].get('cluster_confidence', 0) for cid in character_clusters) / len(character_clusters)
                
                # Build cluster details
                validated_clusters_details = []
                for cid in character_clusters:
                    cluster_info = self.clusters[cid]
                    cluster_detail = {
                        "cluster_id": cid,
                        "status": cluster_info.get('cluster_status', 'VALID'),
                        "confidence": round(cluster_info.get('cluster_confidence', 0), 2),
                        "face_count": cluster_info.get('face_count', 0),
                        "assignment_method": cluster_info.get('assignment_method', 'unknown'),
                        "distance_to_character_median": 0.0,  # Recalculated medians, so this is now 0
                        "similarity_to_character_median": 1.0
                    }
                    validated_clusters_details.append(cluster_detail)
                
                median_quality = "high" if len(character_clusters) >= 3 and total_faces >= 20 else \
                               "medium" if len(character_clusters) >= 2 and total_faces >= 10 else "low"
                
                summary["characters"][character_name] = {
                    "character_median_id": f"{episode_code}_character_{character_name.replace(' ', '_')}_median",
                    "cluster_count": len(character_clusters),
                    "face_count": total_faces,
                    "avg_cluster_confidence": round(avg_confidence, 2),
                    "validated_clusters": validated_clusters_details,
                    "cross_episode_ready": True,
                    "median_quality": median_quality,
                    "suspicious_assignment": False,  # No outliers in final summary
                    "outlier_clusters_detected": 0,
                    "outlier_details": None,
                    "avg_distance_to_character_median": 0.0,
                    "max_distance_to_character_median": 0.0,
                    "min_distance_to_character_median": 0.0,
                    "avg_similarity_to_character_median": 1.0,
                    "cluster_cohesion": "high"
                }
            
            summary["total_characters"] = len(summary["characters"])
            summary["total_validated_clusters"] = sum(len(char_data["validated_clusters"]) for char_data in summary["characters"].values())
            
            # Save final summary
            output_path = self.path_handler.get_character_medians_summary_path()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 [CLEAN-SUMMARY] Final clean summary saved: {output_path}")
            logger.info(f"   📊 {summary['total_characters']} characters, {summary['total_validated_clusters']} validated clusters")
            
        except Exception as e:
            logger.error(f"❌ [CLEAN-SUMMARY] Error generating final summary: {e}")
            import traceback
            traceback.print_exc()

    def _save_processing_summary(self) -> None:
        """
        Save streamlined processing summary with key metrics.
        Replaces the verbose cluster summary with essential information.
        """
        try:
            episode_code = self.path_handler.get_episode_code()
            
            # Calculate quality metrics
            valid_clusters = [c for c in self.clusters.values() if c.get('cluster_status') == 'VALID']
            resolved_ambiguous = [c for c in self.clusters.values() if c.get('cluster_status') == 'RESOLVED_AMBIGUOUS']
            spatial_outliers = [c for c in self.clusters.values() if c.get('cluster_status') == 'SPATIAL_OUTLIER']
            ambiguous_clusters = [c for c in self.clusters.values() if c.get('cluster_status') == 'AMBIGUOUS']
            
            total_clusters = len(self.clusters)
            utilized_clusters = len(valid_clusters) + len(resolved_ambiguous)
            utilization_rate = (utilized_clusters / total_clusters * 100) if total_clusters > 0 else 0
            
            # Count assignment methods
            assignment_counts = {
                'high_confidence_llm': 0,
                'face_recognition': 0, 
                'low_confidence': 0,
                'cross_episode': 0
            }
            
            # This would need dialogue processing info - for now use placeholder
            total_dialogues = len(getattr(self, 'dialogue_lines', []))
            
            summary = {
                "episode_code": episode_code,
                "timestamp": pd.Timestamp.now().isoformat(),
                "processing": {
                    "total_dialogues": total_dialogues,
                    "faces_detected": sum(self._safe_get_numeric(c, 'face_count', 0) for c in self.clusters.values()),
                    "clusters_formed": total_clusters,
                    "characters_identified": len(self.character_medians),
                    "processing_method": "face_clustering_with_character_medians"
                },
                "quality_metrics": {
                    "cluster_utilization": f"{utilization_rate:.1f}%",
                    "valid_clusters": len(valid_clusters),
                    "resolved_ambiguous": len(resolved_ambiguous),
                    "spatial_outliers_removed": len(spatial_outliers),
                    "unresolved_ambiguous": len(ambiguous_clusters),
                    "character_coverage": f"{len(self.character_medians)}/{total_clusters} clusters assigned",
                    "cross_episode_ready": len(self.character_medians) > 0
                },
                "cluster_status_breakdown": {
                    "VALID": len(valid_clusters),
                    "RESOLVED_AMBIGUOUS": len(resolved_ambiguous),
                    "SPATIAL_OUTLIER": len(spatial_outliers),
                    "AMBIGUOUS": len(ambiguous_clusters),
                    "OTHER": total_clusters - len(valid_clusters) - len(resolved_ambiguous) - len(spatial_outliers) - len(ambiguous_clusters)
                },
                "character_summary": {
                    character_name: {
                        "cluster_count": len([c for c in self.clusters.values() 
                                            if c.get('character_name') == character_name]),
                        "face_count": sum(self._safe_get_numeric(c, 'face_count', 0) for c in self.clusters.values() 
                                        if c.get('character_name') == character_name),
                        "avg_confidence": round(
                            sum(self._safe_get_numeric(c, 'cluster_confidence', 0) for c in self.clusters.values() 
                                if c.get('character_name') == character_name) / 
                            max(1, len([c for c in self.clusters.values() 
                                      if c.get('character_name') == character_name])), 2
                        )
                    }
                    for character_name in self.character_medians.keys()
                }
            }
            
            # Save to file using dedicated processing summary path
            output_path = self.path_handler.get_face_processing_summary_path()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Processing summary saved: {output_path}")
            logger.info(f"   📈 Cluster utilization: {utilization_rate:.1f}% ({utilized_clusters}/{total_clusters})")
            
        except Exception as e:
            logger.error(f"❌ Error saving processing summary: {e}")

    def _cleanup_unvalidated_clusters(self, series: str, episode_code: str) -> None:
        """
        Remove unvalidated clusters from previous runs of this episode.
        This ensures we don't accumulate invalid cluster medians.
        """
        try:
            collection = self.face_vector_store.get_face_collection()
            
            # Find unvalidated clusters for this episode
            unvalidated_results = collection.get(
                where={
                    "$and": [
                        {"type": "cluster_median"},
                        {"series": series},
                        {"episode_code": episode_code},
                        {"has_been_validated": False}
                    ]
                },
                include=["metadatas"]
            )
            
            if unvalidated_results["ids"]:
                # Delete unvalidated clusters
                collection.delete(ids=unvalidated_results["ids"])
                logger.info(f"🧹 Cleaned up {len(unvalidated_results['ids'])} unvalidated clusters from episode {episode_code}")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up unvalidated clusters: {e}")

    def _is_cluster_validated(self, cluster_id: int, cluster_info: Dict) -> bool:
        """
        Determine if a cluster has been validated through spatial outlier detection.
        Initially all clusters are unvalidated (False).
        After spatial validation, good clusters become validated (True).
        """
        cluster_status = cluster_info.get('cluster_status', 'VALID')
        
        # If spatial outlier detection has run and cluster is VALID, then it's validated
        if config.enable_spatial_outlier_removal or config.enable_parity_detection:
            # Only VALID clusters are considered validated
            return cluster_status == 'VALID'
        else:
            # No validation run yet - initially unvalidated
            return False

    def validate_and_finalize_clusters(self) -> None:
        """
        Mark surviving clusters as validated and remove unvalidated ones.
        This should be called after spatial outlier detection and parity detection.
        """
        logger.info("✅ Finalizing cluster validation...")
        
        try:
            collection = self.face_vector_store.get_face_collection()
            series = self.path_handler.get_series()
            episode_code = self.path_handler.get_episode_code()
            
            # Update validation status for clusters that passed validation
            validated_count = 0
            
            for cluster_id, cluster_info in self.clusters.items():
                cluster_status = cluster_info.get('cluster_status', 'VALID')
                character_name = cluster_info.get('character_name', '')
                
                # If cluster survived validation, mark as validated (only VALID status)
                if cluster_status == 'VALID' and character_name:
                    cluster_median_id = f"{episode_code}_cluster_{cluster_id}_median"
                    
                    try:
                        # Update the cluster median metadata to mark as validated
                        existing = collection.get(
                            ids=[cluster_median_id],
                            include=["metadatas"]
                        )
                        
                        if existing["ids"]:
                            # Update metadata
                            updated_metadata = existing["metadatas"][0].copy()
                            updated_metadata["has_been_validated"] = True
                            updated_metadata["validation_timestamp"] = pd.Timestamp.now().isoformat()
                            updated_metadata["cluster_status"] = cluster_status
                            
                            # Update in ChromaDB
                            collection.update(
                                ids=[cluster_median_id],
                                metadatas=[updated_metadata]
                            )
                            
                            validated_count += 1
                            logger.debug(f"✅ Validated cluster {cluster_id} for character '{character_name}'")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to update validation status for cluster {cluster_id}: {e}")
            
            # Clean up any remaining unvalidated clusters from this episode
            self._cleanup_unvalidated_clusters(series, episode_code)
            
            logger.info(f"✅ Cluster validation complete: {validated_count} clusters validated")
            
        except Exception as e:
            logger.error(f"❌ Error finalizing cluster validation: {e}")

    def load_validated_clusters_from_series(self) -> Dict[str, List[Dict]]:
        """
        Load validated clusters from all episodes in the same series.
        These can be used for cross-episode character matching.
        """
        try:
            collection = self.face_vector_store.get_face_collection()
            series = self.path_handler.get_series()
            
            # Get all validated clusters for this series
            validated_results = collection.get(
                where={
                    "$and": [
                        {"type": "cluster_median"},
                        {"series": series},
                        {"has_been_validated": True}
                    ]
                },
                include=["metadatas", "embeddings"]
            )
            
            # Group by character
            character_clusters = defaultdict(list)
            
            if validated_results["ids"]:
                for i, (id_, metadata, embedding) in enumerate(zip(
                    validated_results["ids"], 
                    validated_results["metadatas"], 
                    validated_results["embeddings"]
                )):
                    character_name = metadata.get("character_name", "")
                    if character_name:
                        character_clusters[character_name].append({
                            'id': id_,
                            'metadata': metadata,
                            'embedding': embedding,
                            'episode_code': metadata.get('episode_code', ''),
                            'cluster_confidence': metadata.get('cluster_confidence', 0.0),
                            'face_count': metadata.get('face_count', 0)
                        })
                
                logger.info(f"📚 Loaded {len(validated_results['ids'])} validated clusters for {len(character_clusters)} characters from series '{series}'")
                
                # Sort clusters by confidence for each character
                for character in character_clusters:
                    character_clusters[character].sort(
                        key=lambda x: x['cluster_confidence'], 
                        reverse=True
                    )
            
            return dict(character_clusters)
        
        except Exception as e:
            logger.error(f"❌ Error loading validated clusters from series: {e}")
            return {}
    
    def load_validated_character_medians_from_series(self) -> Dict[str, Dict]:
        """
        Load validated character medians from previous episodes in the same series.
        These provide robust character representations for cross-episode matching.
        """
        try:
            collection = self.face_vector_store.get_face_collection()
            series = self.path_handler.get_series()
            current_episode_code = self.path_handler.get_episode_code()
            
            # Get all character medians for this series (excluding current episode)
            character_median_results = collection.get(
                where={
                    "$and": [
                        {"type": "character_median"},
                        {"series": series},
                        {"episode_code": {"$ne": current_episode_code}}  # Exclude current episode
                    ]
                },
                include=["metadatas", "embeddings"]
            )
            
            # Process character medians (keep most recent/best for each character)
            character_medians = {}
            
            if character_median_results["ids"]:
                for i, (id_, metadata, embedding) in enumerate(zip(
                    character_median_results["ids"], 
                    character_median_results["metadatas"], 
                    character_median_results["embeddings"]
                )):
                    character_name = metadata.get("character_name", "")
                    if character_name:
                        # Keep the character median with highest confidence/most recent
                        if (character_name not in character_medians or 
                            metadata.get('avg_cluster_confidence', 0) > character_medians[character_name].get('avg_cluster_confidence', 0)):
                            
                            character_medians[character_name] = {
                                'id': id_,
                                'embedding': np.array(embedding),
                                'metadata': metadata,
                                'episode_code': metadata.get('episode_code', ''),
                                'cluster_count': metadata.get('cluster_count', 0),
                                'face_count': metadata.get('face_count', 0),
                                'avg_cluster_confidence': metadata.get('avg_cluster_confidence', 0.0)
                            }
                
                logger.info(f"🎭 Loaded {len(character_medians)} character medians from series '{series}'")
            
            return character_medians
        
        except Exception as e:
            logger.error(f"❌ Error loading validated character medians from series: {e}")
            return {}

    def _match_clusters_with_validated_characters(self, df_faces: pd.DataFrame) -> None:
        """
        Match current episode clusters with validated character medians from previous episodes.
        This enables robust cross-episode character consistency using character-level medians.
        """
        if not self.validated_character_medians:
            logger.info("🎭 No validated character medians available for cross-episode matching")
            return
        
        try:
            cluster_col = 'face_id' if 'face_id' in df_faces.columns else 'cluster_id'
            current_clusters = df_faces[df_faces[cluster_col] >= 0][cluster_col].unique()
            
            if len(current_clusters) == 0:
                logger.info("📚 No current clusters to match")
                return
            
            # Calculate medians for current clusters
            logger.info(f"🔍 Calculating medians for {len(current_clusters)} current clusters for character matching...")
            current_cluster_medians = self._calculate_normalized_centroids(df_faces, 'embedding', cluster_col)
            
            matches_found = 0
            similarity_threshold = config.cross_episode_character_similarity_threshold
            
            # For each current cluster, find best match with validated character medians
            for cluster_id in current_clusters:
                if cluster_id not in current_cluster_medians:
                    continue
                
                current_median = current_cluster_medians[cluster_id]
                best_match = None
                best_similarity = 0.0
                best_character = None
                
                # Compare with all validated character medians (Level 2 comparison)
                for character_name, character_info in self.validated_character_medians.items():
                    character_median = character_info['embedding']
                    
                    # Calculate cosine similarity between cluster median and character median
                    similarity = cosine_similarity([current_median], [character_median])[0][0]
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = character_info
                        best_character = character_name
                
                # If similarity is above threshold, assign character
                if best_similarity >= similarity_threshold and best_character:
                    logger.info(f"🎭 [CHARACTER-MATCH] Cluster {cluster_id} -> '{best_character}' (similarity: {best_similarity:.3f} vs character median)")
                    
                    # Pre-assign this character to the cluster
                    self.clusters[cluster_id] = {
                        'cluster_id': cluster_id,
                        'character_name': best_character,
                        'face_count': int(len(df_faces[df_faces[cluster_col] == cluster_id])),
                        'cluster_confidence': float(best_similarity * 100),
                        'assignment_method': 'cross_episode_character_median',  # NEW: Character-level matching
                        'matched_episode': best_match['episode_code'],
                        'character_median_source': best_match['episode_code'],
                        'original_cluster_count': best_match['cluster_count'],
                        'original_face_count': best_match['face_count'],
                        'character_median_confidence': best_match['avg_cluster_confidence'],
                        'cross_episode_similarity': float(best_similarity),
                        'cluster_status': 'CROSS_EPISODE_ASSIGNED'
                    }
                    
                    # Note: Character assignments now tracked in self.clusters only for consistency
                    matches_found += 1
                    
                    logger.debug(f"   📊 Matched to character median from {best_match['episode_code']} ({best_match['cluster_count']} clusters, {best_match['face_count']} faces)")
                    logger.debug(f"🔍 Cluster {cluster_id}: best similarity {best_similarity:.3f} < threshold {similarity_threshold:.3f}")
            
            if matches_found > 0:
                logger.info(f"✅ [CHARACTER-MATCH] Cross-episode character matching: {matches_found} clusters assigned via character medians")
            else:
                logger.info(f"📚 [CHARACTER-MATCH] No clusters matched character median threshold ({similarity_threshold:.2f})")
                
        except Exception as e:
            logger.error(f"❌ Error in cross-episode character matching: {e}")
            import traceback
            traceback.print_exc()

    def _enhance_cluster_assignments(self, df_faces: pd.DataFrame, cluster_col: str) -> None:
        """
        Apply cluster assignment enhancements: parity detection and spatial outlier removal.
        """
        logger.info("🔍 Starting cluster assignment enhancements...")
        
        # Step 1: Detect and handle parity (ties)
        if config.enable_parity_detection:
            parity_detected = self._detect_parity_assignments(df_faces, cluster_col)
            logger.info(f"🤔 Detected {parity_detected} clusters with parity (ties)")
        
        # Step 2: Detect and remove spatial outliers
        if config.enable_spatial_outlier_removal:
            outliers_removed = self._detect_spatial_outliers(df_faces, cluster_col)
            logger.info(f"🚫 Removed {outliers_removed} spatial outliers")
        
        logger.info("✅ Cluster assignment enhancements completed")
        
        # NOTE: Medians will be recalculated AFTER this step in the main pipeline
        # to ensure consistency across all data structures
    
    def _normalize_cluster_ids(self, df_faces: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize cluster IDs across all data structures to ensure consistency.
        
        This method ensures that:
        1. All cluster IDs are integers (not numpy types)
        2. self.clusters keys match character assignments in clusters  
        3. DataFrame cluster_id column is consistent
        4. ChromaDB metadata is consistent
        
        Args:
            df_faces: DataFrame with face data and cluster assignments
            
        Returns:
            DataFrame with normalized cluster IDs
        """
        logger.info("🔧 Normalizing cluster IDs for consistency...")
        
        # Step 1: Normalize self.clusters keys to integers
        normalized_clusters = {}
        for cluster_id, cluster_info in self.clusters.items():
            # Ensure cluster_id is an integer
            normalized_id = int(cluster_id)
            cluster_info['cluster_id'] = normalized_id  # Ensure internal consistency
            normalized_clusters[normalized_id] = cluster_info
        self.clusters = normalized_clusters
        
        # Step 2: Character clusters are now dynamically generated, no normalization needed
        
        # Step 3: Verify consistency between data structures
        cluster_ids_in_clusters = set(self.clusters.keys())
        character_clusters = self.get_character_clusters()
        cluster_ids_in_char_clusters = set()
        for cluster_list in character_clusters.values():
            cluster_ids_in_char_clusters.update(cluster_list)
        
        # Log any inconsistencies
        missing_from_clusters = cluster_ids_in_char_clusters - cluster_ids_in_clusters
        missing_from_char_clusters = cluster_ids_in_clusters - cluster_ids_in_char_clusters
        
        if missing_from_clusters:
            logger.warning(f"⚠️ Cluster IDs in character_clusters but not in clusters: {missing_from_clusters}")
        if missing_from_char_clusters:
            logger.warning(f"⚠️ Cluster IDs in clusters but not assigned to characters: {missing_from_char_clusters}")
        
        # Step 4: Normalize DataFrame cluster IDs
        cluster_col = 'face_id' if 'face_id' in df_faces.columns else 'cluster_id'
        if cluster_col in df_faces.columns:
            df_faces[cluster_col] = df_faces[cluster_col].astype(int)
            df_faces['cluster_id'] = df_faces[cluster_col]  # Ensure both columns exist and are consistent
        
        # Step 5: Log normalization summary
        logger.info(f"✅ Cluster ID normalization completed:")
        logger.info(f"   📊 {len(self.clusters)} clusters normalized")
        character_clusters = self.get_character_clusters()
        logger.info(f"   🎭 {len(character_clusters)} character assignments normalized")
        logger.info(f"   📋 DataFrame cluster column normalized: {cluster_col}")
        
        return df_faces
    
    def _save_all_cluster_files_consistently(self, df_faces: pd.DataFrame) -> None:
        """
        Save all cluster-related files with consistent data to prevent inconsistencies.
        
        CRITICAL: This method saves all files at once using the same data state
        to ensure perfect consistency across all output files.
        
        Args:
            df_faces: DataFrame with final cluster assignments
        """
        logger.info("💾 Saving all cluster files with consistent data...")
        
        try:
            # Normalize all data structures one final time before saving
            df_faces = self._normalize_cluster_ids(df_faces)
            
            # Save all files in sequence with the same data state
            logger.info("💾 Step 8.1: Saving cluster information summary")
            self._save_cluster_info(df_faces)
        
            logger.info("💾 Step 8.2: Saving cluster assignments JSON")
            if hasattr(self, '_last_cluster_assignments'):
                self._actually_save_cluster_assignments_json(self._last_cluster_assignments)
            
            logger.info("💾 Step 8.3: Saving character medians summary")
            self._save_character_medians_summary()
            
            logger.info("💾 Step 8.4: Saving cluster medians to vector store")
            self._save_cluster_medians_to_vector_store()
            
            logger.info("💾 Step 8.5: Saving character medians to vector store")
            self._save_character_medians_to_vector_store()
            
            logger.info("💾 Step 8.6: Saving processing summary")
            self._save_processing_summary()
            
            logger.info("💾 Step 8.7: Generating clustering tracking report")
            self._generate_clustering_tracking_report()
            
            # Verification step: Check for consistency across saved files
            logger.info("🔍 Step 8.8: Verifying file consistency")
            self._verify_file_consistency()
            
            logger.info("✅ All cluster files saved consistently")
            
        except Exception as e:
            logger.error(f"❌ Error saving cluster files consistently: {e}")
            raise
    
    def _verify_file_consistency(self) -> None:
        """
        Verify that all saved files have consistent cluster assignments.
        
        This method performs a quick consistency check to catch any data
        integrity issues before they become user-visible problems.
        """
        try:
            logger.info("🔍 Verifying cluster file consistency...")
            
            # Get cluster assignments from different files
            cluster_summary_path = self.path_handler.get_face_clusters_summary_path()
            cluster_assignments_path = cluster_summary_path.replace('.json', '_assignments.json')
            character_medians_path = self.path_handler.get_character_medians_summary_path()
            
            inconsistencies = []
            
            # Check if files exist
            if os.path.exists(cluster_summary_path) and os.path.exists(character_medians_path):
                # Load character assignments from both files
                with open(cluster_summary_path, 'r', encoding='utf-8') as f:
                    cluster_summary = json.load(f)
                    
                with open(character_medians_path, 'r', encoding='utf-8') as f:
                    character_medians = json.load(f)
                
                # Compare character cluster assignments
                summary_char_clusters = cluster_summary.get('character_clusters', {})
                
                for character_name, cluster_data in character_medians.get('characters', {}).items():
                    validated_clusters = cluster_data.get('validated_clusters', [])
                    if validated_clusters:
                        # Get the main cluster ID for this character
                        main_cluster_id = validated_clusters[0].get('cluster_id')
                        summary_cluster_id = summary_char_clusters.get(character_name)
                        
                        if main_cluster_id != summary_cluster_id:
                            inconsistencies.append(
                                f"Character '{character_name}': medians file shows cluster {main_cluster_id}, "
                                f"summary file shows cluster {summary_cluster_id}"
                            )
            
            if inconsistencies:
                logger.error(f"❌ Found {len(inconsistencies)} file consistency issues:")
                for issue in inconsistencies:
                    logger.error(f"   • {issue}")
                logger.error("❌ DATA INTEGRITY PROBLEM DETECTED - Files are inconsistent!")
            else:
                logger.info("✅ File consistency verification passed")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not verify file consistency: {e}")

    def _detect_parity_assignments(self, df_faces: pd.DataFrame, cluster_col: str) -> int:
        """
        Detect clusters with tied character assignments (parity) and mark them as AMBIGUOUS.
        """
        logger.debug("🔍 Detecting parity in cluster assignments...")
        parity_count = 0
        
        # Re-analyze each cluster for ties
        valid_clusters = df_faces[df_faces[cluster_col] >= 0][cluster_col].unique()
        
        for cluster_id in valid_clusters:
            if cluster_id not in self.clusters:
                continue
                
            # Get all faces in this cluster with confident assignments
            cluster_faces = df_faces[
                (df_faces[cluster_col] == cluster_id) & 
                (df_faces['is_llm_confident'] == True) &
                (df_faces['speaker'].notna()) &
                (df_faces['speaker'] != '')
            ]
            
            if cluster_faces.empty:
                continue
            
            # Count speaker occurrences
            speaker_counts = cluster_faces['speaker'].value_counts().to_dict()
            
            if len(speaker_counts) >= 2:
                # Check for ties (same count for multiple speakers)
                sorted_counts = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)
                
                if len(sorted_counts) >= 2 and sorted_counts[0][1] == sorted_counts[1][1]:
                    # Parity detected!
                    tied_speakers = [speaker for speaker, count in sorted_counts if count == sorted_counts[0][1]]
                    
                    logger.debug(f"🤔 Parity detected in cluster {cluster_id}: {tied_speakers} (each with {sorted_counts[0][1]} faces)")
                    
                    # Mark as ambiguous but keep the original character name for reference
                    original_character = self.clusters[cluster_id].get('character_name', 'Unknown')
                    self.clusters[cluster_id].update({
                        'cluster_status': 'AMBIGUOUS',
                        'cluster_confidence': None,  # No meaningful confidence for ambiguous clusters
                        'ambiguous_characters': tied_speakers,
                        'parity_count': sorted_counts[0][1],
                        'assignment_method': 'parity_detected',
                        'original_speaker_breakdown': speaker_counts,
                        'original_character_assignment': original_character
                    })
                    
                    # Note: Character assignments are now dynamically tracked based on cluster status
                    
                    parity_count += 1
        
        return parity_count

    def _detect_spatial_outliers(self, df_faces: pd.DataFrame, cluster_col: str) -> int:
        """
        Detect and remove clusters that are spatially far from a character's main cluster group.
        """
        logger.debug("🚫 Detecting spatial outliers...")
        outliers_removed = 0
        
        # Group clusters by assigned character (excluding already processed special cases)
        character_cluster_map = defaultdict(list)
        
        for cluster_id, cluster_info in self.clusters.items():
            character_name = cluster_info.get('character_name')
            if character_name and character_name not in ['AMBIGUOUS', 'SPATIAL_OUTLIER']:
                character_cluster_map[character_name].append(cluster_id)
        
        # Calculate cluster centroids
        cluster_centroids = self._calculate_normalized_centroids(df_faces, 'embedding', cluster_col)
        
        # Process each character with multiple clusters
        for character, cluster_ids in character_cluster_map.items():
            if len(cluster_ids) >= config.min_clusters_for_outlier_detection:
                outlier_clusters = self._find_spatial_outliers_for_character(
                    cluster_ids, cluster_centroids, character
                )
                
                for outlier_cluster_id in outlier_clusters:
                    logger.debug(f"🚫 Marking cluster {outlier_cluster_id} as spatial outlier for character '{character}'")
                    
                    # Mark as spatial outlier but keep the original character name
                    self.clusters[outlier_cluster_id].update({
                        'cluster_status': 'SPATIAL_OUTLIER',
                        'original_character': character,
                        'outlier_reason': 'spatially_detached',
                        'assignment_method': 'spatial_outlier_removed'
                    })
                    
                    # Note: Character assignments are now dynamically tracked based on cluster status
                    
                    outliers_removed += 1
        
        return outliers_removed

    def _find_spatial_outliers_for_character(self, cluster_ids: List[int], cluster_centroids: dict, character: str) -> List[int]:
        """
        Find clusters that are spatially far from the character's median centroid.
        Better approach: calculate character's median centroid, then find clusters far from it.
        """
        if len(cluster_ids) < config.min_clusters_for_outlier_detection:
            return []
        
        # Get centroids for this character's clusters
        character_centroids = {}
        for cluster_id in cluster_ids:
            if cluster_id in cluster_centroids:
                character_centroids[cluster_id] = cluster_centroids[cluster_id]
        
        if len(character_centroids) < config.min_clusters_for_outlier_detection:
            return []
        
        # Calculate the median centroid for this character
        # This represents the "typical" face for this character
        centroid_vectors = list(character_centroids.values())
        character_median_centroid = np.median(centroid_vectors, axis=0)
        
        # Calculate distance from each cluster to the character's median centroid
        distances_from_median = {}
        for cluster_id, centroid in character_centroids.items():
            # Calculate cosine distance from character's median centroid
            distance = 1.0 - np.dot(centroid, character_median_centroid) / (
                np.linalg.norm(centroid) * np.linalg.norm(character_median_centroid)
            )
            distances_from_median[cluster_id] = distance
        
        # Calculate threshold for outlier detection
        distances = list(distances_from_median.values())
        median_distance = np.median(distances)
        base_threshold = config.spatial_outlier_threshold  # 0.45
        
        # Adaptive threshold: be more strict if distances are generally small
        if median_distance < 0.2:
            # Character clusters are very tight - use base threshold
            threshold = base_threshold
        elif median_distance < 0.3:
            # Character clusters are reasonably tight - slightly more lenient
            threshold = max(base_threshold, median_distance * 2.0)
        else:
            # Character clusters are spread - be more lenient
            threshold = max(base_threshold, median_distance * 1.5)
        
        logger.debug(f"   🔍 Character '{character}' median-based spatial analysis:")
        logger.debug(f"      Clusters: {list(cluster_ids)}")
        logger.debug(f"      Distances from median: {[f'{d:.3f}' for d in distances]}")
        logger.debug(f"      Median distance: {median_distance:.3f}")
        logger.debug(f"      Base threshold: {base_threshold:.3f}")
        logger.debug(f"      Adaptive threshold: {threshold:.3f}")
        
        # Find outliers - clusters far from character's median centroid
        outliers = []
        for cluster_id, distance in distances_from_median.items():
            if distance > threshold:
                # CHECK: Don't mark as outlier if this is a high-quality cluster
                # Get cluster information to check character percentage and face count
                cluster_info = self.clusters.get(cluster_id, {})
                speaker_breakdown = cluster_info.get('speaker_breakdown', {})
                character_occurrences = speaker_breakdown.get(character, 0)
                total_cluster_probability = cluster_info.get('total_cluster_probability', 1)
                face_count = cluster_info.get('face_count', 0)
                
                # Calculate character percentage in cluster using actual speaker breakdown
                character_percentage = (character_occurrences / total_cluster_probability * 100) if total_cluster_probability > 0 else 0
                
                # OVERRIDE: Don't mark as outlier if character percentage > threshold and substantial size
                protection_percentage = config.cluster_protection_percentage_threshold
                protection_min_faces = config.cluster_protection_min_faces
                
                logger.debug(f"      🔍 Cluster {cluster_id} protection analysis:")
                logger.debug(f"         Character: {character}")
                logger.debug(f"         Character occurrences: {character_occurrences}")
                logger.debug(f"         Total cluster probability: {total_cluster_probability}")
                logger.debug(f"         Character percentage: {character_percentage:.1f}%")
                logger.debug(f"         Face count: {face_count}")
                logger.debug(f"         Protection thresholds: {protection_percentage}% / {protection_min_faces} faces")
                
                if character_percentage > protection_percentage and face_count >= protection_min_faces:
                    logger.info(f"      🛡️ Cluster {cluster_id} PROTECTED from outlier marking:")
                    logger.info(f"         Character: {character} ({character_percentage:.1f}% of cluster)")
                    logger.info(f"         Faces: {face_count}, Distance: {distance:.3f}")
                    logger.info(f"         Reason: High character percentage (>{protection_percentage}%) + substantial size (>={protection_min_faces})")
                    logger.debug(f"      Cluster {cluster_id} distance: {distance:.3f} > {threshold:.3f} (outlier BUT PROTECTED)")
                else:
                    logger.debug(f"      Cluster {cluster_id} distance: {distance:.3f} > {threshold:.3f} (outlier)")
                    logger.debug(f"         Character %: {character_percentage:.1f}%, Faces: {face_count}")
                    logger.info(f"      ❌ Cluster {cluster_id} NOT PROTECTED - marking as outlier:")
                    logger.info(f"         Percentage check: {character_percentage:.1f}% > {protection_percentage}% = {character_percentage > protection_percentage}")
                    logger.info(f"         Face count check: {face_count} >= {protection_min_faces} = {face_count >= protection_min_faces}")
                    outliers.append(cluster_id)
            else:
                logger.debug(f"      Cluster {cluster_id} distance: {distance:.3f} ≤ {threshold:.3f} (keep)")
        
        return outliers
    
    def _resolve_ambiguous_clusters(self) -> int:
        """
        Resolve ambiguous clusters using character median similarity.
        
        For clusters marked as AMBIGUOUS due to tied character assignments,
        compare the cluster median against character medians to determine
        the best assignment based on embedding similarity.
        
        Returns:
            Number of ambiguous clusters resolved
        """
        if not self.character_medians or not self.cluster_medians:
            logger.warning("⚠️ Cannot resolve ambiguous clusters: missing medians")
            return 0
        
        resolved_count = 0
        ambiguous_clusters = []
        
        # Find all ambiguous clusters
        for cluster_id, cluster_info in self.clusters.items():
            if cluster_info.get('cluster_status') == 'AMBIGUOUS':
                ambiguous_characters = cluster_info.get('ambiguous_characters', [])
                if len(ambiguous_characters) >= 2 and cluster_id in self.cluster_medians:
                    ambiguous_clusters.append((cluster_id, ambiguous_characters))
        
        if not ambiguous_clusters:
            return 0
        
        logger.info(f"🔍 Found {len(ambiguous_clusters)} ambiguous clusters to resolve")
        
        # Resolve each ambiguous cluster
        for cluster_id, ambiguous_characters in ambiguous_clusters:
            cluster_median = self.cluster_medians[cluster_id]
            
            # Calculate similarity to each character median
            character_similarities = {}
            for character in ambiguous_characters:
                if character in self.character_medians:
                    char_median = self.character_medians[character]
                    similarity = np.dot(cluster_median, char_median)
                    character_similarities[character] = similarity
            
            if character_similarities:
                # Find character with highest similarity
                best_character = max(character_similarities.items(), key=lambda x: x[1])
                character, similarity = best_character
                
                # Check if similarity meets threshold
                threshold = config.ambiguous_resolution_threshold
                if similarity >= threshold:
                    # Resolve ambiguous cluster to this character
                    logger.info(f"✅ [AMBIGUOUS-RESOLUTION] Cluster {cluster_id}: {ambiguous_characters} → '{character}' (similarity: {similarity:.3f})")
                    
                    # Note: Character assignments now tracked in self.clusters only for consistency
                    self.clusters[cluster_id].update({
                        'character_name': character,
                        'cluster_status': 'RESOLVED_AMBIGUOUS',
                        'cluster_confidence': similarity * 100,
                        'assignment_method': 'ambiguous_resolution',
                        'original_ambiguous_characters': ambiguous_characters,
                        'resolution_similarity': float(similarity),
                        'resolution_threshold': float(threshold)
                    })
                    
                    resolved_count += 1
                else:
                    logger.info(f"🤔 [AMBIGUOUS-RESOLUTION] Cluster {cluster_id}: similarity {similarity:.3f} < threshold {threshold:.3f}, keeping ambiguous")
            else:
                logger.warning(f"⚠️ [AMBIGUOUS-RESOLUTION] Cluster {cluster_id}: no character medians available for {ambiguous_characters}")
        
        return resolved_count

    def get_character_clusters(self) -> Dict[str, List[int]]:
        """
        Dynamically generate character-to-clusters mapping from self.clusters.
        
        This replaces the problematic self.character_clusters dict that caused inconsistency.
        
        Returns:
            Dictionary mapping character names to lists of cluster IDs
        """
        character_clusters = {}
        
        for cluster_id, cluster_info in self.clusters.items():
            character_name = cluster_info.get('character_name')
            if character_name and cluster_info.get('cluster_status') not in ['SPATIAL_OUTLIER', 'SPATIAL_OUTLIER_FROM_CHARACTER_MEDIAN', 'AMBIGUOUS', 'INSUFFICIENT_EVIDENCE']:
                if character_name not in character_clusters:
                    character_clusters[character_name] = []
                character_clusters[character_name].append(cluster_id)
        
        return character_clusters
    
    def _perform_sex_validation(self, df_faces: pd.DataFrame, dialogue_lines: List) -> None:
        """
        Perform sex-based validation of speaker clusters using DeepFace.
        
        This method:
        1. Loads character sex data from the speaker identification pipeline
        2. Validates each cluster's sex against assigned character's biological sex
        3. Invalidates clusters with sex mismatches
        4. Attempts reassignment to same-sex characters when possible
        
        Args:
            df_faces: DataFrame containing face data with cluster assignments
            dialogue_lines: List of dialogue lines (to extract character data)
        """
        if not self.sex_validator.is_available():
            logger.info("🧬 [SEX-VALIDATION] Skipped - not available or disabled")
            return
        
        logger.info("🧬 [SEX-VALIDATION] Starting sex-based cluster validation")
        
        # Step 1: Load character sex data
        try:
            # Get character data from speaker identification pipeline
            from .speaker_identification_pipeline import SpeakerIdentificationPipeline
            from ..narrative_storage_management.repositories import DatabaseSessionManager
            from ..narrative_storage_management.character_service import CharacterService
            from ..narrative_storage_management.repositories import CharacterRepository
            
            # Extract series information
            series = self.path_handler.get_series()
            
            # Load character data with biological sex information
            db_manager = DatabaseSessionManager()
            with db_manager.session_scope() as session:
                character_service = CharacterService(CharacterRepository(session))
                characters = character_service.get_episode_entities(series)
                
                # Convert to plain data while session is active (including biological_sex)
                characters_data = []
                for char in characters:
                    appellations_data = [app.appellation for app in char.appellations]
                    char_data = {
                        'entity_name': char.entity_name,
                        'best_appellation': char.best_appellation,
                        'appellations': appellations_data,
                        'series': char.series,
                        'biological_sex': char.biological_sex
                    }
                    characters_data.append(char_data)
            
            # Load sex data into validator
            self.sex_validator.load_character_sex_data(characters_data)
            
        except Exception as e:
            logger.warning(f"⚠️ [SEX-VALIDATION] Failed to load character sex data: {e}")
            return
        
        # Step 2: Perform validation
        try:
            validation_results = self.sex_validator.validate_all_clusters(
                df_faces, self.clusters, self.character_medians
            )
            
            if not validation_results['validation_enabled']:
                return
            
            # Step 3: Apply validation results
            clusters_invalidated = 0
            clusters_reassigned = 0
            
            for cluster_id, validation_detail in validation_results['validation_details'].items():
                action = validation_detail['action']
                
                if action == 'INVALIDATE':
                    # Invalidate the cluster
                    if cluster_id in self.clusters:
                        self.clusters[cluster_id]['cluster_status'] = 'SEX_VALIDATION_FAILED'
                        self.clusters[cluster_id]['sex_validation_reason'] = validation_detail['reason']
                        clusters_invalidated += 1
                        
                        if config.enable_sex_validation_logging:
                            logger.info(f"❌ [SEX-VALIDATION] Invalidated cluster {cluster_id}: {validation_detail['reason']}")
            
            # Step 4: Apply reassignments
            for cluster_id, reassignment in validation_results.get('reassignment_actions', {}).items():
                if cluster_id in self.clusters:
                    old_character = reassignment['from_character']
                    new_character = reassignment['to_character']
                    
                    # Reassign cluster to new character
                    self.clusters[cluster_id]['character_name'] = new_character
                    self.clusters[cluster_id]['cluster_status'] = 'SEX_VALIDATION_REASSIGNED'
                    self.clusters[cluster_id]['sex_reassignment_reason'] = reassignment['reason']
                    self.clusters[cluster_id]['sex_reassignment_similarity'] = reassignment['similarity']
                    self.clusters[cluster_id]['original_character'] = old_character
                    
                    clusters_reassigned += 1
                    
                    if config.enable_sex_validation_logging:
                        logger.info(f"🔄 [SEX-VALIDATION] Reassigned cluster {cluster_id}: {old_character} → {new_character} (similarity: {reassignment['similarity']:.3f})")
            
            # Log summary
            logger.info(f"🧬 [SEX-VALIDATION] Validation completed:")
            logger.info(f"   📊 Clusters validated: {validation_results['clusters_validated']}")
            logger.info(f"   ✅ Clusters kept: {validation_results['clusters_kept']}")
            logger.info(f"   ❌ Clusters invalidated: {clusters_invalidated}")
            logger.info(f"   🔄 Clusters reassigned: {clusters_reassigned}")
            
        except Exception as e:
            logger.error(f"❌ [SEX-VALIDATION] Validation failed: {e}")
            import traceback
            logger.error(f"❌ [SEX-VALIDATION] Traceback: {traceback.format_exc()}")
    
    def get_primary_character_cluster(self, character_name: str) -> Optional[int]:
        """
        Get the primary (highest confidence) cluster for a character.
        
        Args:
            character_name: Name of the character
            
        Returns:
            Cluster ID with highest confidence for this character, or None if not found
        """
        character_clusters = []
        
        for cluster_id, cluster_info in self.clusters.items():
            if cluster_info.get('character_name') == character_name:
                confidence = cluster_info.get('cluster_confidence', 0)
                character_clusters.append((cluster_id, confidence))
        
        if character_clusters:
            # Return cluster with highest confidence
            return max(character_clusters, key=lambda x: x[1])[0]
        
        return None
