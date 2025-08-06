"""
Face cluster visualization for SEMAMORPH speaker identification.
Creates 2D PCA visualizations of face embeddings and clusters.
Integrates with ChromaDB for face embedding storage and retrieval.
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm

from .face_embedding_vector_store import FaceEmbeddingVectorStore
from .face_clustering_system import FaceClusteringSystem
from ..utils.logger_utils import setup_logging
from ..config import config

logger = setup_logging(__name__)

class FaceClusterVisualizer:
    """Creates visualizations for face clusters and embeddings."""
    
    def __init__(self, path_handler):
        self.path_handler = path_handler
        self.face_vector_store = FaceEmbeddingVectorStore(path_handler)
        self.face_clustering_system = FaceClusteringSystem(path_handler, self.face_vector_store)
    
    def create_cluster_visualizations(
        self,
        df_faces: pd.DataFrame,
        output_format: str = "matplotlib",  # Only matplotlib - no Plotly
        save_plots: bool = True
    ) -> Dict[str, str]:
        """
        Create 2D PCA visualizations of face clusters.
        
        Args:
            df_faces: DataFrame with face data and embeddings
            output_format: Format for visualizations ("matplotlib" only)
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        if df_faces.empty or 'embedding' not in df_faces.columns:
            logger.warning("⚠️ No face embeddings available for visualization")
            return {}
        
        # Filter valid embeddings with better validation
        valid_mask = df_faces['embedding'].apply(
            lambda emb: isinstance(emb, np.ndarray) and emb.ndim == 1 and emb.size > 0 and np.linalg.norm(emb) > 1e-9
        )
        df_valid = df_faces[valid_mask].copy()
        
        if len(df_valid) < 2:
            logger.warning("⚠️ Not enough faces for clustering visualization")
            return {}
        
        logger.info(f"📊 Creating cluster visualizations for {len(df_valid)} faces")
        
        # Prepare embedding matrix with validation
        try:
            embeddings_matrix = np.vstack(df_valid['embedding'].values)
            if embeddings_matrix.shape[0] != len(df_valid):
                logger.error("❌ Embedding matrix shape mismatch")
                return {}
        except Exception as e:
            logger.error(f"❌ Error creating embedding matrix: {e}")
            return {}
        
        # Standardize embeddings
        try:
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings_matrix)
        except Exception as e:
            logger.error(f"❌ Error standardizing embeddings: {e}")
            return {}
        
        # Apply PCA for 2D visualization with error handling
        try:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings_scaled)
            
            # Validate PCA results
            if embeddings_2d.shape[1] != 2:
                logger.error("❌ PCA did not produce 2D output")
                return {}
                
            # Add PCA coordinates to dataframe
            df_valid['pca_x'] = embeddings_2d[:, 0]
            df_valid['pca_y'] = embeddings_2d[:, 1]
            
        except Exception as e:
            logger.error(f"❌ Error during PCA transformation: {e}")
            return {}
        
        # Create cluster labels if not already present - ALWAYS create clustering
        if 'cluster_id' not in df_valid.columns and 'face_id' not in df_valid.columns:
            df_valid = self._create_cluster_labels(df_valid, embeddings_scaled)
        elif 'face_id' in df_valid.columns and 'cluster_id' not in df_valid.columns:
            # Use face_id as cluster_id if available
            df_valid['cluster_id'] = df_valid['face_id']
        elif 'cluster_id' not in df_valid.columns:
            # Force create clustering if no cluster info available
            df_valid = self._create_cluster_labels(df_valid, embeddings_scaled)
        
        # Ensure we have a face_id column for downstream processing
        if 'face_id' not in df_valid.columns:
            df_valid['face_id'] = df_valid.get('cluster_id', range(len(df_valid)))
        
        # Setup output directory
        viz_dir = self.path_handler.get_cluster_visualization_dir()
        os.makedirs(viz_dir, exist_ok=True)
        
        plot_paths = {}
        
        # Create 2D visualization with face images - reference style only
        reference_plot_path = self.visualize_cluster_centroids_2d(
            df_clustered=df_valid,
            expected_embedding_dim=512,
            reducer_type='pca',
            save_figure=save_plots,
            output_filename=f"{self.path_handler.get_episode_code()}_face_clusters_2d_pca.png"
        )
        if reference_plot_path:
            plot_paths['reference_2d_pca'] = reference_plot_path
        
        # Create cluster grids for detailed view
        self.show_all_clusters(
            df_clustered=df_valid,
            max_images_per_cluster=60,
            cols=8,
            save_figures=save_plots
        )
        
        # Add grid count to paths
        if save_plots:
            grid_files = [f for f in os.listdir(viz_dir) if f.startswith('Cluster_') and f.endswith('.png')]
            if grid_files:
                plot_paths['cluster_grids'] = f"{len(grid_files)} cluster grid files"
        
        # Save cluster summary
        self._save_cluster_summary(df_valid, viz_dir)
        
        logger.info(f"✅ Created {len(plot_paths)} visualization plots")
        return plot_paths
    
    def _create_cluster_labels(self, df_valid: pd.DataFrame, embeddings_scaled: np.ndarray) -> pd.DataFrame:
        """Create cluster labels using HDBSCAN with parameters matching your pipeline config."""
        try:
            # Use your pipeline configuration parameters
            min_cluster_size = 2  # min_cluster_size_final from your config
            cosine_similarity_threshold = 0.70  # From your config
            
            # Use cosine metric to match your approach
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,  # Aggressive clustering like your pipeline
                metric='cosine',
                cluster_selection_epsilon=1.0 - cosine_similarity_threshold,  # 0.30
                cluster_selection_method='eom'
            )
            
            cluster_labels = clusterer.fit_predict(embeddings_scaled)
            df_valid['cluster_id'] = cluster_labels
            
            # Apply additional filtering based on cosine similarity like your pipeline
            if len(set(cluster_labels[cluster_labels >= 0])) > 1:
                refined_labels = self._refine_clusters_by_similarity(
                    df_valid, embeddings_scaled, cluster_labels, cosine_similarity_threshold
                )
                df_valid['cluster_id'] = refined_labels
            
            num_clusters = len(set(df_valid['cluster_id'][df_valid['cluster_id'] >= 0]))
            logger.info(f"📊 Created {num_clusters} clusters using cosine similarity threshold {cosine_similarity_threshold}")
            
        except Exception as e:
            logger.warning(f"⚠️ HDBSCAN clustering failed, trying fallback: {e}")
            # Fall back to speaker-based grouping
            speaker_to_cluster = {}
            
            if 'speaker' in df_valid.columns:
                for i, speaker in enumerate(df_valid['speaker'].unique()):
                    if pd.notna(speaker) and speaker != '':
                        speaker_to_cluster[speaker] = i
                
                df_valid['cluster_id'] = df_valid['speaker'].map(speaker_to_cluster).fillna(-1)
            else:
                # Last resort: single cluster
                df_valid['cluster_id'] = 0
        
        return df_valid
    
    def _refine_clusters_by_similarity(
        self, 
        df_valid: pd.DataFrame, 
        embeddings_scaled: np.ndarray, 
        initial_labels: np.ndarray,
        similarity_threshold: float = 0.70
    ) -> np.ndarray:
        """
        Refine clusters by merging those with centroids above similarity threshold.
        Mimics the centroid_merge_threshold logic from your pipeline.
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Calculate centroids for each cluster
            unique_labels = np.unique(initial_labels)
            cluster_centroids = {}
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                cluster_mask = initial_labels == label
                if np.sum(cluster_mask) > 0:
                    cluster_embeddings = embeddings_scaled[cluster_mask]
                    centroid = np.mean(cluster_embeddings, axis=0)
                    # Normalize centroid
                    norm = np.linalg.norm(centroid)
                    if norm > 1e-9:
                        cluster_centroids[label] = centroid / norm
            
            if len(cluster_centroids) < 2:
                return initial_labels
            
            # Calculate similarities between centroids
            labels_list = list(cluster_centroids.keys())
            centroids_matrix = np.vstack([cluster_centroids[label] for label in labels_list])
            similarity_matrix = cosine_similarity(centroids_matrix)
            
            # Find clusters to merge (centroid_merge_threshold = 0.60 in your config)
            centroid_merge_threshold = 0.60  # From your config
            merge_map = {}
            
            for i in range(len(labels_list)):
                for j in range(i + 1, len(labels_list)):
                    if similarity_matrix[i, j] >= centroid_merge_threshold:
                        label_i, label_j = labels_list[i], labels_list[j]
                        # Merge into the smaller label ID
                        target_label = min(label_i, label_j)
                        source_label = max(label_i, label_j)
                        merge_map[source_label] = target_label
            
            # Apply merges
            refined_labels = initial_labels.copy()
            for source, target in merge_map.items():
                refined_labels[refined_labels == source] = target
            
            if merge_map:
                logger.info(f"🔄 Merged {len(merge_map)} cluster pairs based on centroid similarity >= {centroid_merge_threshold}")
            
            return refined_labels
            
        except Exception as e:
            logger.warning(f"⚠️ Cluster refinement failed: {e}")
            return initial_labels

    
    def create_reference_style_visualizations(
        self,
        df_faces: pd.DataFrame,
        expected_embedding_dim: int = 512,
        reducer_type: str = 'tsne',
        save_plots: bool = True
    ) -> Dict[str, str]:
        """
        Create visualizations following the reference implementation style.
        
        Args:
            df_faces: DataFrame with face data and embeddings
            expected_embedding_dim: Expected embedding dimension
            reducer_type: Type of dimensionality reduction ('tsne' or 'pca')
            save_plots: Whether to save plots
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        from sklearn.manifold import TSNE
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import cv2
        
        logger.info(f"\n--- Creating Reference Style Visualizations ({reducer_type.upper()}) ---")
        
        # Input Validation
        req_cols = ['face_id', 'embedding', 'image_path'] if 'face_id' in df_faces.columns else ['embedding', 'image_path']
        if df_faces.empty or not all(col in df_faces.columns for col in req_cols):
            logger.warning(f"⚠️ DataFrame empty or missing columns ({', '.join(req_cols)}). Cannot visualize.")
            return {}

        # Ensure embeddings are valid numpy arrays and correct dimension
        valid_mask = df_faces['embedding'].apply(
            lambda x: isinstance(x, np.ndarray) and x.ndim == 1 and x.shape[0] == expected_embedding_dim
        )
        if not valid_mask.any():
            logger.warning("⚠️ No valid embeddings found.")
            return {}

        df_valid = df_faces[valid_mask].copy()
        initial_len = len(df_faces)
        if len(df_valid) < initial_len:
            logger.info(f"⚠️ Excluded {initial_len - len(df_valid)} invalid embeddings rows.")

        # Use face_id if available, otherwise create cluster_id
        cluster_col = 'face_id' if 'face_id' in df_valid.columns else 'cluster_id'
        if cluster_col not in df_valid.columns:
            df_valid['cluster_id'] = 0  # Single cluster
            cluster_col = 'cluster_id'

        df_clusters = df_valid[df_valid[cluster_col] >= 0].copy()
        if df_clusters.empty:
            logger.info("ℹ️ No non-noise clusters found.")
            return {}

        # Calculate Centroids
        cluster_data = {}  # {cluster_id: {'centroid': ndarray, 'members': [{'embedding': ndarray, 'path': str}]}}
        logger.info("   Calculating cluster centroids...")
        grouped = df_clusters.groupby(cluster_col)

        for cluster_id, group in grouped:
            member_embeddings = []
            member_info = []
            for _, row in group.iterrows():
                emb = row['embedding']
                norm = np.linalg.norm(emb)
                if norm > 1e-9:
                    normalized_emb = (emb / norm).astype(np.float32)
                    member_embeddings.append(normalized_emb)
                    member_info.append({'embedding': normalized_emb, 'path': row['image_path']})

            if not member_embeddings:
                continue
            member_matrix = np.vstack(member_embeddings)
            centroid = np.mean(member_matrix, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-9:
                continue
            normalized_centroid = (centroid / centroid_norm).astype(np.float32)
            cluster_data[cluster_id] = {'centroid': normalized_centroid, 'members': member_info}

        if not cluster_data:
            logger.error("❌ No valid cluster centroids calculated.")
            return {}

        cluster_ids = list(cluster_data.keys())
        centroid_list = [cluster_data[fid]['centroid'] for fid in cluster_ids]
        centroid_matrix = np.vstack(centroid_list)
        num_clusters = len(cluster_ids)
        logger.info(f"   Calculated {num_clusters} centroids.")

        # Dimensionality Reduction
        if num_clusters < 2:
            logger.warning("⚠️ Need >= 2 clusters for 2D viz.")
            return {}

        logger.info(f"   Performing {reducer_type.upper()} reduction...")
        centroids_2d = None
        if reducer_type.lower() == 'tsne':
            effective_perplexity = min(10, num_clusters - 1)
            if effective_perplexity < 5:
                logger.warning(f"⚠️ Too few clusters ({num_clusters}) for t-SNE perplexity {effective_perplexity}. Falling back to PCA.")
                reducer_type = 'pca'
            else:
                try:
                    tsne = TSNE(n_components=2, perplexity=effective_perplexity, random_state=42, init='pca', learning_rate='auto', n_iter=1000)
                    centroids_2d = tsne.fit_transform(centroid_matrix)
                except Exception as e:
                    logger.error(f"❌ t-SNE failed: {e}. Trying PCA...")
                    reducer_type = 'pca'

        if centroids_2d is None and reducer_type.lower() == 'pca':
            try:
                pca = PCA(n_components=2, random_state=42)
                centroids_2d = pca.fit_transform(centroid_matrix)
            except Exception as e:
                logger.error(f"❌ PCA failed: {e}.")
                return {}

        if centroids_2d is None:
            logger.error("❌ Dimensionality reduction failed.")
            return {}
        
        logger.info("   Dimensionality reduction complete.")

        # Find Closest Image
        logger.info("   Finding closest image to each centroid...")
        closest_images = {}
        for i, cluster_id in enumerate(cluster_ids):
            centroid = cluster_data[cluster_id]['centroid']
            members_info = cluster_data[cluster_id]['members']
            if not members_info:
                continue
            member_embeddings_matrix = np.vstack([m['embedding'] for m in members_info])
            similarities = np.dot(member_embeddings_matrix, centroid)  # Dot product on normalized vectors
            if similarities.size > 0:
                closest_images[cluster_id] = members_info[np.argmax(similarities)]['path']

        # Plotting
        logger.info("   Generating plot...")
        fig, ax = plt.subplots(figsize=(18, 18))
        plotted_count = 0
        img_zoom = 0.15
        
        for i, cluster_id in enumerate(cluster_ids):
            x_coord, y_coord = centroids_2d[i, 0], centroids_2d[i, 1]
            img_path = closest_images.get(cluster_id)

            if img_path and os.path.exists(img_path):
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError("imread failed")
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imagebox = OffsetImage(img_rgb, zoom=img_zoom)
                    # Remove problematic line - imagebox.image.axes = ax
                    ab = AnnotationBbox(imagebox, (x_coord, y_coord), frameon=True, pad=0.1)
                    ax.add_artist(ab)
                    ax.text(x_coord, y_coord + 50, f"ID: {cluster_id}", ha='center', va='bottom', 
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5))
                    plotted_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error plotting image for cluster {cluster_id} (Path: {img_path}): {e}")
                    ax.scatter([x_coord], [y_coord], marker='x', color='red', s=50)
                    ax.text(x_coord, y_coord, f"ID: {cluster_id}\n(Img Err)", fontsize=8, color='red')
            else:
                ax.scatter([x_coord], [y_coord], marker='o', facecolors='none', edgecolors='blue', s=50)
                ax.text(x_coord, y_coord, f"ID: {cluster_id}\n(No Img)", fontsize=8, color='blue')

        ax.set_title(f'2D Visualization of Cluster Centroids ({reducer_type.upper()}) - {plotted_count}/{num_clusters} Plotted')
        ax.set_xlabel(f"{reducer_type.upper()} Component 1")
        ax.set_ylabel(f"{reducer_type.upper()} Component 2")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        plot_paths = {}
        # Save or Show
        if save_plots:
            viz_dir = self.path_handler.get_cluster_visualization_dir()
            os.makedirs(viz_dir, exist_ok=True)
            output_filename = f"cluster_centroids_2d_{reducer_type}.png"
            fig_path = os.path.join(viz_dir, output_filename)
            try:
                plt.savefig(fig_path, bbox_inches='tight', dpi=150)
                logger.info(f"✅ Reference style visualization saved to: {fig_path}")
                plot_paths['reference_centroids'] = fig_path
            except Exception as e:
                logger.error(f"❌ Error saving reference figure {fig_path}: {e}")
            plt.close(fig)
        else:
            plt.show()

        return plot_paths
    
    def _create_matplotlib_plots(self, df_valid: pd.DataFrame, viz_dir: str, pca: PCA) -> Dict[str, str]:
        """Create matplotlib visualizations using reference style with face images on plots."""
        plot_paths = {}
        
        try:
            # Use the reference style visualization for 2D centroids
            plot_path = self.visualize_cluster_centroids_2d(
                df_clustered=df_valid,
                expected_embedding_dim=512,
                reducer_type='pca',
                save_figure=True,
                output_filename=f"{self.path_handler.get_episode_code()}_face_clusters_2d_pca.png"
            )
            
            if plot_path:
                plot_paths['matplotlib_2d_pca'] = plot_path
                logger.info(f"✅ Created reference-style 2D PCA visualization: {plot_path}")
            
            # Also create cluster grids
            self.show_all_clusters(
                df_clustered=df_valid,
                max_images_per_cluster=60,
                cols=8,
                save_figures=True
            )
            
            # Add grid visualization paths
            cluster_viz_dir = self.path_handler.get_cluster_visualization_dir()
            if os.path.exists(cluster_viz_dir):
                grid_files = [f for f in os.listdir(cluster_viz_dir) if f.startswith('Cluster_') and f.endswith('.png')]
                if grid_files:
                    plot_paths['cluster_grids'] = f"{len(grid_files)} cluster grid files in {cluster_viz_dir}"
                    
        except Exception as e:
            logger.error(f"❌ Error creating matplotlib plots: {e}")
            # Fallback: Create simple scatter plot if face image visualization fails
            try:
                logger.info("🔄 Attempting fallback scatter plot visualization...")
                plot_paths.update(self._create_fallback_scatter_plot(df_valid, viz_dir, pca))
            except Exception as fallback_error:
                logger.error(f"❌ Fallback visualization also failed: {fallback_error}")
        
        return plot_paths
    
    def _create_fallback_scatter_plot(self, df_valid: pd.DataFrame, viz_dir: str, pca: PCA) -> Dict[str, str]:
        """Create a simple scatter plot as fallback when face image visualization fails."""
        plot_paths = {}
        
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 9))
            
            # Safe variance ratio formatting
            pc1_var = f"{pca.explained_variance_ratio_[0]:.1%}" if hasattr(pca, 'explained_variance_ratio_') and len(pca.explained_variance_ratio_) > 0 else "Unknown"
            pc2_var = f"{pca.explained_variance_ratio_[1]:.1%}" if hasattr(pca, 'explained_variance_ratio_') and len(pca.explained_variance_ratio_) > 1 else "Unknown"
            
            ax.set_title(f'2D PCA Face Clusters (Scatter) - {self.path_handler.get_episode_code()}', fontsize=14)
            ax.set_xlabel(f'PC1 ({pc1_var} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({pc2_var} variance)', fontsize=12)
            
            # Create scatter plot colored by cluster
            cluster_col = 'cluster_id' if 'cluster_id' in df_valid.columns else 'face_id'
            if cluster_col in df_valid.columns:
                scatter = ax.scatter(
                    df_valid['pca_x'], 
                    df_valid['pca_y'], 
                    c=df_valid[cluster_col], 
                    cmap='tab10', 
                    alpha=0.7,
                    s=60
                )
                plt.colorbar(scatter, ax=ax, label='Cluster ID')
            else:
                ax.scatter(df_valid['pca_x'], df_valid['pca_y'], alpha=0.7, s=60)
            
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            # Save fallback plot
            plot_path = os.path.join(viz_dir, f"{self.path_handler.get_episode_code()}_face_clusters_scatter.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            plot_paths['matplotlib_scatter_fallback'] = plot_path
            logger.info(f"✅ Created fallback scatter plot: {plot_path}")
            
        except Exception as e:
            logger.error(f"❌ Error creating fallback scatter plot: {e}")
            plt.close('all')
        
        return plot_paths
    
    def _create_plotly_plots(self, df_valid: pd.DataFrame, viz_dir: str, pca: PCA) -> Dict[str, str]:
        """Plotly plots disabled - only using matplotlib visualizations."""
        logger.info("📊 Plotly visualizations disabled - using matplotlib only")
        return {}
    
    def _save_cluster_summary(self, df_valid: pd.DataFrame, viz_dir: str) -> None:
        """Save cluster summary statistics."""
        try:
            # Create cluster summary
            cluster_summary = []
            
            # Use either cluster_id or face_id column
            cluster_col = 'cluster_id' if 'cluster_id' in df_valid.columns else 'face_id'
            
            if cluster_col not in df_valid.columns:
                logger.warning("⚠️ No cluster column found for summary")
                return
            
            for cluster_id in df_valid[cluster_col].unique():
                if cluster_id == -1:  # Skip noise points
                    continue
                    
                cluster_faces = df_valid[df_valid[cluster_col] == cluster_id]
                
                # Get most common speaker for this cluster (if speaker column exists)
                main_speaker = 'Unknown'
                if 'speaker' in cluster_faces.columns:
                    speaker_counts = cluster_faces['speaker'].value_counts()
                    main_speaker = speaker_counts.index[0] if len(speaker_counts) > 0 else 'Unknown'
                
                # Safe column access with defaults
                avg_confidence = cluster_faces['speaker_confidence'].mean() if 'speaker_confidence' in cluster_faces.columns else 0.0
                avg_blur_score = cluster_faces['blur_score'].mean() if 'blur_score' in cluster_faces.columns else 0.0
                time_span_start = cluster_faces['timestamp_seconds'].min() if 'timestamp_seconds' in cluster_faces.columns else 0.0
                time_span_end = cluster_faces['timestamp_seconds'].max() if 'timestamp_seconds' in cluster_faces.columns else 0.0
                dialogue_lines = list(cluster_faces['dialogue_index'].unique()) if 'dialogue_index' in cluster_faces.columns else []
                
                summary = {
                    'cluster_id': cluster_id,
                    'face_count': len(cluster_faces),
                    'main_speaker': main_speaker,
                    'avg_confidence': float(avg_confidence) if pd.notna(avg_confidence) else 0.0,
                    'avg_blur_score': float(avg_blur_score) if pd.notna(avg_blur_score) else 0.0,
                    'time_span_start': float(time_span_start) if pd.notna(time_span_start) else 0.0,
                    'time_span_end': float(time_span_end) if pd.notna(time_span_end) else 0.0,
                    'dialogue_lines': dialogue_lines
                }
                
                cluster_summary.append(summary)
            
            # Save summary
            if cluster_summary:
                summary_df = pd.DataFrame(cluster_summary)
                summary_path = os.path.join(viz_dir, f"{self.path_handler.get_episode_code()}_cluster_summary.csv")
                summary_df.to_csv(summary_path, index=False)
                
                logger.info(f"💾 Saved cluster summary to: {summary_path}")
            else:
                logger.info("ℹ️ No clusters found for summary")
            
        except Exception as e:
            logger.error(f"❌ Error saving cluster summary: {e}")
    
    def create_cluster_visualizations_with_chromadb(
        self,
        df_faces: pd.DataFrame,
        embedding_model_name: str = "Facenet512",
        cosine_similarity_threshold: float = 0.65,
        min_cluster_size: int = 5,
        merge_threshold: float = 0.85,
        output_format: str = "both",
        save_plots: bool = True
    ) -> Dict[str, str]:
        """
        Create visualizations using ChromaDB clustering following the reference implementation.
        
        Args:
            df_faces: DataFrame with face data and embeddings
            embedding_model_name: Name of the embedding model
            cosine_similarity_threshold: Threshold for ChromaDB clustering
            min_cluster_size: Minimum cluster size
            merge_threshold: Threshold for cluster merging
            output_format: Format for visualizations
            save_plots: Whether to save plots
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        if df_faces.empty or 'embedding' not in df_faces.columns:
            logger.warning("⚠️ No face embeddings available for cluster visualization")
            return {}
        
        logger.info(f"📊 Creating cluster visualizations for {len(df_faces)} faces")
        
        try:
            # Step 1: Use existing clusters or perform clustering if needed
            logger.info("🔗 Step 1: Using/generating face clusters")
            
            if 'cluster_id' not in df_faces.columns and 'face_id' not in df_faces.columns:
                # If no clusters exist, run the face clustering system
                logger.info("🎭 No existing clusters found, running face clustering system")
                temp_dialogue = []  # Empty dialogue list for clustering only
                _, cluster_info, character_clusters = self.face_clustering_system.run_face_clustering_pipeline(
                    df_faces=df_faces,
                    dialogue_lines=temp_dialogue,
                    cosine_similarity_threshold=cosine_similarity_threshold,
                    min_cluster_size=min_cluster_size,
                    merge_threshold=merge_threshold,
                    expected_embedding_dim=512
                )
                df_clustered = df_faces.copy()
            else:
                # Use existing clusters
                logger.info("✅ Using existing cluster assignments")
                df_clustered = df_faces.copy()
            
            # Map face_id or cluster_id to face_id for compatibility with visualization code
            logger.info(f"🔍 Visualizer debugging - df_clustered shape: {df_clustered.shape}")
            logger.info(f"🔍 Visualizer debugging - df_clustered columns: {df_clustered.columns.tolist()}")
            logger.info(f"🔍 Visualizer debugging - df_clustered empty: {df_clustered.empty}")
            
            if 'face_id' in df_clustered.columns:
                logger.info("✅ Found 'face_id' column")
                face_id_dist = df_clustered['face_id'].value_counts()
                logger.info(f"🔍 Face ID distribution: {face_id_dist.to_dict()}")
                df_clustered['face_id'] = df_clustered['face_id']  # Already has face_id
            elif 'cluster_id' in df_clustered.columns:
                logger.info("✅ Found 'cluster_id' column, mapping to face_id")
                cluster_id_dist = df_clustered['cluster_id'].value_counts()
                logger.info(f"🔍 Cluster ID distribution: {cluster_id_dist.to_dict()}")
                df_clustered['face_id'] = df_clustered['cluster_id']  # Map cluster_id to face_id
            else:
                logger.error("❌ Neither 'face_id' nor 'cluster_id' columns found")
                logger.error(f"❌ Available columns: {df_clustered.columns.tolist()}")
            
            if df_clustered.empty or 'face_id' not in df_clustered.columns:
                logger.error("❌ Face clustering failed")
                return {}
            
            # Check if we have valid clusters (not all noise)
            valid_clusters = df_clustered[df_clustered['face_id'] >= 0]
            if valid_clusters.empty:
                logger.warning("⚠️ No valid clusters found (all faces are noise). Skipping visualization.")
                return {}
            
            logger.info(f"📊 Found {len(valid_clusters)} faces in {len(valid_clusters['face_id'].unique())} valid clusters")

            # Step 2: Save embeddings to vector store (if needed)
            logger.info("💾 Step 2: Ensuring embeddings are in vector store")
            self.face_vector_store.save_embeddings_to_vector_store(
                df_faces=df_clustered,
                embedding_model_name=embedding_model_name
            )
            
            # Step 3: Create visualizations
            logger.info("📈 Step 3: Creating visualizations")
            return self.create_cluster_visualizations(
                df_faces=df_clustered,
                output_format="matplotlib",  # Only use matplotlib with face images
                save_plots=save_plots
            )
            
        except Exception as e:
            logger.error(f"❌ Error in ChromaDB visualization pipeline: {e}")
            return {}
    
    def get_chromadb_statistics(self, embedding_model_name: str = "Facenet512") -> Dict:
        """Get statistics from ChromaDB face collection."""
        try:
            return self.face_vector_store.get_collection_stats(embedding_model_name)
        except Exception as e:
            logger.error(f"❌ Error getting ChromaDB statistics: {e}")
            return {'error': str(e)}
    
    def reset_chromadb_collection(self, embedding_model_name: str = "Facenet512") -> bool:
        """Reset the ChromaDB face collection."""
        try:
            return self.face_vector_store.reset_collection(embedding_model_name)
        except Exception as e:
            logger.error(f"❌ Error resetting ChromaDB collection: {e}")
            return False
    
    def show_all_clusters(
        self, 
        df_clustered: pd.DataFrame, 
        max_images_per_cluster: int = 60, 
        cols: int = 8,
        save_figures: bool = True
    ) -> None:
        """
        Shows or saves grid images for faces belonging to each cluster ID.
        Includes metadata like Seg/Line, Confidence, Gender/Age, and Blur Metrics in the title.
        """
        output_dir = self.path_handler.get_cluster_visualization_dir()
        if save_figures: 
            os.makedirs(output_dir, exist_ok=True)

        if 'face_id' not in df_clustered.columns: 
            logger.warning("⚠️ 'face_id' column missing.")
            return
        if 'image_path' not in df_clustered.columns: 
            logger.warning("⚠️ 'image_path' column missing.")
            return

        # Ensure necessary metric columns exist, even if null, to avoid errors
        for col in ['blur_score', 'speaker', 'detection_confidence', 'dialogue_index', 'face_index']:
             if col not in df_clustered.columns:
                  logger.info(f"ℹ️ Adding missing column '{col}' for visualization title.")
                  df_clustered[col] = None # Add column if missing

        face_ids = sorted(df_clustered["face_id"].unique())
        if not face_ids: 
            logger.info("ℹ️ No face IDs found to visualize.")
            return

        # Filter out clusters smaller than minimum size (respect config.ini setting)
        min_cluster_size = config.min_cluster_size_final
        valid_face_ids = []
        for face_id in face_ids:
            if face_id == -1:  # Always include noise cluster
                valid_face_ids.append(face_id)
            else:
                cluster_size = len(df_clustered[df_clustered["face_id"] == face_id])
                if cluster_size >= min_cluster_size:
                    valid_face_ids.append(face_id)
                else:
                    logger.debug(f"🚫 Skipping visualization for cluster {face_id}: size ({cluster_size}) < minimum ({min_cluster_size})")
        
        if not valid_face_ids:
            logger.info("ℹ️ No valid clusters found for visualization after size filtering.")
            return

        logger.info(f"🖼️ Generating cluster grid visualizations for {len(valid_face_ids)} valid ID(s) (filtered from {len(face_ids)} total)...")

        for face_id in tqdm(valid_face_ids, desc="Visualizing Clusters"):
            cluster_label = f"Cluster_{face_id}" if face_id >= 0 else "Noise_-1"

            cluster_df = df_clustered[df_clustered["face_id"] == face_id].copy()
            n_images_total = len(cluster_df)
            if n_images_total == 0: continue

            # Sample or take head
            if n_images_total > max_images_per_cluster:
                cluster_df_display = cluster_df.sample(n=max_images_per_cluster, random_state=42)
                display_count = max_images_per_cluster
            else:
                cluster_df_display = cluster_df.head(max_images_per_cluster)
                display_count = n_images_total

            n_images = len(cluster_df_display)
            if n_images == 0: continue

            rows = (n_images + cols - 1) // cols
            fig_height_base = 2.0
            fig_height_per_row = 1.0
            try:
                fig, axs = plt.subplots(rows, cols, figsize=(2.5 * cols, fig_height_base + fig_height_per_row * rows), squeeze=False)
                axs = axs.flatten()
            except Exception as e:
                logger.error(f"❌ Error creating subplot figure for {cluster_label}: {e}")
                continue

            plot_idx = 0
            for _, row in cluster_df_display.iterrows():
                if plot_idx >= len(axs): break
                ax = axs[plot_idx]
                img_path = row.get("image_path")
                img = None
                if pd.notna(img_path) and os.path.exists(img_path): 
                    img = cv2.imread(img_path)

                if img is None:
                    ax.set_title(f"Not found:\n{os.path.basename(str(img_path))}", fontsize=7, color='red')
                    ax.text(0.5, 0.5, 'X', ha='center', va='center', fontsize=20, color='red')
                else:
                    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_display)

                    # Get image dimensions
                    height, width = img.shape[:2]
                    
                    # Title construction
                    title_parts = [f"D:{row.get('dialogue_index','?')}", f"F:{row.get('face_index','?')}"]
                    title_line1 = " ".join(title_parts)
                    title_line2 = f"Conf: {row.get('detection_confidence', 0):.2f}"

                    # Speaker line
                    title_line3 = ""
                    if pd.notna(row.get('speaker')):
                        speaker = str(row.get('speaker', ''))[:10]  # Truncate long names
                        title_line3 = f"Spk: {speaker}"

                    # Blur line
                    title_line4 = ""
                    blur_score = row.get('blur_score', None)
                    blur_str = f"Blur:{blur_score:.1f}" if pd.notna(blur_score) else "Blur:?"
                    title_line4 = blur_str
                    
                    # Dimensions line
                    title_line5 = f"Dim: {width}x{height}"

                    # Combine title lines
                    final_title = f"{title_line1}\n{title_line2}\n{title_line3}\n{title_line4}\n{title_line5}".strip()
                    ax.set_title(final_title, fontsize=7)

                ax.axis('off')
                plot_idx += 1

            # Turn off remaining axes
            for ax_idx in range(plot_idx, len(axs)): 
                axs[ax_idx].axis('off')

            fig_title = f"{cluster_label} ({display_count} of {n_images_total} samples shown)"
            plt.suptitle(fig_title, fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            if save_figures:
                import re
                safe_label = re.sub(r'[^\w\-]+', '_', cluster_label)
                fig_path = os.path.join(output_dir, f"{safe_label}.png")
                try:
                    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
                except Exception as e:
                    logger.error(f"❌ Error saving figure {fig_path}: {e}")
            else:
                plt.show()
            plt.close(fig)

        logger.info(f"✅ Cluster grid visualizations potentially saved to: {output_dir}")

    def visualize_cluster_centroids_2d(
        self,
        df_clustered: pd.DataFrame,
        expected_embedding_dim: int = 512,
        reducer_type: str = 'tsne',
        tsne_perplexity: int = 10,
        save_figure: bool = True,
        img_zoom: float = 0.15,
        output_filename: str = "cluster_centroids_2d.png"
    ) -> str:
        """
        Visualizes cluster centroids in 2D using dimensionality reduction.
        Plots the face image nearest to each centroid.
        """
        print(f"\n--- Visualizing Cluster Centroids in 2D ({reducer_type.upper()}) ---")

        # --- Input Validation ---
        req_cols = ['face_id', 'embedding', 'image_path']
        if df_clustered.empty or not all(col in df_clustered.columns for col in req_cols):
            print(f"⚠️ DataFrame empty or missing columns ({', '.join(req_cols)}). Cannot visualize centroids.")
            return ""

        # Ensure embeddings are valid numpy arrays and correct dimension
        valid_mask = df_clustered['embedding'].apply(lambda x: isinstance(x, np.ndarray) and x.ndim == 1 and x.shape[0] == expected_embedding_dim)
        if not valid_mask.any(): 
            print("⚠️ No valid embeddings found.")
            return ""

        df_valid = df_clustered[valid_mask].copy()
        initial_len = len(df_clustered)
        if len(df_valid) < initial_len: 
            print(f"⚠️ Excluded {initial_len - len(df_valid)} invalid embeddings rows.")

        df_clusters = df_valid[df_valid['face_id'] >= 0].copy()
        if df_clusters.empty: 
            print("ℹ️ No non-noise clusters found.")
            return ""

        # --- Calculate Centroids ---
        cluster_data = {} # {face_id: {'centroid': ndarray, 'members': [{'embedding': ndarray, 'path': str}]}}
        print("   Calculating cluster centroids...")
        grouped = df_clusters.groupby('face_id')

        # Use tqdm here for progress
        for face_id, group in tqdm(grouped, desc="Processing Clusters", total=len(grouped)):
            member_embeddings = []
            member_info = []
            for _, row in group.iterrows():
                emb = row['embedding']
                norm = np.linalg.norm(emb)
                if norm > 1e-9:
                    normalized_emb = (emb / norm).astype(np.float32)
                    member_embeddings.append(normalized_emb)
                    member_info.append({'embedding': normalized_emb, 'path': row['image_path']})

            if not member_embeddings: continue
            member_matrix = np.vstack(member_embeddings)
            centroid = np.mean(member_matrix, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-9: continue
            normalized_centroid = (centroid / centroid_norm).astype(np.float32)
            cluster_data[face_id] = {'centroid': normalized_centroid, 'members': member_info}

        if not cluster_data: 
            print("❌ No valid cluster centroids calculated.")
            return ""

        cluster_ids = list(cluster_data.keys())
        centroid_list = [cluster_data[fid]['centroid'] for fid in cluster_ids]
        centroid_matrix = np.vstack(centroid_list)
        num_clusters = len(cluster_ids)
        print(f"   Calculated {num_clusters} centroids.")

        # --- Dimensionality Reduction ---
        if num_clusters < 2: 
            print("⚠️ Need >= 2 clusters for 2D viz.")
            return ""

        print(f"   Performing {reducer_type.upper()} reduction...")
        centroids_2d = None
        if reducer_type.lower() == 'tsne':
            effective_perplexity = min(tsne_perplexity, num_clusters - 1)
            if effective_perplexity < 5:
                 print(f"⚠️ Too few clusters ({num_clusters}) for t-SNE perplexity {effective_perplexity}. Falling back to PCA.")
                 reducer_type = 'pca'
            else:
                try:
                    tsne = TSNE(n_components=2, perplexity=effective_perplexity, random_state=42, init='pca', learning_rate='auto', n_iter=1000) # Use n_iter instead of max_iter
                    centroids_2d = tsne.fit_transform(centroid_matrix)
                except Exception as e: 
                    print(f"❌ t-SNE failed: {e}. Trying PCA...")
                    reducer_type = 'pca'

        if centroids_2d is None and reducer_type.lower() == 'pca':
             try: 
                 pca = PCA(n_components=2, random_state=42)
                 centroids_2d = pca.fit_transform(centroid_matrix)
             except Exception as e: 
                 print(f"❌ PCA failed: {e}.")
                 return ""

        if centroids_2d is None: 
            print("❌ Dimensionality reduction failed.")
            return ""
        print("   Dimensionality reduction complete.")

        # --- Find Closest Image ---
        print("   Finding closest image to each centroid...")
        closest_images = {}
        for i, face_id in enumerate(cluster_ids):
            centroid = cluster_data[face_id]['centroid']
            members_info = cluster_data[face_id]['members']
            if not members_info: continue
            member_embeddings_matrix = np.vstack([m['embedding'] for m in members_info])
            similarities = np.dot(member_embeddings_matrix, centroid) # Dot product on normalized vectors
            if similarities.size > 0: 
                closest_images[face_id] = members_info[np.argmax(similarities)]['path']

        # --- Plotting ---
        print("   Generating plot...")
        fig, ax = plt.subplots(figsize=(18, 18))
        plotted_count = 0
        for i, face_id in enumerate(cluster_ids):
            x_coord, y_coord = centroids_2d[i, 0], centroids_2d[i, 1]
            img_path = closest_images.get(face_id)

            if img_path and os.path.exists(img_path):
                try:
                    img = cv2.imread(img_path)
                    if img is None: raise ValueError("imread failed")
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imagebox = OffsetImage(img_rgb, zoom=img_zoom)
                    # Remove problematic line - imagebox.image.axes = ax
                    ab = AnnotationBbox(imagebox, (x_coord, y_coord), frameon=True, pad=0.1) # Removed borderpad
                    ax.add_artist(ab)
                    # Use fixed offset instead of get_extent which may not work
                    ax.text(x_coord, y_coord + 25, f"ID: {face_id}", ha='center', va='bottom', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5))
                    plotted_count += 1
                except Exception as e:
                    print(f"⚠️ Error plotting image for cluster {face_id} (Path: {img_path}): {e}")
                    ax.scatter([x_coord], [y_coord], marker='x', color='red', s=50)
                    ax.text(x_coord, y_coord, f"ID: {face_id}\n(Img Err)", fontsize=8, color='red')
            else:
                ax.scatter([x_coord], [y_coord], marker='o', facecolors='none', edgecolors='blue', s=50)
                ax.text(x_coord, y_coord, f"ID: {face_id}\n(No Img)", fontsize=8, color='blue')

        ax.set_title(f'2D Visualization of Cluster Centroids ({reducer_type.upper()}) - {plotted_count}/{num_clusters} Plotted')
        ax.set_xlabel(f"{reducer_type.upper()} Component 1")
        ax.set_ylabel(f"{reducer_type.upper()} Component 2")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # --- Save or Show ---
        plot_path = ""
        if save_figure:
            output_dir = self.path_handler.get_cluster_visualization_dir()
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, output_filename)
            try: 
                plt.savefig(plot_path, bbox_inches='tight', dpi=150)
                print(f"✅ Centroid visualization saved to: {plot_path}")
            except Exception as e: 
                print(f"❌ Error saving centroid figure {plot_path}: {e}")
            plt.close(fig)
        else: 
            plt.show()
        
        return plot_path

    def get_visualization_urls(self) -> Dict[str, str]:
        """Get URLs/paths to existing visualizations."""
        viz_dir = self.path_handler.get_cluster_visualization_dir()
        
        if not os.path.exists(viz_dir):
            return {}
        
        viz_files = {}
        episode_code = self.path_handler.get_episode_code()
        
        # Check for existing visualization files
        potential_files = {
            'matplotlib_overview': f"{episode_code}_face_clusters_matplotlib.png",
            'plotly_interactive': f"{episode_code}_face_clusters_interactive.html",
            'plotly_speakers': f"{episode_code}_speakers_interactive.html",
            'cluster_summary': f"{episode_code}_cluster_summary.csv",
            'cluster_centroids_2d': 'cluster_centroids_2d.png'
        }
        
        for viz_type, filename in potential_files.items():
            file_path = os.path.join(viz_dir, filename)
            if os.path.exists(file_path):
                viz_files[viz_type] = file_path
        
        return viz_files
