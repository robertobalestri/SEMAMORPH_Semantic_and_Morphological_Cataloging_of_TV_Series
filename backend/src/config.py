"""
Configuration loader for SEMAMORPH project.

This module provides functionality to load and access configuration settings
from the config.ini file.
"""

import configparser
import os
from pathlib import Path
from typing import Any, Optional


class Config:
    """Configuration manager for SEMAMORPH project."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration from file.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config = configparser.ConfigParser()
        
        # Find config file in project root
        if config_file is None:
            # Try different locations
            possible_paths = [
                "config.ini",                    # Current directory
                "../config.ini",                 # Parent directory (when running from backend/)
                "../../config.ini",              # Two levels up
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "config.ini")  # Project root
            ]
            
            found_config = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_config = path
                    break
            
            # Use found config or default fallback
            config_file = found_config if found_config else "config.ini"
        
        self.config_file = config_file
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        
        self.config.read(self.config_file)
    
    def get_bool(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get boolean value from config."""
        return self.config.getboolean(section, key, fallback=fallback)
    
    def get_str(self, section: str, key: str, fallback: str = "") -> str:
        """Get string value from config."""
        return self.config.get(section, key, fallback=fallback)
    
    def get_int(self, section: str, key: str, fallback: int = 0) -> int:
        """Get integer value from config."""
        return self.config.getint(section, key, fallback=fallback)
    
    def get_float(self, section: str, key: str, fallback: float = 0.0) -> float:
        """Get float value from config."""
        return self.config.getfloat(section, key, fallback=fallback)
    
    def set_value(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value."""
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, str(value))
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    
    # Convenience properties for commonly used settings
    @property
    def pronoun_replacement_batch_size(self) -> int:
        """Batch size for pronoun replacement processing."""
        return self.get_int('processing', 'pronoun_replacement_batch_size')
    
    @property
    def pronoun_replacement_context_size(self) -> int:
        """Context size for pronoun replacement processing."""
        return self.get_int('processing', 'pronoun_replacement_context_size')
    
    @property
    def text_simplification_batch_size(self) -> int:
        """Batch size for text simplification processing."""
        return self.get_int('processing', 'text_simplification_batch_size')
    
    @property
    def semantic_segmentation_window_size(self) -> int:
        """Window size for semantic segmentation processing."""
        return self.get_int('processing', 'semantic_segmentation_window_size')
    
    @property
    def semantic_correction_batch_size(self) -> int:
        """Batch size for semantic segment correction."""
        return self.get_int('processing', 'semantic_correction_batch_size')
    
    @property
    def data_dir(self) -> str:
        """Data directory path."""
        return self.get_str('paths', 'data_dir')
    
    @property
    def narrative_storage_dir(self) -> str:
        """Narrative storage directory path."""
        return self.get_str('paths', 'narrative_storage_dir')
    
    @property
    def api_host(self) -> str:
        """API server host."""
        return self.get_str('api', 'host')
    
    @property
    def api_port(self) -> int:
        """API server port."""
        return self.get_int('api', 'port')
    
    @property
    def log_level(self) -> str:
        """Logging level."""
        return self.get_str('logging', 'level')
    
    @property
    def log_to_file(self) -> bool:
        """Whether to log to file."""
        return self.get_bool('logging', 'log_to_file')
    
    @property
    def log_file(self) -> str:
        """Log file path."""
        return self.get_str('logging', 'log_file')
    
    # Face processing configuration properties
    @property
    def face_detector(self) -> str:
        """Face detection model."""
        return self.get_str('face_processing', 'detector')
    
    @property
    def face_min_confidence(self) -> float:
        """Minimum face detection confidence."""
        return self.get_float('face_processing', 'min_confidence')
    
    @property
    def face_min_area_ratio(self) -> float:
        """Minimum face area ratio."""
        return self.get_float('face_processing', 'min_face_area_ratio')
    
    @property
    def face_blur_threshold(self) -> float:
        """Face blur threshold for quality filtering (legacy/laplacian method)."""
        return self.get_float('face_processing', 'blur_threshold')
    
    @property
    def blur_detection_method(self) -> str:
        """Method for blur detection: 'laplacian', 'gradient', 'tenengrad', 'composite'."""
        return self.get_str('face_processing', 'blur_detection_method', fallback='gradient')
    
    @property
    def blur_threshold_gradient(self) -> float:
        """Blur threshold for gradient magnitude method."""
        return self.get_float('face_processing', 'blur_threshold_gradient', fallback=15.0)
    
    @property
    def blur_threshold_tenengrad(self) -> float:
        """Blur threshold for Tenengrad method."""
        return self.get_float('face_processing', 'blur_threshold_tenengrad', fallback=500.0)
    
    @property
    def blur_threshold_composite(self) -> float:
        """Blur threshold for composite method."""
        return self.get_float('face_processing', 'blur_threshold_composite', fallback=25.0)
    
    @property
    def face_embedding_model(self) -> str:
        """Face embedding model."""
        return self.get_str('face_processing', 'embedding_model')
    
    @property
    def face_embedding_dimension(self) -> int:
        """Face embedding dimension."""
        return self.get_int('face_processing', 'embedding_dimension', fallback=512)
    
    @property
    def face_enable_eye_validation(self) -> bool:
        """Enable eye landmark validation."""
        return self.get_bool('face_processing', 'enable_eye_validation')
    
    @property
    def face_eye_alignment_threshold(self) -> float:
        """Eye alignment threshold for validation."""
        return self.get_float('face_processing', 'eye_alignment_threshold')
    
    @property
    def face_eye_distance_threshold(self) -> float:
        """Eye distance threshold for validation."""
        return self.get_float('face_processing', 'eye_distance_threshold')
    
    @property
    def speaker_face_similarity_threshold(self) -> float:
        """Face similarity threshold for speaker identification."""
        return self.get_float('speaker_identification', 'face_similarity_threshold')

    @property
    def similarity_to_character_median_threshold_for_speaker_assignment(self) -> float:
        """Similarity threshold for speaker assignment from character median comparison (0-1)."""
        return self.get_float('speaker_identification', 'similarity_to_character_median_threshold_for_speaker_assignment', fallback=0.75)

    # Clustering configuration properties
    @property
    def cosine_similarity_threshold(self) -> float:
        """Cosine similarity threshold for face clustering (0 to 1, higher is stricter)."""
        return self.get_float('clustering', 'cosine_similarity_threshold')
    
    @property
    def cross_episode_similarity_threshold(self) -> float:
        """Similarity threshold for matching clusters across episodes (0 to 1, higher is stricter)."""
        return self.get_float('clustering', 'cross_episode_similarity_threshold', fallback=0.85)
    
    @property
    def min_cluster_size_final(self) -> int:
        """Minimum faces required to form a persistent cluster ID."""
        return self.get_int('clustering', 'min_cluster_size_final')
    
    @property
    def centroid_merge_threshold(self) -> float:
        """Threshold for merging similar cluster centroids (0 to 1)."""
        return self.get_float('clustering', 'centroid_merge_threshold')

    # Multi-face processing configuration properties
    @property
    def enable_multiface_processing(self) -> bool:
        """Enable multi-face processing for speaker identification."""
        return self.get_bool('multiface_processing', 'enable_multiface_processing', fallback=True)
    
    @property
    def multiface_max_faces_per_dialogue(self) -> int:
        """Maximum number of faces to consider per dialogue line."""
        return self.get_int('multiface_processing', 'max_faces_per_dialogue', fallback=3)
    
    @property
    def multiface_equal_probability_distribution(self) -> bool:
        """Use equal probability distribution among faces (vs. confidence weighting)."""
        return self.get_bool('multiface_processing', 'equal_probability_distribution', fallback=True)
    
    @property
    def multiface_llm_disambiguation_threshold(self) -> float:
        """Minimum similarity threshold to trigger LLM disambiguation."""
        return self.get_float('multiface_processing', 'llm_disambiguation_threshold', fallback=0.8)
    
    @property
    def cluster_minimum_occurrences(self) -> float:
        """Minimum occurrences required for a character to be assigned to a cluster."""
        return self.get_float('multiface_processing', 'cluster_minimum_occurrences', fallback=2.0)

    # Cluster assignment enhancement properties
    @property
    def enable_parity_detection(self) -> bool:
        """Enable detection of clusters with tied character assignments."""
        return self.get_bool('cluster_assignment', 'enable_parity_detection', fallback=True)

    @property
    def enable_spatial_outlier_removal(self) -> bool:
        """Enable removal of clusters that are spatially far from character's main group."""
        return self.get_bool('cluster_assignment', 'enable_spatial_outlier_removal', fallback=True)

    @property
    def spatial_outlier_threshold(self) -> float:
        """Cosine distance threshold for spatial outlier detection."""
        return self.get_float('cluster_assignment', 'spatial_outlier_threshold', fallback=0.35)

    @property
    def min_clusters_for_outlier_detection(self) -> int:
        """Minimum number of clusters needed for a character to detect outliers."""
        return self.get_int('cluster_assignment', 'min_clusters_for_outlier_detection', fallback=3)

    @property
    def enable_ambiguous_resolution(self) -> bool:
        """Enable resolution of ambiguous clusters using character median similarity."""
        return self.get_bool('cluster_assignment', 'enable_ambiguous_resolution', fallback=True)

    @property
    def ambiguous_resolution_threshold(self) -> float:
        """Cosine similarity threshold for resolving ambiguous clusters to character medians."""
        return self.get_float('cluster_assignment', 'ambiguous_resolution_threshold', fallback=0.75)
    
    @property
    def enable_outlier_cluster_detection(self) -> bool:
        """Enable outlier cluster detection for wrong character assignments."""
        return self.get_bool('cluster_assignment', 'enable_outlier_cluster_detection', fallback=True)
    
    @property
    def outlier_distance_threshold(self) -> float:
        """Distance threshold for detecting outlier clusters."""
        return self.get_float('cluster_assignment', 'outlier_distance_threshold', fallback=0.3)
    
    @property
    def outlier_score_threshold(self) -> int:
        """Minimum outlier score to flag a cluster as wrong assignment."""
        return self.get_int('cluster_assignment', 'outlier_score_threshold', fallback=4)
    
    @property
    def cluster_protection_percentage_threshold(self) -> float:
        """Percentage threshold for protecting clusters from spatial outlier detection."""
        return self.get_float('cluster_assignment', 'cluster_protection_percentage_threshold', fallback=75.0)
    
    @property
    def cluster_protection_min_faces(self) -> int:
        """Minimum face count for cluster protection from spatial outlier detection."""
        return self.get_int('cluster_assignment', 'cluster_protection_min_faces', fallback=15)
    
    @property
    def show_all_face_candidates_in_srt(self) -> bool:
        """Whether to show all detected faces in enhanced SRT including low-confidence ones."""
        return self.get_bool('cluster_assignment', 'show_all_face_candidates_in_srt', fallback=True)

    @property
    def cross_episode_character_similarity_threshold(self) -> float:
        """Cosine similarity threshold for matching new clusters to existing character medians across episodes."""
        return self.get_float('cross_episode', 'character_similarity_threshold', fallback=0.75)

    @property
    def enable_debug_output(self) -> bool:
        """Enable generation of verbose debug files (dialogue_faces_debug.json, etc.)."""
        return self.get_bool('output', 'enable_debug_files', fallback=False)

    # Sex validation properties
    @property
    def enable_sex_validation(self) -> bool:
        """Enable sex-based validation of speaker clusters."""
        return self.get_bool('sex_validation', 'enable_sex_validation', fallback=True)
    
    @property
    def max_faces_for_sex_analysis(self) -> int:
        """Maximum number of faces to analyze per cluster for sex detection."""
        return self.get_int('sex_validation', 'max_faces_for_sex_analysis', fallback=5)
    
    @property
    def sex_confidence_threshold(self) -> float:
        """Minimum confidence difference between M/F required for reliable sex determination."""
        return self.get_float('sex_validation', 'sex_confidence_threshold', fallback=75.0)
    
    @property
    def sex_reassignment_similarity_threshold(self) -> float:
        """Similarity threshold for cluster reassignment to same-sex characters."""
        return self.get_float('sex_validation', 'sex_reassignment_similarity_threshold', fallback=0.7)
    
    @property
    def enable_sex_validation_logging(self) -> bool:
        """Enable detailed logging of sex validation decisions."""
        return self.get_bool('sex_validation', 'enable_sex_validation_logging', fallback=True)

    # Character median comparison thresholds
    @property
    def character_median_similarity_threshold(self) -> float:
        """Minimum similarity threshold for face-to-character median qualification."""
        return self.get_float('character_median_matching', 'similarity_threshold', fallback=0.50)
    
    @property
    def character_median_assignment_threshold(self) -> float:
        """Minimum similarity threshold for direct character median assignment."""
        return self.get_float('character_median_matching', 'assignment_threshold', fallback=0.70)


# Global configuration instance
config = Config()
