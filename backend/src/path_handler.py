import os
from typing import List

class PathHandler:
    def __init__(self, series: str, season: str, episode: str, base_dir: str = "data"):
        self.series = series
        self.season = season
        self.episode = episode
        self.base_dir = base_dir

    def get_series(self) -> str:
        """Get the series name."""
        return self.series
    
    def get_season(self) -> str:
        """Get the season name."""
        return self.season
    
    def get_episode(self) -> str:
        """Get the episode name."""
        return self.episode

    # Define methods to get file paths
    def get_raw_plot_file_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_plot.txt")

    def get_full_dialogues_file_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_full_dialogues.json")

    def get_simplified_plot_file_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_plot_simplified.txt")

    def get_named_plot_file_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_plot_named.txt")
    
    def get_entity_substituted_plot_file_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_plot_entities_substituted.txt")

    def get_entity_normalized_plot_file_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_plot_entities_normalized.txt")

    def get_episode_refined_entities_path(self) -> str:
        """Path for saving refined entities for the episode."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_refined_entities.json")

    def get_episode_raw_spacy_entities_path(self) -> str:
        """Path for saving raw spaCy entities for the episode."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_raw_spacy_entities.json")
    
    def get_season_extracted_refined_entities_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, f"{self.series}{self.season}_extracted_entities.json")
    
    def get_suggested_episode_arc_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_multiagent_suggested_episode_arcs.json")

    def get_srt_file_path(self) -> str:
        """Get the path to the SRT subtitle file for this episode."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}.srt")

    def get_enhanced_srt_path(self) -> str:
        """Get the path to the enhanced SRT file with speaker identification for this episode."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_enhanced.srt")

    def get_possible_speakers_srt_path(self) -> str:
        """Get the path to the SRT file with possible speakers for this episode."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_possible_speakers.srt")

    def get_plot_possible_speakers_path(self) -> str:
        """Get the path to the plot file generated from possible speakers SRT."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_plot_possible_speakers.txt")

    def get_corrected_scene_timestamps_path(self) -> str:
        """Get the path to the corrected scene timestamps file."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_corrected_scene_timestamps.json")

    def get_plot_scenes_json_path(self) -> str:
        """Get the path to the plot scenes JSON file."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_plot_scenes.json")
    
    def get_scene_timestamps_path(self) -> str:
        """Get the path to the scene timestamps JSON file."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_scene_timestamps.json")

    # Face tracking and speaker identification paths
    def get_video_file_path(self) -> str:
        """Get the path to the main video file for this episode."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}.mp4")

    def get_audio_file_path(self) -> str:
        """Get the path to the extracted audio file for this episode."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}.wav")

    def get_vocals_file_path(self, audio_path: str = None) -> str:
        """
        Get the path to the vocals file extracted from the audio.
        
        Args:
            audio_path: Path to the original audio file. If None, uses the default audio path.
            
        Returns:
            Path to the vocals file
        """
        if audio_path is None:
            audio_path = self.get_audio_file_path()
        
        # Get the directory and filename from the audio path
        audio_dir = os.path.dirname(audio_path)
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Create vocals directory
        vocals_dir = os.path.join(audio_dir, "vocals")
        
        # Return the vocals file path
        return os.path.join(vocals_dir, f"{audio_name}_vocals.wav")
    
    def get_speaker_analysis_path(self) -> str:
        """Get the path to the speaker analysis JSON file."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_speaker_analysis.json")
    
    def get_dialogue_faces_csv_path(self) -> str:
        """Get the path to the dialogue faces CSV file."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_dialogue_faces.csv")
    
    def get_dialogue_faces_debug_path(self) -> str:
        """Get the path to the dialogue faces debug JSON file."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_dialogue_faces_debug.json")
    
    def get_character_medians_summary_path(self) -> str:
        """Get the path to the character medians summary JSON file."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_character_medians.json")
    
    def get_face_processing_summary_path(self) -> str:
        """Get the path to the streamlined face processing summary JSON file."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_face_processing_summary.json")
    
    def get_dialogue_faces_dir(self) -> str:
        """Get the directory for storing dialogue face images."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, "dialogue_faces")
    
    def get_dialogue_frames_dir(self) -> str:
        """Get the directory for storing dialogue frame images."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, "dialogue_frames")
    
    def get_dialogue_embeddings_dir(self) -> str:
        """Get the directory for storing dialogue face embeddings."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, "dialogue_embeddings")
    
    def get_dialogue_embeddings_metadata_csv_path(self) -> str:
        """Get the path to the dialogue embeddings metadata CSV."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_dialogue_embeddings_metadata.csv")
    
    def get_cluster_visualization_dir(self) -> str:
        """Get the directory for cluster visualizations."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, "cluster_visualizations")
    
    def get_face_clusters_summary_path(self) -> str:
        """Get the path to the face clusters summary JSON."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_face_clusters_summary.json")
    
    def get_face_clusters_assignments_path(self) -> str:
        """Get the path to the face cluster assignments JSON for multi-face processing."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_face_cluster_assignments.json")
    
    def get_speaker_face_associations_path(self) -> str:
        """Get the path to the speaker-face associations JSON."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_speaker_face_associations.json")
    
    def get_llm_checkpoint_path(self) -> str:
        """Get the path to the LLM speaker assignments checkpoint file."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_llm_checkpoint.json")
    
    def get_speaker_identification_checkpoint_dir(self) -> str:
        """Get the directory for speaker identification checkpoints."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, "speaker_identification_checkpoints")

    def get_speaker_identification_checkpoint_path(self, mode: str) -> str:
        """Get the path to the speaker identification checkpoint file for a specific mode."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_speaker_id_{mode}_checkpoint.json")
    
    def get_chroma_db_path(self) -> str:
        """Get the path to the ChromaDB directory."""
        return os.path.join("narrative_storage", "chroma_db")
    
    def get_dialogue_json_path(self) -> str:
        """Get the path to the parsed dialogue JSON file."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_dialogue.json")
    
    def get_episode_code(self) -> str:
        """Get the episode code (e.g., GAS01E01)."""
        return f"{self.series}{self.season}{self.episode}"

    def get_season_summary_path(self) -> str:
        """Get the path to the cumulative season summary file."""
        return os.path.join(self.base_dir, self.series, self.season, f"{self.series}{self.season}_season_summary.txt")
    
    def get_episode_summary_path(self) -> str:
        """Get the path to the episode summary file."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_episode_summary.txt")

    def get_present_running_plotlines_path(self) -> str:
        """Get the path to the present running plotlines file."""
        recap_dir = os.path.join(self.base_dir, self.series, self.season, self.episode, "recap_files")
        return os.path.join(recap_dir, f"{self.series}{self.season}{self.episode}_present_running_plotlines.json")

    # Recap generation specific paths
    def get_recap_files_dir(self) -> str:
        """Get the recap_files directory path."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, "recap_files")
    
    def get_recap_clips_json_path(self) -> str:
        """Get the path to the recap clips specifications JSON file."""
        recap_dir = self.get_recap_files_dir()
        return os.path.join(recap_dir, f"{self.series}{self.season}{self.episode}_recap_clips.json")
    
    def get_final_recap_video_path(self) -> str:
        """Get the path to the final recap video file."""
        recap_dir = self.get_recap_files_dir()
        return os.path.join(recap_dir, f"{self.series}{self.season}{self.episode}_recap.mp4")
    
    def get_recap_metadata_path(self) -> str:
        """Get the path to the recap metadata JSON file."""
        recap_dir = self.get_recap_files_dir()
        return os.path.join(recap_dir, f"{self.series}{self.season}{self.episode}_recap_metadata.json")
    
    def get_individual_clip_path(self, clip_id: str) -> str:
        """Get the path to an individual extracted clip file."""
        recap_dir = self.get_recap_files_dir()
        clips_dir = os.path.join(recap_dir, "clips")
        return os.path.join(clips_dir, f"{self.series}{self.season}{self.episode}_clip_{clip_id}.mp4")
    
    def get_recap_clips_dir(self) -> str:
        """Get the directory for storing individual recap clips."""
        recap_dir = self.get_recap_files_dir()
        return os.path.join(recap_dir, "clips")
    
    def validate_episode_processed(self) -> bool:
        """Check if episode has completed the full SEMAMORPH processing pipeline."""
        required_files = [
            self.get_plot_possible_speakers_path(),
            self.get_present_running_plotlines_path(),
            self.get_possible_speakers_srt_path(),
            self.get_video_file_path()
        ]
        return all(os.path.exists(file_path) for file_path in required_files)
    
    def get_missing_required_files(self) -> List[str]:
        """Get list of missing files required for recap generation."""
        required_files = [
            ("plot_possible_speakers", self.get_plot_possible_speakers_path()),
            ("present_running_plotlines", self.get_present_running_plotlines_path()),
            ("possible_speakers_srt", self.get_possible_speakers_srt_path()),
            ("video_file", self.get_video_file_path())
        ]
        
        missing = []
        for file_type, file_path in required_files:
            if not os.path.exists(file_path):
                missing.append(f"{file_type}: {file_path}")
        
        return missing

    @staticmethod
    def get_episode_plot_path(base_dir: str, series: str, season: str, episode: str) -> str:
        """Get the path for an episode's plot file."""
        return os.path.join(base_dir, series, season, episode, f"{series}{season}{episode}_plot.txt")
    
    @staticmethod
    def get_season_plot_path(base_dir: str, series: str, season: str) -> str:
        """Get the path for a season's plot file."""
        return os.path.join(base_dir, series, season, f"{series}{season}_season_plot.txt")
    
    @staticmethod
    def list_episode_folders(base_dir: str, series: str, season: str) -> List[str]:
        """List all episode folders in a season directory."""
        season_path = os.path.join(base_dir, series, season)
        if not os.path.exists(season_path):
            return []
        return sorted([d for d in os.listdir(season_path) 
                      if os.path.isdir(os.path.join(season_path, d)) and d.startswith('E')])
