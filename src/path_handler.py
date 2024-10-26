import os

class PathHandler:
    def __init__(self, series: str, season: str, episode: str, base_dir: str = "data"):
        self.series = series
        self.season = season
        self.episode = episode
        self.base_dir = base_dir

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

    def get_season_plot_file_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, f"{self.series}{self.season}_season_plot.txt")

    def get_semantic_segments_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_plot_semantic_segments.json")

    def get_episode_narrative_arcs_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_multiagent_episode_narrative_arcs.json")

    def get_season_narrative_arcs_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, f"{self.series}{self.season}_multiagent_season_narrative_arcs.json")

    def get_episode_narrative_analysis_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_multiagent_episode_narrative_analysis.txt")

    def get_season_narrative_analysis_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, f"{self.series}{self.season}_multiagent_season_narrative_analysis.txt")

    def get_episode_refined_entities_path(self) -> str:
        """Path for saving refined entities for the episode."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_refined_entities.json")

    def get_episode_raw_spacy_entities_path(self) -> str:
        """Path for saving raw spaCy entities for the episode."""
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_raw_spacy_entities.json")
    
    def get_season_extracted_refined_entities_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, f"{self.series}{self.season}_extracted_entities.json")
    
    def get_plot_localized_sentences_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_plot_localized_sentences.json")
    
    def get_summarized_plot_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_summarized_plot.txt")
    
    def get_suggested_episode_arc_path(self) -> str:
        return os.path.join(self.base_dir, self.series, self.season, self.episode, f"{self.series}{self.season}{self.episode}_multiagent_suggested_episode_arcs.json")
