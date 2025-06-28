from src.utils.logger_utils import setup_logging
from src.plot_processing.plot_text_processing import replace_pronouns_with_names
from src.plot_processing.plot_semantic_processing import semantic_split
from src.plot_processing.plot_ner_entity_extraction import extract_and_refine_entities_with_path_handler, force_reextract_entities_with_path_handler, substitute_appellations_with_names, normalize_entities_names_to_best_appellation
from src.plot_processing.plot_processing_models import EntityLink, EntityLinkEncoder
from src.plot_processing.subtitle_processing import (
    parse_srt_file, 
    generate_plot_from_subtitles, 
    map_scene_to_timestamps, 
    save_plot_files,
    save_scene_timestamps,
    PlotScene
)
from src.utils.text_utils import load_text, clean_text
from src.ai_models.ai_models import get_llm
from src.path_handler import PathHandler
import os
import json
import agentops
from src.langgraph_narrative_arcs_extraction.narrative_arc_graph import extract_narrative_arcs
from src.ai_models.ai_models import LLMType
from src.plot_processing.process_suggested_arcs import process_suggested_arcs

from dotenv import load_dotenv

load_dotenv(override=True)

agentops.init()

# Set up logging
logger = setup_logging(__name__)

def process_single_episode(series: str, season: str, episode: str) -> None:
    """
    Process a single episode with given series, season, and episode.
    
    Args:
        series (str): Series name (e.g., "GA")
        season (str): Season name (e.g., "S01")
        episode (str): Episode name (e.g., "E01")
    """
    try:
        path_handler = PathHandler(series, season, episode)
        logger.info(f"Processing {series} {season} {episode}")
        process_text(path_handler, series, season, episode)
        logger.info(f"Successfully processed {series} {season} {episode}")
    except Exception as e:
        logger.error(f"Error processing {series} {season} {episode}: {e}")
        raise

def process_text(path_handler: PathHandler, series: str, season: str, episode: str) -> None:
    """
    Process the input SRT file to generate plot and perform semantic analysis.

    Args:
        path_handler (PathHandler): An instance of PathHandler to manage file paths.
        series (str): Series name
        season (str): Season name  
        episode (str): Episode name

    Returns:
        None
    """
    try:
        logger.info("Initializing language models.")
        llm_intelligent = get_llm(LLMType.INTELLIGENT)
        llm_cheap = get_llm(LLMType.CHEAP)
        
        # Step 1: Generate plot from SRT file if it doesn't exist
        raw_plot_path = path_handler.get_raw_plot_file_path()
        
        if not os.path.exists(raw_plot_path):
            # Look for SRT file
            srt_path = path_handler.get_srt_file_path()
            logger.info(f"Looking for SRT file: {srt_path}")
            
            if not os.path.exists(srt_path):
                logger.error(f"SRT file not found: {srt_path}")
                return
            
            logger.info("Generating plot from SRT subtitles")
            
            # Parse SRT file
            subtitles = parse_srt_file(srt_path)
            
            # Load previous season summary for context
            season_summary_path = path_handler.get_season_summary_path()
            from src.plot_processing.subtitle_processing import load_previous_season_summary
            previous_season_summary = load_previous_season_summary(season_summary_path)
            
            # Generate plot from subtitles with optional season context
            plot_data = generate_plot_from_subtitles(subtitles, llm_intelligent, previous_season_summary)
            
            # Save plot files (TXT and JSON)
            episode_prefix = f"{series}{season}{episode}"
            episode_dir = os.path.dirname(raw_plot_path)
            txt_path, scenes_json_path = save_plot_files(plot_data, episode_dir, episode_prefix)
            
            logger.info(f"Generated plot saved to: {txt_path}")
        else:
            logger.info(f"Plot file already exists: {raw_plot_path}")
        
        # Step 2: Map scenes to timestamps (check if timestamps file exists)
        episode_prefix = f"{series}{season}{episode}"
        episode_dir = os.path.dirname(raw_plot_path)
        timestamps_path = path_handler.get_scene_timestamps_path()
        
        if not os.path.exists(timestamps_path):
            logger.info("Mapping scenes to subtitle timestamps")
            
            # Parse SRT file for timestamp mapping
            srt_path = path_handler.get_srt_file_path()
            if not os.path.exists(srt_path):
                logger.error(f"SRT file not found for timestamp mapping: {srt_path}")
            else:
                subtitles = parse_srt_file(srt_path)
                
                # Load plot scenes data
                scenes_json_path = path_handler.get_plot_scenes_json_path()
                if os.path.exists(scenes_json_path):
                    with open(scenes_json_path, 'r') as f:
                        plot_data = json.load(f)
                    
                    scenes = []
                    for scene_data in plot_data.get("scenes", []):
                        scene = PlotScene(
                            scene_number=scene_data.get("scene_number", len(scenes) + 1),
                            plot_segment=scene_data.get("plot_segment", "")
                        )
                        scenes.append(scene)
                    
                    # Map each scene to timestamps one by one
                    mapped_scenes = []
                    for scene in scenes:
                        mapped_scene = map_scene_to_timestamps(scene, subtitles, llm_cheap)
                        mapped_scenes.append(mapped_scene)
                    
                    # Save scene timestamps
                    timestamps_path = save_scene_timestamps(mapped_scenes, episode_dir, episode_prefix)
                    logger.info(f"Scene timestamps saved to: {timestamps_path}")
                else:
                    logger.error(f"Plot scenes JSON not found: {scenes_json_path}")
        else:
            logger.info(f"Scene timestamps already exist: {timestamps_path}")
        
        # Step 3: Continue with existing processing pipeline 
        # Use the full generated plot for further processing
        raw_plot = load_text(raw_plot_path)
        logger.debug(f"Raw plot content: {raw_plot[:100]}...")
        
        # Use the original raw plot directly (no text simplification needed since it's already well-structured)
        simplified_text = raw_plot
        logger.info("Using full generated plot for further processing")
        
        # Named the entities if the file does not exist
        named_file_path = path_handler.get_named_plot_file_path()
        
        if not os.path.exists(named_file_path):
            logger.info(f"Replacing pronouns with names and saving to: {named_file_path}")
            named_plot = replace_pronouns_with_names(text=simplified_text, intelligent_llm=llm_intelligent, cheap_llm=llm_cheap)
            logger.debug(f"Named plot content: {named_plot[:100]}...")  # Log the first 100 characters for brevity
            
            with open(named_file_path, "w") as named_file:
                named_file.write(named_plot)
        
        else:
            logger.info(f"Loading named plot from: {named_file_path}")
            with open(named_file_path, "r") as named_file:
                named_plot = named_file.read()
        
        # Extract entities using the new NER-based method
        episode_extracted_refined_entities_path = path_handler.get_episode_refined_entities_path()
        season_extracted_refined_entities_path = path_handler.get_season_extracted_refined_entities_path()
        
        if not os.path.exists(episode_extracted_refined_entities_path):
            logger.info("Extracting and refining entities from named plot.")
            entities = extract_and_refine_entities_with_path_handler(path_handler, series)
        else:
            logger.info(f"Loading existing entities from: {episode_extracted_refined_entities_path}")
            with open(episode_extracted_refined_entities_path, "r") as episode_extracted_refined_entities_file:
                entities_data = json.load(episode_extracted_refined_entities_file)
                entities = [EntityLink(**entity) for entity in entities_data]

        
        entity_substituted_plot_path = path_handler.get_entity_substituted_plot_file_path()
        entity_normalized_plot_path = path_handler.get_entity_normalized_plot_file_path()
        
        if not os.path.exists(entity_substituted_plot_path) :
            entity_substituted_plot = substitute_appellations_with_names(named_plot, entities, llm_intelligent)
            with open(entity_substituted_plot_path, "w") as entity_substituted_plot_file:
                entity_substituted_plot_file.write(entity_substituted_plot)
        else:
            logger.info(f"Loading entity substituted plot from: {entity_substituted_plot_path}")
            with open(entity_substituted_plot_path, "r") as entity_substituted_plot_file:
                entity_substituted_plot = entity_substituted_plot_file.read()
        
        
        if not os.path.exists(entity_normalized_plot_path):
            entity_normalized_plot = normalize_entities_names_to_best_appellation(entity_substituted_plot, entities)
            with open(entity_normalized_plot_path, "w") as entity_normalized_plot_file:
                entity_normalized_plot_file.write(entity_normalized_plot)
        
        else:           
            logger.info(f"Loading entity normalized plot from: {entity_normalized_plot_path}")
            with open(entity_normalized_plot_path, "r") as entity_normalized_plot_file:
                entity_normalized_plot = entity_normalized_plot_file.read()
                
        # Perform semantic splitting if the file does not exist
        semantic_segments_path = path_handler.get_semantic_segments_path()
          
        if not os.path.exists(semantic_segments_path):
            logger.info("Performing semantic splitting.")
            semantic_segments = semantic_split(text=entity_normalized_plot, llm=llm_intelligent)

            with open(semantic_segments_path, "w", encoding='utf-8') as semantic_segments_file:
                json.dump(semantic_segments, semantic_segments_file, indent=2, ensure_ascii=False)
                logger.info(f"Semantic splitting complete. Results saved to {semantic_segments_path}")
        else:
            logger.info(f"Loading semantic segments from: {semantic_segments_path}")
            with open(semantic_segments_path, "r") as semantic_segments_file:
                semantic_segments = json.load(semantic_segments_file)          
        
        # Use the original generated plot (no summarization needed)
        logger.info("Using full generated plot for narrative arc extraction")
                
        suggested_episode_arc_path = path_handler.get_suggested_episode_arc_path()
        
        if not os.path.exists(suggested_episode_arc_path):
            # Prepare file paths for narrative arc extraction
            file_paths_for_graph = {
                "episode_plot_path": path_handler.get_raw_plot_file_path(),  # Use the original generated plot
                "seasonal_narrative_analysis_output_path": path_handler.get_season_narrative_analysis_path(),
                "episode_narrative_analysis_output_path": path_handler.get_episode_narrative_analysis_path(),
                "season_entities_path": path_handler.get_season_extracted_refined_entities_path(),
                "suggested_episode_arc_path": suggested_episode_arc_path
            }

            # Extract narrative arcs using LangGraph
            logger.info("Extracting narrative arcs.")
            extract_narrative_arcs(file_paths_for_graph, series, season, episode)

        # Process the suggested arcs and update the database
        logger.info("Processing suggested arcs and updating database.")

        updated_arcs = process_suggested_arcs(
            suggested_episode_arc_path,
            series,
            season,
            episode
        )

        logger.info(f"Updated {len(updated_arcs)} arcs in the database.")
        
        # Step: Create/update season summary after episode processing
        logger.info("Creating/updating season summary.")
        try:
            from backend.src.plot_processing.plot_summarizing import create_or_update_season_summary
            
            # Define paths for season summary management
            episode_plot_path = path_handler.get_raw_plot_file_path()
            season_summary_path = path_handler.get_season_summary_path()
            episode_summary_path = path_handler.get_episode_summary_path()
            
            # Create/update season summary
            season_summary = create_or_update_season_summary(
                episode_plot_path,
                season_summary_path,
                episode_summary_path,
                llm_intelligent
            )
            
            if season_summary:
                logger.info("✅ Season summary updated successfully.")
            else:
                logger.warning("⚠️ Season summary creation failed.")
                
        except Exception as e:
            logger.error(f"❌ Error creating season summary: {e}")
            # Don't fail the entire process if summary creation fails
        
        logger.info("Processing complete.")

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process TV series episodes')
    parser.add_argument('--series', default='GA', help='Series name (default: GA)')
    parser.add_argument('--season', default='S01', help='Season name (default: S01)')
    parser.add_argument('--episode', help='Specific episode to process (e.g., E01)')
    parser.add_argument('--start-episode', type=int, default=1, help='Starting episode number (default: 1)')
    parser.add_argument('--end-episode', type=int, default=9, help='Ending episode number (default: 9)')
    
    args = parser.parse_args()
    
    logger.info("Starting text processing.")
    
    if args.episode:
        # Process single episode
        logger.info(f"Processing single episode: {args.series} {args.season} {args.episode}")
        process_single_episode(args.series, args.season, args.episode)
    else:
        # Process range of episodes
        logger.info(f"Processing episodes from E{args.start_episode:02d} to E{args.end_episode:02d}")
        for ep in range(args.start_episode, args.end_episode + 1):
            episode = f"E{ep:02d}"
            process_single_episode(args.series, args.season, episode)
