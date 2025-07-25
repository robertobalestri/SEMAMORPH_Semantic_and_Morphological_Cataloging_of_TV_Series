from src.utils.logger_utils import setup_logging
from src.plot_processing.plot_text_processing import replace_pronouns_with_names, simplify_text
from src.plot_processing.plot_semantic_processing import semantic_split
from src.plot_processing.plot_ner_entity_extraction import extract_and_refine_entities, substitute_appellations_with_names, normalize_entities_names_to_best_appellation
from src.plot_processing.plot_processing_models import EntityLink, EntityLinkEncoder
from src.utils.text_utils import load_text, clean_text
from src.ai_models.ai_models import get_llm
from src.path_handler import PathHandler
from src.config import config
import os
import json
import shutil
import agentops
from src.langgraph_narrative_arcs_extraction.narrative_arc_graph import extract_narrative_arcs
from src.ai_models.ai_models import LLMType
from src.plot_processing.plot_summarizing import create_season_summary
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
    Process the input text file, extracting entities and performing semantic analysis.

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
        
        # Check if season summary needs to be created
        season_plot_path = path_handler.get_season_plot_file_path()
        if not os.path.exists(season_plot_path):
            logger.info("Season summary not found. Creating season summary from episode plots.")
            
            # Get all episode folders
            episode_folders = PathHandler.list_episode_folders(
                path_handler.base_dir,
                path_handler.series,
                path_handler.season
            )
            
            # Collect paths for all episode plots
            episode_plots = []
            for ep_folder in episode_folders:
                ep_plot_path = PathHandler.get_episode_plot_path(
                    path_handler.base_dir,
                    path_handler.series,
                    path_handler.season,
                    ep_folder
                )
                if os.path.exists(ep_plot_path):
                    episode_plots.append(ep_plot_path)
            
            if episode_plots:
                logger.info(f"Found {len(episode_plots)} episode plots to summarize")
                create_season_summary(episode_plots, llm_cheap, season_plot_path)
            else:
                logger.warning("No episode plots found to create season summary.")
        else:
            logger.info(f"Season summary already exists at: {season_plot_path}")

        # Simplify the text if the file does not exist
        simplified_file_path = path_handler.get_simplified_plot_file_path()
        
        if not os.path.exists(simplified_file_path):
            # Load raw text
            input_file = path_handler.get_raw_plot_file_path()
            logger.info(f"Loading raw plot from: {input_file}")
            
            if not os.path.exists(input_file):
                logger.error(f"Raw plot file not found: {input_file}")
                return
            
            raw_plot = load_text(input_file)
            logger.debug(f"Raw plot content: {raw_plot[:100]}...")  # Log the first 100 characters for brevity
            
            cleaned_plot = clean_text(raw_plot)
            logger.info("Cleaning plot text completed.")
            logger.debug(f"cleaned_plot plot content: {cleaned_plot[:100]}...")  # Log the first 100 characters for brevity
                
            logger.info(f"Simplifying text and saving to: {simplified_file_path}")
            simplified_text = simplify_text(cleaned_plot, llm_intelligent)
            logger.debug(f"Simplified text content: {simplified_text[:100]}...")  # Log the first 100 characters for brevity

            with open(simplified_file_path, "w") as simplified_file:
                simplified_file.write(simplified_text)
        else:
            logger.info(f"Loading simplified text from: {simplified_file_path}")
            with open(simplified_file_path, "r") as simplified_file:
                simplified_text = simplified_file.read()

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
            entities = extract_and_refine_entities(
                named_plot, 
                series,
                llm_intelligent, 
                episode_extracted_refined_entities_path, 
                path_handler.get_episode_raw_spacy_entities_path(),
                season_extracted_refined_entities_path  # Pass the season entities path
            )
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
        
        # Handle summarized plot creation based on configuration
        summarized_plot_path = path_handler.get_summarized_plot_path()
        
        if not os.path.exists(summarized_plot_path):
            if config.use_original_plot_as_summary:
                # Use original plot as summarized plot
                logger.info("Using original plot as summarized plot (as configured)")
                original_plot_path = path_handler.get_raw_plot_file_path()
                if os.path.exists(original_plot_path):
                    shutil.copy2(original_plot_path, summarized_plot_path)
                    logger.info(f"Copied original plot to: {summarized_plot_path}")
                else:
                    logger.warning(f"Original plot file not found: {original_plot_path}")
                    # Fall back to using simplified plot if original not found
                    simplified_plot_path = path_handler.get_simplified_plot_file_path()
                    if os.path.exists(simplified_plot_path):
                        shutil.copy2(simplified_plot_path, summarized_plot_path)
                        logger.info(f"Fallback: Copied simplified plot to: {summarized_plot_path}")
            else:
                # Generate summarized plot using LLM (existing behavior)
                logger.info("Generating summarized plot using LLM")
                # Note: The actual summarization logic should be implemented here
                # For now, we'll use the simplified plot as a fallback
                simplified_plot_path = path_handler.get_simplified_plot_file_path()
                if os.path.exists(simplified_plot_path):
                    shutil.copy2(simplified_plot_path, summarized_plot_path)
                    logger.info(f"Temporary: Using simplified plot as summarized plot: {summarized_plot_path}")
        else:
            logger.info(f"Summarized plot already exists: {summarized_plot_path}")
                
        suggested_episode_arc_path = path_handler.get_suggested_episode_arc_path()
        
        if not os.path.exists(suggested_episode_arc_path):
            # Prepare file paths for narrative arc extraction
            file_paths_for_graph = {
                "season_plot_path": path_handler.get_season_plot_file_path(),
                "episode_plot_path": path_handler.get_simplified_plot_file_path(),
                "seasonal_narrative_analysis_output_path": path_handler.get_season_narrative_analysis_path(),
                "episode_narrative_analysis_output_path": path_handler.get_episode_narrative_analysis_path(),
                "summarized_plot_path": path_handler.get_summarized_plot_path(),
                "season_entities_path": path_handler.get_season_extracted_refined_entities_path(),
                "suggested_episode_arc_path": suggested_episode_arc_path
            }

            # Extract narrative arcs using LangGraph
            if config.narrative_arc_extraction_method == "enhanced":
                logger.info("Extracting narrative arcs using ENHANCED method.")
                from src.langgraph_narrative_arcs_extraction.enhanced_narrative_arc_graph import extract_enhanced_narrative_arcs
                extract_enhanced_narrative_arcs(file_paths_for_graph, series, season, episode)
            else:
                logger.info("Extracting narrative arcs using ORIGINAL method.")
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
