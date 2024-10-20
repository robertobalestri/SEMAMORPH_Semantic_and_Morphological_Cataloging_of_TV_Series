# narrative_arc_graph.py

from typing import Dict, List, TypedDict, Tuple
from langgraph.graph import StateGraph, END
import os
import json

# Import necessary modules (adjust the import paths as needed)

from src.utils.llm_utils import clean_llm_json_response
from src.utils.logger_utils import setup_logging
from src.utils.text_utils import load_json, load_text
from src.vectorstore.vectorstore_collection import CollectionType, get_vectorstore_collection
from src.ai_models.ai_models import get_llm, LLMType
from src.narrative_classes.narrative_classes import NarrativeArc, ArcProgression
logger = setup_logging(__name__)


from .prompts import (
    SEASONAL_NARRATIVE_ANALYZER_PROMPT,
    EPISODE_NARRATIVE_ANALYZER_PROMPT,
    PRESENT_SEASON_ARCS_IDENTIFIER_PROMPT,
    ARC_EXTRACTOR_FROM_ANALYSIS_PROMPT,
    ARC_EXTRACTOR_FROM_PLOT_PROMPT,
    ARC_VERIFIER_PROMPT,
    NARRATIVE_ARC_GUIDELINES,
    ARC_PROGRESSION_VERIFIER_PROMPT
)

llm = get_llm(LLMType.INTELLIGENT)

from pydantic import BaseModel, Field
from typing import List, Optional

class IntermediateNarrativeArc(BaseModel):
    """Model representing an intermediate narrative arc during extraction process."""
    title: str = Field(..., description="The title of the narrative arc")
    arc_type: str = Field(..., description="Type of the arc such as 'Soap Arc'/'Genre-Specific Arc'/'Character Arc'/'Episodic Arc'/'Mythology Arc'")
    description: str = Field(..., description="A brief description of the narrative arc")
    episodic: bool = Field(..., description="If the arc is episodic or not")
    characters: List[str] = Field(default_factory=list, description="Characters involved in this arc")
    single_episode_progression_string: str = Field("", description="The progression of the arc for the current episode")

    class Config:
        populate_by_name = True


class NarrativeArcsExtractionState(TypedDict):
    """State representation for the narrative analysis process."""
    season_analysis: str
    episode_narrative_analysis: str
    arcs_from_analysis: List[IntermediateNarrativeArc]
    arcs_from_plot: List[IntermediateNarrativeArc]
    episode_arcs: List[IntermediateNarrativeArc]  # Final arcs after verification
    episode_arcs_with_ids: List[Tuple[NarrativeArc, str]]  # Arcs with their IDs
    season_arcs: Dict[str, Dict]  # Updated season arcs with IDs as keys
    existing_season_arcs: List[Dict]  # Existing season arcs
    present_season_arcs: List[Dict]  # Season arcs present in the episode
    file_paths: Dict[str, str]
    series: str
    season: str
    episode: str
    
def seasonal_narrative_analysis(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Analyze the season plot to identify overarching themes, character development, and narrative arcs.

    Args:
        state (NarrativeArcsExtractionState): The current state of the narrative analysis.

    Returns:
        NarrativeArcsExtractionState: The updated state after analysis.
    """
    logger.info("Starting seasonal narrative analysis.")
    
    # Check if the seasonal analysis output file already exists
    if os.path.exists(state['file_paths']['seasonal_narrative_analysis_output_path']):
        logger.info("Seasonal narrative analysis output already exists. Loading from file.")
        with open(state['file_paths']['seasonal_narrative_analysis_output_path'], 'r') as f:
            state['season_analysis'] = f.read()
        return state

    season_plot = load_text(state['file_paths']['season_plot_path'])
    logger.debug(f"Season plot content: {season_plot[:100]}...")  # Log the first 100 characters for brevity

    season_analysis = llm.invoke(SEASONAL_NARRATIVE_ANALYZER_PROMPT.format_messages(season_plot=season_plot))
    logger.info("Season narrative analysis completed.")
    
    # Save the structured season analysis
    state['season_analysis'] = season_analysis.content.strip()

    with open(state['file_paths']['seasonal_narrative_analysis_output_path'], 'w') as f:
        f.write(state['season_analysis'])
    
    logger.info("Seasonal narrative analysis output saved.")
    
    return state

def episode_narrative_analysis(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Analyze the episode plot and provide a structured analysis based on the overall season analysis.

    Args:
        state (NarrativeArcsExtractionState): The current state of the narrative analysis.

    Returns:
        NarrativeArcsExtractionState: The updated state after analysis.
    """
    logger.info("Starting episode narrative analysis.")
    
    # Check if the episode narrative analysis output file already exists
    if os.path.exists(state['file_paths']['episode_narrative_analysis_output_path']):
        logger.info("Episode narrative analysis output already exists. Loading from file.")
        with open(state['file_paths']['episode_narrative_analysis_output_path'], 'r') as f:
            state['episode_narrative_analysis'] = f.read()
        return state

    episode_plot = load_text(state['file_paths']['episode_plot_path'])
    logger.debug(f"Episode plot content: {episode_plot[:100]}...")  # Log the first 100 characters for brevity

    episode_analysis = llm.invoke(EPISODE_NARRATIVE_ANALYZER_PROMPT.format_messages(
        episode_plot=episode_plot,
        season_analysis=state['season_analysis'],
    ))
    
    logger.info("Episode narrative analysis completed.")
    
    # Save the structured episode analysis
    state['episode_narrative_analysis'] = episode_analysis.content.strip()

    with open(state['file_paths']['episode_narrative_analysis_output_path'], 'w') as f:
        f.write(state['episode_narrative_analysis'])
    
    logger.info("Episode narrative analysis output saved.")

    return state

def identify_present_season_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Identify which existing season arcs are clearly present in the current episode."""
    logger.info("Identifying present season arcs in the episode.")
    # Read episode plot
    summarized_episode_plot = load_text(state['file_paths']['summarized_plot_path'])
    logger.debug(f"Episode summarized plot content: {summarized_episode_plot[:100]}...")

    # Load existing season arcs
    existing_season_arcs = load_json(state['file_paths']['season_narrative_arcs_path'])
    state['existing_season_arcs'] = existing_season_arcs  # Store in state for later use

    # Prepare existing season arcs summaries
    existing_season_arcs_summaries = [
        {"title": arc['title'], "description": arc['description']}
        for arc in existing_season_arcs if not arc['episodic']
    ]

    if len(existing_season_arcs_summaries) == 0:
        logger.warning("No existing season arcs found. Skipping present season arcs identification.")
        state['present_season_arcs'] = []
        return state

    response = llm.invoke(PRESENT_SEASON_ARCS_IDENTIFIER_PROMPT.format_messages(
        summarized_episode_plot=summarized_episode_plot,
        existing_season_arcs_summaries=json.dumps(existing_season_arcs_summaries, indent=2),
    ))

    try:
        arcs_data = clean_llm_json_response(response.content)
        present_arcs = []
        for arc in arcs_data:
            present_arcs.append({
                "title": arc['title'],
                "description": arc['description'],
                "presence_explanation": arc.get('presence_explanation', '')
            })
        state['present_season_arcs'] = present_arcs
        logger.info(f"Identified {len(present_arcs)} season arcs present in the episode.")
    except Exception as e:
        logger.error(f"Error identifying present season arcs: {e}")
        state['present_season_arcs'] = []

    return state

def extract_arcs_from_analysis(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Extract narrative arcs from the seasonal analysis and episode analysis."""
    logger.info("Extracting arcs from seasonal and episode analyses.")

    # Read episode analysis
    if not state.get('episode_narrative_analysis'):
        episode_analysis = load_text(state['file_paths']['episode_narrative_analysis_output_path'])
        state['episode_narrative_analysis'] = episode_analysis
    else:
        episode_analysis = state['episode_narrative_analysis']

    # Use present season arcs identified earlier
    present_season_arcs_summaries = [
        {"title": arc['title'], "description": arc['description']}
        for arc in state.get('present_season_arcs', [])
    ]

    response = llm.invoke(ARC_EXTRACTOR_FROM_ANALYSIS_PROMPT.format_messages(
        episode_analysis=episode_analysis,
        present_season_arcs_summaries=json.dumps(present_season_arcs_summaries, indent=2),
        guidelines=NARRATIVE_ARC_GUIDELINES
    ))

    try:
        arcs_data = clean_llm_json_response(response.content)
        logger.debug(f"Cleaned LLM response: {json.dumps(arcs_data, indent=2)}")
        arcs = []
        for arc in arcs_data:
            new_arc = IntermediateNarrativeArc(
                title=arc['title'],
                arc_type=arc['arc_type'],
                description=arc['description'],
                episodic=arc['episodic'],
                characters=arc['characters'],
                single_episode_progression_string=arc['single_episode_progression_string']
            )
            arcs.append(new_arc)
        state['arcs_from_analysis'] = arcs
        logger.info(f"Extracted {len(arcs)} arcs from analyses.")
        logger.debug(f"Extracted arcs: {json.dumps([arc.model_dump() for arc in arcs], indent=2)}")
    except Exception as e:
        logger.error(f"Error extracting arcs from analyses: {e}")
        state['arcs_from_analysis'] = []

    return state

def extract_arcs_from_plot(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Extract narrative arcs from the episode plot, considering existing season arcs."""
    logger.info("Extracting arcs from episode plot.")

    # Read episode plot
    episode_plot = load_text(state['file_paths']['episode_plot_path'])
    #season_plot = load_text(state['file_paths']['season_plot_path'])

    # Use present season arcs identified earlier
    present_season_arcs_summaries = [
        {"title": arc['title'], "description": arc['description']}
        for arc in state.get('present_season_arcs', [])
    ]

    response = llm.invoke(ARC_EXTRACTOR_FROM_PLOT_PROMPT.format_messages(
        #season_plot=season_plot,
        episode_plot=episode_plot,
        present_season_arcs_summaries=json.dumps(present_season_arcs_summaries, indent=2),
        guidelines=NARRATIVE_ARC_GUIDELINES
    ))

    try:
        arcs_data = clean_llm_json_response(response.content)
        logger.debug(f"Cleaned LLM response: {json.dumps(arcs_data, indent=2)}")
        arcs = []
        for arc in arcs_data:
            new_arc = IntermediateNarrativeArc(
                title=arc['title'],
                arc_type=arc['arc_type'],
                description=arc['description'],
                episodic=arc['episodic'],
                characters=arc['characters'],
                single_episode_progression_string=arc['single_episode_progression_string']
            )
            arcs.append(new_arc)
        state['arcs_from_plot'] = arcs
        logger.info(f"Extracted {len(arcs)} arcs from plot.")
        logger.debug(f"Extracted arcs: {json.dumps([arc.model_dump() for arc in arcs], indent=2)}")
    except Exception as e:
        logger.error(f"Error extracting arcs from plot: {e}")
        state['arcs_from_plot'] = []

    return state

def verify_and_finalize_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Verify and finalize the arcs, ensuring they have the right metadata, titles, descriptions."""
    logger.info("Verifying and finalizing arcs.")

    # Combine arcs from analysis and plot
    arcs_from_analysis = state.get('arcs_from_analysis', [])
    arcs_from_plot = state.get('arcs_from_plot', [])

    combined_arcs = arcs_from_analysis + arcs_from_plot

    # Read episode plot
    episode_plot = load_text(state['file_paths']['episode_plot_path'])

    # Prepare arcs to verify
    arcs_to_verify = combined_arcs

    response = llm.invoke(ARC_VERIFIER_PROMPT.format_messages(
        episode_plot=episode_plot,
        arcs_to_verify=arcs_to_verify,
        present_season_arcs_summaries=json.dumps(state['present_season_arcs'], indent=2),
        guidelines=NARRATIVE_ARC_GUIDELINES
    ))

    try:
        arcs_data = clean_llm_json_response(response.content)
        arcs = []
        for arc in arcs_data:
            new_arc = IntermediateNarrativeArc(
                title=arc['title'],
                arc_type=arc['arc_type'],
                description=arc['description'],
                episodic=arc['episodic'],
                characters=arc['characters'],
                single_episode_progression_string=arc['single_episode_progression_string']
            )
            arcs.append(new_arc)

        state['episode_arcs'] = arcs
        logger.info(f"Verified and finalized {len(arcs)} unique arcs.")
    except Exception as e:
        logger.error(f"Error verifying arcs: {e}")
        state['episode_arcs'] = combined_arcs  # Use the combined arcs if verification fails

    return state

def verify_arc_progression(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Verify and adjust the progression and description of each arc."""
    logger.info("Verifying arc progressions.")

    # Read episode plot
    episode_plot = load_text(state['file_paths']['episode_plot_path'])

    verified_arcs = []
    for arc in state['episode_arcs']:
        response = llm.invoke(ARC_PROGRESSION_VERIFIER_PROMPT.format_messages(
            episode_plot=episode_plot,
            arc_to_verify=arc.model_dump()
        ))

        try:
            verified_arc_data = clean_llm_json_response(response.content)
            verified_arc = IntermediateNarrativeArc(
                title=verified_arc_data['title'],
                arc_type=verified_arc_data['arc_type'],
                description=verified_arc_data['description'],
                episodic=verified_arc_data['episodic'],
                characters=verified_arc_data['characters'],
                single_episode_progression_string=verified_arc_data['single_episode_progression_string']
            )
            verified_arcs.append(verified_arc)
        except Exception as e:
            logger.error(f"Error verifying arc progression: {e}")
            verified_arcs.append(arc)  # Keep the original arc if verification fails

    state['episode_arcs'] = verified_arcs
    logger.info(f"Verified progressions for {len(verified_arcs)} arcs.")
    return state

def update_vectorstore(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Update the vector store with the verified narrative arcs and collect IDs."""
    logger.info("Updating vector store with verified narrative arcs.")
    vectorstore_collection = get_vectorstore_collection(collection_type=CollectionType.NARRATIVE_ARCS)
    
    # Convert IntermediateNarrativeArc to NarrativeArc
    final_arcs = []
    for arc in state['episode_arcs']:
        progression = ArcProgression(
            content=arc.single_episode_progression_string,
            series=state['series'],
            season=state['season'],
            episode=state['episode']
        )
        final_arc = NarrativeArc(
            title=arc.title,
            arc_type=arc.arc_type,
            description=arc.description,
            episodic=arc.episodic,
            characters=arc.characters,
            series=state['series'],
            progressions=[progression]
        )
        final_arcs.append(final_arc)
    
    arc_id_map = vectorstore_collection.add_or_update_narrative_arcs(
        final_arcs,
        state['series'],
        state['season'],
        state['episode']
    )
    state['episode_arcs_with_ids'] = arc_id_map
    
    # Verification step
    for arc, arc_id in arc_id_map:
        verified_arc = vectorstore_collection.get_arc_by_id(arc_id)
        if verified_arc:
            logger.info(f"Verified arc '{verified_arc.title}' has {len(verified_arc.progressions)} progressions")
            for prog in verified_arc.progressions:
                logger.debug(f"progression for episode {prog.episode}: {prog.content}")
        else:
            logger.warning(f"Could not verify arc with id {arc_id}")
    
    logger.info("Vector store updated with verified narrative arcs.")
    return state

def manage_season_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Update the season arcs by adding new arcs and updating existing ones."""
    logger.info("Managing season arcs.")

    vectorstore_collection = get_vectorstore_collection(collection_type=CollectionType.NARRATIVE_ARCS)
    
    # Fetch all season arcs from the vectorstore
    season_arcs = vectorstore_collection.get_all_season_arcs(state['series'], state['season'])
    
    state['season_arcs'] = season_arcs
    logger.info(f"Updated season arcs. Total arcs: {len(season_arcs)}")
    return state

def save_episode_arcs_to_json(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Save the episode arcs with IDs to a JSON file."""
    logger.info("Saving episode arcs with IDs to JSON file.")

    episode_arcs = []
    for arc, arc_id in state['episode_arcs_with_ids']:
        # The arc object here should already have the correct progression from the vectorstore
        episode_arc = NarrativeArc(
            id=arc_id,
            title=arc.title,
            arc_type=arc.arc_type,
            description=arc.description,
            episodic=arc.episodic,
            characters=arc.characters,
            series=arc.series,
            progressions=[arc.progressions[-1]]  # Keep only the last progression added that refers to the current episode
        )
        episode_arcs.append(episode_arc)

    episode_arcs_data = [arc.model_dump() for arc in episode_arcs]

    with open(state['file_paths']['episode_narrative_arcs_path'], 'w') as f:
        json.dump(episode_arcs_data, f, indent=2)
    logger.info("Episode narrative arcs with IDs saved.")

    return state

def save_season_arcs_to_json(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Save the season arcs with all progressions to a JSON file."""
    logger.info("Saving season arcs to JSON file.")

    # The season_arcs in the state should already be in the correct format
    season_arcs_data = [arc.model_dump() for arc in state['season_arcs']]

    with open(state['file_paths']['season_narrative_arcs_path'], 'w') as f:
        json.dump(season_arcs_data, f, indent=2)
    logger.info("Season narrative arcs saved.")

    return state

# Adjust the workflow to include the new agent
def create_narrative_arc_graph():
    """Create and configure the state graph for narrative arc extraction."""
    workflow = StateGraph(NarrativeArcsExtractionState)

    # Add all existing nodes
    workflow.add_node("season_analysis_node", seasonal_narrative_analysis)
    workflow.add_node("episode_analysis_node", episode_narrative_analysis)
    workflow.add_node("identify_present_season_arcs_node", identify_present_season_arcs)
    workflow.add_node("extract_arcs_from_analysis_node", extract_arcs_from_analysis)
    workflow.add_node("extract_arcs_from_plot_node", extract_arcs_from_plot)
    workflow.add_node("verify_arc_progression_node", verify_arc_progression)  # New node
    workflow.add_node("verify_and_finalize_arcs_node", verify_and_finalize_arcs)
    workflow.add_node("update_vectorstore_node", update_vectorstore)
    workflow.add_node("manage_season_arcs_node", manage_season_arcs)
    workflow.add_node("save_episode_arcs_to_json_node", save_episode_arcs_to_json)
    workflow.add_node("save_season_arcs_to_json_node", save_season_arcs_to_json)

    # Set the entry point and edges
    workflow.set_entry_point("season_analysis_node")
    workflow.add_edge("season_analysis_node", "episode_analysis_node")
    workflow.add_edge("episode_analysis_node", "identify_present_season_arcs_node")
    workflow.add_edge("identify_present_season_arcs_node", "extract_arcs_from_analysis_node")
    workflow.add_edge("extract_arcs_from_analysis_node", "extract_arcs_from_plot_node")
    workflow.add_edge("extract_arcs_from_plot_node", "verify_arc_progression_node")  # New edge
    workflow.add_edge("verify_arc_progression_node", "verify_and_finalize_arcs_node")  # Updated edge
    workflow.add_edge("verify_and_finalize_arcs_node", "update_vectorstore_node")
    workflow.add_edge("update_vectorstore_node", "manage_season_arcs_node")
    workflow.add_edge("manage_season_arcs_node", "save_episode_arcs_to_json_node")
    workflow.add_edge("save_episode_arcs_to_json_node", "save_season_arcs_to_json_node")
    workflow.add_edge("save_season_arcs_to_json_node", END)

    logger.info("Narrative arc graph workflow created successfully.")

    return workflow.compile()

# Function to run the graph remains unchanged
def extract_narrative_arcs(file_paths: Dict[str, str], series: str, season: str, episode: str) -> NarrativeArcsExtractionState:
    """Extract narrative arcs from the provided file paths and save the results."""
    logger.info("Starting extract_narrative_arcs function")
    graph = create_narrative_arc_graph()
    initial_state = NarrativeArcsExtractionState(
        season_analysis="",
        episode_narrative_analysis="",
        arcs_from_analysis=[],
        arcs_from_plot=[],
        episode_arcs=[],
        episode_arcs_with_ids=[],
        season_arcs={},  # Initialize as an empty dictionary
        existing_season_arcs=[],
        present_season_arcs=[],
        file_paths=file_paths,
        series=series,
        season=season,
        episode=episode
    )
    logger.info("Invoking the graph")
    result = graph.invoke(initial_state)
    logger.info("Graph execution completed")

    logger.info("Narrative arcs extraction process completed.")
    return result

