# narrative_arc_graph.py

from typing import Dict, List, TypedDict, Tuple
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import os
import json
try:
    from src.utils.llm_utils import clean_llm_json_response
    from src.utils.logger_utils import setup_logging
    from src.vectorstore.vectorstore_collection import CollectionType, get_vectorstore_collection
    from src.ai_models.ai_models import get_llm, LLMType
    from src.narrative_classes.narrative_classes import NarrativeArc, ArcProgression
    # Set up logging
    logger = setup_logging(__name__)
except:
    from ..utils.llm_utils import clean_llm_json_response
    from ..utils.logger_utils import setup_logging
    from ..vectorstore.vectorstore_collection import CollectionType, get_vectorstore_collection
    from ..ai_models.ai_models import get_llm, LLMType
    from ..narrative_classes.narrative_classes import NarrativeArc, ArcProgression

    # Set up logging
    logger = setup_logging(__name__)
    pass



class NarrativeArcsExtractionState(TypedDict):
    """State representation for the narrative analysis process."""
    season_analysis: str
    episode_arcs: List[NarrativeArc]
    episode_arcs_with_ids: List[Tuple[NarrativeArc, str]]  # Arcs with their IDs
    season_arcs: List[Dict]  # Updated season arcs with IDs
    file_paths: Dict[str, str]
    episode_narrative_analysis: str
    series: str
    season: str
    episode: str

llm = get_llm(LLMType.CHEAP)

def read_file_content(file_path: str) -> str:
    """Read the content of a file.

    Args:
        file_path (str): The path to the file to be read.

    Returns:
        str: The content of the file.
    """
    with open(file_path, 'r') as file:
        return file.read()

def load_existing_season_arcs(file_path: str) -> List[Dict]:
    """Load existing season arcs from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing season arcs.

    Returns:
        List[Dict]: A list of existing season arcs.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}. Returning empty season arcs.")
        return []

def seasonal_narrative_analysis(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Analyze the season plot to identify overarching themes and character development.

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

    season_plot = read_file_content(state['file_paths']['season_plot_path'])
    logger.debug(f"Season plot content: {season_plot[:100]}...")  # Log the first 100 characters for brevity

    prompt = ChatPromptTemplate.from_template(
        """As a Seasonal Narrative Analyzer, your task is to analyze the following season plot:

        {season_plot}

        Provide a comprehensive analysis and critical commentary focusing on:
        1. Major themes of the season and how they evolve.
        2. Key character arcs and their development throughout the season.
        3. Overarching narrative arcs that span multiple episodes.
        4. The interconnectedness of themes and character arcs.

        Your analysis should be detailed, insightful, and avoid generic or vague statements.
        """
    )
    season_analysis = llm.invoke(prompt.format_messages(season_plot=season_plot))
    logger.info("Season narrative analysis completed.")
    state['season_analysis'] = str(season_analysis.content)

    # Save the seasonal narrative analysis output to a file
    with open(state['file_paths']['seasonal_narrative_analysis_output_path'], 'w') as f:
        f.write(state['season_analysis'])
    logger.info("Seasonal narrative analysis output saved.")

    return state

def episode_narrative_analysis(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Analyze the episode plot to provide a discoursive narrative analysis based on the season analysis.

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

    episode_plot = read_file_content(state['file_paths']['episode_plot_path'])
    logger.debug(f"Episode plot content: {episode_plot[:100]}...")  # Log the first 100 characters for brevity

    prompt = ChatPromptTemplate.from_template(
        """As an Episode Narrative Analyzer, your task is to analyze the following episode plot in the context of the overall season analysis:
        
        Episode Plot:
        {episode_plot}

        Season Analysis:
        {season_analysis}

        Provide a comprehensive narrative analysis focusing on:
        1. How this episode and its narration contributes to the overarching narrative of the season.
        2. Key themes and motifs present in the episode.
        3. Character development and interactions.
        4. Narrative techniques and stylistic choices.
        
        Don't lose yourself in generic statements, be concise and specific. Avoid generic or vague phrases.
        Your analysis should be detailed and insightful. The analysis should be concise and not too long.
        """
    )
    
    episode_analysis = llm.invoke(prompt.format_messages(
        episode_plot=episode_plot,
        season_analysis=state['season_analysis'],
    ))
    
    logger.info("Episode narrative analysis completed.")
    state['episode_narrative_analysis'] = str(episode_analysis.content)

    # Save the episode narrative analysis output to a file
    with open(state['file_paths']['episode_narrative_analysis_output_path'], 'w') as f:
        f.write(state['episode_narrative_analysis'])
    logger.info("Episode narrative analysis output saved.")

    return state

# Add this near the top of the file, after the imports

NARRATIVE_ARC_GUIDELINES = """
Guidelines for Narrative Arc Extraction:

1. Arc Types:
   - Soap Arc: Focuses on romantic relationships, family dynamics, or friendships.
   - Genre-Specific Arc: Relates to the show's genre (e.g., medical challenges, political intrigues).
   - Episodic Arc: Self-contained story within a single episode.

2. Title Creation:
   - Be specific and descriptive.
   - Include main entities involved (e.g., "The Trial of Benedict Arnold", "Walter White and Skyler's Relationship").
   - For episodic content, use format: "[Genre] Case: [Specific Case Name]" (e.g., "Medical Case: Rare Genetic Disorder", "Procedural Case: Presidential Assassination").
   - Especially for an episodic arc, the title should be descriptive and not vague like "Character X's difficulties" or "Character X's problems" or "Character X's professional growth" etc. etc.

   3. Description:
   - Provide an overall summary of the arc's content season-wide.
   - Unless the arc is episodic, avoid focusing on the arc's development within the specific episode.

4. Episodic Flag:
   - Set to True for self-contained, anthology-like plots.
   - Set to False for arcs spanning multiple episodes.

5. Character List:
   - Include all relevant characters involved in the arc.

6. Distinctness:
   - Each arc should be well-defined and distinct from others.
   - Avoid overlap between arcs.
   - Sometimes there arcs that seems indipendent in the episode, but they are actually part of a season's overarching plot.

7. Progression:
   - List key points in the arc's development.
   - Focus on major events or turning points.
"""

def episode_narrative_arc_extraction(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Analyze the episode plot to identify narrative arcs based on the season analysis."""
    logger.info("Starting episode narrative arc extraction.")
    
    # Check if the episode arcs output file already exists
    if os.path.exists(state['file_paths']['episode_narrative_arcs_path']):
        logger.info("Episode narrative arcs output already exists. Loading from file.")
        with open(state['file_paths']['episode_narrative_arcs_path'], 'r') as f:
            arcs_data = json.load(f)
            state['episode_arcs'] = [NarrativeArc(**arc) for arc in arcs_data]
        return state

    episode_plot = read_file_content(state['file_paths']['episode_plot_path'])
    logger.debug(f"Episode plot content: {episode_plot[:100]}...")

    # Load existing season arcs
    existing_season_arcs = load_existing_season_arcs(state['file_paths']['season_narrative_arcs_path'])
    
    # Extract only title and description from existing season arcs
    season_arc_summaries = [
        {"title": arc['title'], "description": arc['description']}
        for arc in existing_season_arcs
    ]

    prompt = ChatPromptTemplate.from_template(
        """
        You are tasked with identifying and classifying the different types of narrative arcs present in a TV series episode. These arcs focus on the thematic development of stories, character relationships, and specific journeys tied to the genre of the series.

        Please follow these guidelines when identifying and describing narrative arcs:

        {guidelines}

        Analyze the following episode plot:

        [EPISODE PLOT]
        {episode_plot}
        [/EPISODE PLOT]

        Consider these existing (if any) season-wide narrative arcs:
        {season_arc_summaries}

        Important instructions for handling season-wide arcs:
        1. If you identify an arc that continues from the season-wide arcs listed above, use the exact same title and description.
        2. For these continuing arcs, only add the progression specific to this episode.
        3. Do not create a new season arc if it's a continuation of an existing one.
        4. Set the "episodic" flag to False for these continuing season arcs.

        For new arcs specific to this episode:
        1. Create a new arc entry with a unique title and description.
        2. Set the "episodic" flag to True if it's contained within this episode, or False if you believe it will continue in future episodes.

        Identify specific narrative arcs for this episode following the provided guidelines.
        Usually there are five or more arcs in an episode, including both continuing season arcs and new episodic arcs.

        Return the narrative arcs in the following JSON format:
        [
            {{
                "title": "Specific Arc title",
                "arc_type": "Soap Arc/Genre-Specific Arc/Episodic Arc",
                "description": "Brief season-wide description of the arc",
                "progression": ["Key point 1", "Key point 2", ...],
                "episodic": "True/False",
                "characters": ["Character 1", "Character 2", ...]
            }},
            ... more arcs ...
        ]
        Ensure that your response is a valid JSON array containing only the narrative arcs, without any additional text.
        """
    )
    
    logger.info("Sending prompt to LLM for episode narrative arc extraction.")
    
    prompt_formatted = prompt.format_messages(
        episode_plot=episode_plot,
        episode_analysis=state['episode_narrative_analysis'],
        guidelines=NARRATIVE_ARC_GUIDELINES,
        season_arc_summaries=json.dumps(season_arc_summaries, indent=2)
    )
    
    logger.info(f"Prompt formatted: {prompt_formatted}")
    
    try:
        logger.info("Attempting to invoke LLM...")
        episode_analysis = llm.invoke(prompt_formatted)
        logger.info(f"LLM response received: {episode_analysis}")
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        raise
    
    logger.info("Episode narrative arc extraction completed.")
    
    logger.debug(f"Raw episode analysis content: {episode_analysis.content}")
    
    try:
        parsed_arcs = clean_llm_json_response(str(episode_analysis.content))
        new_arcs = []
        for arc in parsed_arcs:
            if not isinstance(arc, dict):
                logger.warning(f"Skipping invalid arc: {arc}")
                continue
            try:
                progression = ArcProgression(
                    content=" | ".join(arc.get('progression', [])),  # Join progression points
                    series=state['series'],
                    season=state['season'],
                    episode=state['episode']
                )
                new_arc = NarrativeArc(
                    title=arc['title'], 
                    arc_type=arc['arc_type'],
                    description=arc['description'],
                    episodic=arc['episodic'],
                    characters=arc['characters'],
                    series=state['series'],
                    progressions=[progression]
                )
                new_arcs.append(new_arc)
                logger.debug(f"Created new arc: {new_arc.title} with progression: {progression.content}")
            except ValueError as e:
                logger.warning(f"Error creating NarrativeArc: {e}. Skipping this arc: {arc}")

        logger.info(f"New arcs identified: {[arc.title for arc in new_arcs]}")
        state['episode_arcs'] = new_arcs

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Raw response content: {episode_analysis.content}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Raw response content: {episode_analysis.content}")

    return state

def identify_missing_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Identify and add any missing narrative arcs based on the episode plot."""
    logger.info("Starting identification of missing narrative arcs.")

    episode_plot = read_file_content(state['file_paths']['episode_plot_path'])
    
    # Convert NarrativeArc instances to dictionaries for JSON serialization
    episode_arcs_dicts = [arc.model_dump() for arc in state['episode_arcs']]
    
    prompt = ChatPromptTemplate.from_template(
        """As a Narrative Arc Identification Specialist, your task is to review the existing narrative arcs and identify any missing ones based on the episode plot.
        You can also modify the existing arcs if you think that some of them are not correctly classified or described and that can be improved.

        Please follow these guidelines when identifying and describing narrative arcs:

        {guidelines}

        Existing Arcs:
        {episode_arcs}

        Episode Plot:
        {episode_plot}

        Your task:
        1. Carefully review the existing arcs and the episode plot.
        2. Identify any significant storylines, character developments, or thematic elements that are not represented in the existing arcs.
        3. For each missing arc you identify, create a new arc entry following the same format as the existing ones and adhering to the provided guidelines.
        4. Ensure that new arcs are distinct from existing ones and provide meaningful additions to the narrative analysis. They should not overlap with existing arcs.
        5. Sometimes there arcs that seems indipendent in the episode, but they are actually part of a season's overarching plot. If you think that an arc is part of the overarching plot, add it as a season arc, updating its description, title and type.

        Return the complete list of arcs (existing + new) in the same JSON format.
        Ensure your response contains only the JSON array of narrative arcs, without any additional text or explanations.
        """
    )
    
    updated_arcs = llm.invoke(prompt.format_messages(
        episode_arcs=json.dumps(episode_arcs_dicts),
        episode_plot=episode_plot,
        guidelines=NARRATIVE_ARC_GUIDELINES
    ))
    
    try:
        cleaned_content = clean_llm_json_response(updated_arcs.content)
        
        if isinstance(cleaned_content, list):
            parsed_arcs = cleaned_content
        else:
            parsed_arcs = json.loads(cleaned_content)

        state['episode_arcs'] = [NarrativeArc(**arc) for arc in parsed_arcs]
        logger.info(f"Updated arcs list with {len(state['episode_arcs']) - len(episode_arcs_dicts)} new arcs.")
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse updated arcs: {e}")
        logger.error(f"Raw response content: {updated_arcs.content}")
        logger.warning("Keeping original arcs due to parsing error")
    
    return state

def verify_narrative_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Verify the identified narrative arcs for consistency and correctness."""
    logger.info("Starting verification of narrative arcs.")

    episode_plot = read_file_content(state['file_paths']['episode_plot_path'])
    season_analysis = state['season_analysis']
    
    # Convert NarrativeArc instances to dictionaries for JSON serialization
    episode_arcs_dicts = [arc.model_dump() for arc in state['episode_arcs']]
    
    prompt = ChatPromptTemplate.from_template(
        """As a Narrative Verification Specialist, your task is to verify the metadata of the following narrative arcs:

        {episode_arcs}

        Based on the plot of the episode:
        {episode_plot}

        Please follow these guidelines when verifying and correcting narrative arcs:

        {guidelines}

        Ensure that for each arc:
        1. The title, description, and other elements adhere to the provided guidelines. This is very important: titles should not be vague or overlapping.
        2. The arc_type classification is appropriate.
        3. The episodic flag correctly indicates whether the arc is self-contained or part of a larger narrative.
        4. The characters list is accurate and complete.

        Do not add new arcs or remove existing ones. Focus solely on verifying and correcting the information for each arc.

        Return the verified and corrected arcs in the same JSON format.
        Ensure your response contains only the JSON array of narrative arcs, without any additional text or explanations.
        """
    )
    
    verified_arcs = llm.invoke(prompt.format_messages(
        episode_arcs=json.dumps(episode_arcs_dicts),
        episode_plot=episode_plot,
        season_analysis=season_analysis,
        guidelines=NARRATIVE_ARC_GUIDELINES
    ))
    
    try:
        cleaned_content = clean_llm_json_response(verified_arcs.content)
        
        if isinstance(cleaned_content, list):
            parsed_arcs = cleaned_content
        else:
            parsed_arcs = json.loads(cleaned_content)

        state['episode_arcs'] = [NarrativeArc(**arc) for arc in parsed_arcs]
        logger.info("Narrative arcs verified successfully.")
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse verified arcs: {e}")
        logger.error(f"Raw response content: {verified_arcs.content}")
        logger.warning("Keeping original arcs due to parsing error")
    
    return state

def update_vectorstore(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Update the vector store with the verified narrative arcs and collect IDs."""
    logger.info("Updating vector store with verified narrative arcs.")
    vectorstore_collection = get_vectorstore_collection(collection_type=CollectionType.NARRATIVE_ARCS)
    arc_id_map = vectorstore_collection.add_or_update_narrative_arcs(
        state['episode_arcs'],
        state['series'],
        state['season'],
        state['episode']
    )
    state['episode_arcs_with_ids'] = arc_id_map
    logger.info("Vector store updated with verified narrative arcs.")
    return state

def manage_season_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Update the season arcs by adding new arcs and updating existing ones.

    Args:
        state (NarrativeArcsExtractionState): The current state of the narrative analysis.

    Returns:
        NarrativeArcsExtractionState: The state after updating the season arcs.
    """
    logger.info("Managing season arcs.")
    # Load existing season arcs
    existing_season_arcs = load_existing_season_arcs(state['file_paths']['season_narrative_arcs_path'])

    # Convert existing season arcs to a dictionary for easy lookup
    existing_arcs_dict = {arc['title']: arc for arc in existing_season_arcs}

    # Update or add arcs
    for arc, arc_id in state['episode_arcs_with_ids']:
        arc_data = arc.model_dump()
        arc_data['id'] = arc_id
        if arc.title in existing_arcs_dict:
            # Update existing arc
            existing_arc = existing_arcs_dict[arc.title]
            existing_arc['description'] = arc.description
            existing_arc['arc_type'] = arc.arc_type
            existing_arc['episodic'] = arc.episodic
            existing_arc['characters'] = list(set(existing_arc['characters'] + arc.characters))
            existing_arc['Progression'].extend(arc.progressions)
        else:
            # Add new arc
            existing_arcs_dict[arc.title] = arc_data

    # Update the season arcs in the state
    state['season_arcs'] = list(existing_arcs_dict.values())
    logger.info("Season arcs updated.")
    return state

def save_arcs_to_json(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Save the arcs with IDs to JSON files."""
    logger.info("Saving arcs with IDs to JSON files.")

    # Save the episode narrative arcs with IDs
    episode_arcs_with_ids = []
    for arc, arc_id in state['episode_arcs_with_ids']:
        arc_data = arc.dict()
        arc_data['id'] = arc_id
        arc_data['progressions'] = [prog.dict() for prog in arc.progressions]
        episode_arcs_with_ids.append(arc_data)

    with open(state['file_paths']['episode_narrative_arcs_path'], 'w') as f:
        json.dump(episode_arcs_with_ids, f, indent=2)
    logger.info("Episode narrative arcs with IDs saved.")

    # Save the updated season arcs
    with open(state['file_paths']['season_narrative_arcs_path'], 'w') as f:
        json.dump(state['season_arcs'], f, indent=2)
    logger.info("Season narrative arcs saved.")

    return state

# Create the graph
def create_narrative_arc_graph():
    """Create and configure the state graph for narrative arc extraction."""
    workflow = StateGraph(NarrativeArcsExtractionState)

    workflow.add_node("seasonal_analysis_node", seasonal_narrative_analysis)
    workflow.add_node("episode_narrative_analysis_node", episode_narrative_analysis)
    workflow.add_node("episode_narrative_arc_extraction_node", episode_narrative_arc_extraction)
    workflow.add_node("identify_missing_arcs_node", identify_missing_arcs)
    workflow.add_node("verify_arcs_node", verify_narrative_arcs)
    workflow.add_node("update_vectorstore_node", update_vectorstore)
    workflow.add_node("manage_season_arcs_node", manage_season_arcs)
    workflow.add_node("save_arcs_to_json_node", save_arcs_to_json)

    workflow.set_entry_point("seasonal_analysis_node")
    workflow.add_edge("seasonal_analysis_node", "episode_narrative_analysis_node")
    workflow.add_edge("episode_narrative_analysis_node", "episode_narrative_arc_extraction_node")
    workflow.add_edge("episode_narrative_arc_extraction_node", "identify_missing_arcs_node")
    workflow.add_edge("identify_missing_arcs_node", "verify_arcs_node")
    workflow.add_edge("verify_arcs_node", "update_vectorstore_node")
    workflow.add_edge("update_vectorstore_node", "manage_season_arcs_node")
    workflow.add_edge("manage_season_arcs_node", "save_arcs_to_json_node")
    workflow.add_edge("save_arcs_to_json_node", END)
    
    logger.info("Narrative arc graph workflow created successfully.")

    return workflow.compile()

# Function to run the graph
def extract_narrative_arcs(file_paths: Dict[str, str], series: str, season: str, episode: str) -> NarrativeArcsExtractionState:
    """Extract narrative arcs from the provided file paths and save the results.

    Args:
        file_paths (Dict[str, str]): A dictionary containing file paths for input and output.
        series (str): The series identifier.
        season (str): The season identifier.
        episode (str): The episode identifier.

    Returns:
        NarrativeArcsExtractionState: The result containing updated narrative arcs.
    """
    logger.info("Starting extract_narrative_arcs function")
    graph = create_narrative_arc_graph()
    initial_state = NarrativeArcsExtractionState(
        season_analysis="",
        episode_arcs=[],
        episode_arcs_with_ids=[],
        season_arcs=[],
        file_paths=file_paths,
        episode_narrative_analysis="",
        series=series,
        season=season,
        episode=episode
    )
    logger.info("Invoking the graph")
    result = graph.invoke(initial_state)
    logger.info("Graph execution completed")

    # Save the seasonal narrative analysis output
    with open(file_paths['seasonal_narrative_analysis_output_path'], 'w') as f:
        f.write(result['season_analysis'])

    # Save the episode narrative analysis
    with open(file_paths['episode_narrative_analysis_output_path'], 'w') as f:
        f.write(result['episode_narrative_analysis'])

    logger.info("Narrative arcs extraction process completed.")
    return result


