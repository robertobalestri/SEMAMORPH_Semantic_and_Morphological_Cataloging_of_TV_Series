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
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are tasked with identifying and classifying the different types of narrative arcs present in a TV series episode. These arcs focus on the thematic development of stories, character relationships, and specific journeys tied to the genre of the series. Below are the main types of narrative arcs commonly found in TV shows.
        Types of Narrative Arcs:

            Soap Arc: Focuses on the development of a romantic relationship between characters or explores the dynamics and evolution of family relationships or friendships.
                Example: Jim and Pam's relationship in The Office or the family interactions in This Is Us.
                
            Genre-Specific Arc: This arc is specific to the genre of the show, such as specific professional challenges in a medical drama, family or political dynamics in a fantasy series, survival in a horror series, professional work in a legal drama, etc.
                Example: The power struggles between families in Game of Thrones, or medical challenges in Grey's Anatomy.

            Episodic Arc: A self-contained story that begins and ends within a single episode. Often refers to the vertical storytelling of a single episode.
                Example: Standalone cases in Law & Order, medical cases in Grey's Anatomy, or main anthologic plots in Black Mirror.

        Task:

        For each storyline you identify in a TV series episode, assign it to one of these narrative arc types. Provide a brief explanation for your classification, focusing on how the storyline fits within the chosen narrative type.

        Analyze the following episode plot:

        [EPISODE PLOT]
        {episode_plot}
        [/EPISODE PLOT]
        
        Consider this season analysis for context:

        [SEASON ANALYSIS]
        {episode_analysis}
        [/SEASON ANALYSIS]

        identify specific narrative arcs for this episode:
        1. identify both episodic arcs and ongoing seasonal arcs, focusing also on character arcs and their development, relationship, genre-specific, episodic and mythology arcs.
        2. Be specific about plot points and motivations.
        3. Each arc should be distinct and well-defined. 
        4. The title should be specific. For example for a medical drama the episodic content should be "Medical Case: [insert case name here]", for a procedural it should be "Procedural Case: [insert case name here]" and so on.
        5. Always avoid vague and not defined arc titles such as "Character's personal growth" or "Character's relationships".
        6. The description should be an overall summary of the content of the arc season-wide, not about the development of the arc inside the specific episode.
        
        Return the narrative arcs in the following JSON format:
        [
            {{
                "title": "Specific Arc title",
                "arc_type": "Soap Arc/Genre-Specific Arc/Episodic Arc",
                "description": "Brief season-wide description of the arc",
                "progression": ["Key point 1", "Key point 2", ...],
                "duration": "Episodic/Seasonal",
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
                    duration=arc['duration'],
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

def verify_narrative_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Verify the identified narrative arcs for consistency and distinctness.

    Args:
        state (NarrativeArcsExtractionState): The current state of the narrative analysis.

    Returns:
        NarrativeArcsExtractionState: The updated state after verification.
    """
    logger.info("Starting verification of narrative arcs.")
    
    # Convert NarrativeArc instances to dictionaries for JSON serialization
    episode_arcs_dicts = [arc.model_dump() for arc in state['episode_arcs']]
    
    prompt = ChatPromptTemplate.from_template(
        """As a Narrative Verification Specialist, your task is to verify the following narrative arcs:

        {episode_arcs}

        Ensure that:
        1. Each arc is distinct and doesn't overlap significantly with others.
        2. Arc titles are consistent with the broader story and not too episode-specific, but should avoid vague titles such as "Character's personal growth" or "Character's relationships".
        3. descriptions and progressions are clear and relevant to the arc.
        4. The categorization (Episodic/Seasonal) is appropriate.

        If you find any issues, please correct them. Return the verified and potentially corrected arcs in the same JSON format.
        Ensure your response contains only the JSON array of narrative arcs, without any additional text or explanations.
        """
    )
    
    verified_arcs = llm.invoke(prompt.format_messages(episode_arcs=json.dumps(episode_arcs_dicts)))
    
    try:
        cleaned_content = clean_llm_json_response(verified_arcs.content)
        
        # Ensure cleaned_content is a list of dictionaries
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
            existing_arc['duration'] = arc.duration
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
    """Create and configure the state graph for narrative arc extraction.

    Returns:
        StateGraph: The compiled state graph for narrative arc extraction.
    """
    workflow = StateGraph(NarrativeArcsExtractionState)

    workflow.add_node("seasonal_analysis_node", seasonal_narrative_analysis)
    workflow.add_node("episode_narrative_analysis_node", episode_narrative_analysis)
    
    logger.info("Adding episode_narrative_arc_extraction_node to the graph")
    workflow.add_node("episode_narrative_arc_extraction_node", episode_narrative_arc_extraction)   
    
    workflow.add_node("verify_arcs_node", verify_narrative_arcs)
    workflow.add_node("update_vectorstore_node", update_vectorstore)
    workflow.add_node("manage_season_arcs_node", manage_season_arcs)
    workflow.add_node("save_arcs_to_json_node", save_arcs_to_json)

    workflow.set_entry_point("seasonal_analysis_node")
    workflow.add_edge("seasonal_analysis_node", "episode_narrative_analysis_node")
    workflow.add_edge("episode_narrative_analysis_node", "episode_narrative_arc_extraction_node")
    workflow.add_edge("episode_narrative_arc_extraction_node", "verify_arcs_node")
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