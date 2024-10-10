from typing import Dict, List, TypedDict
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import os
import json
from src.utils.llm_utils import clean_llm_json_response, get_llm, clean_llm_text_response
from src.utils.logger_utils import setup_logging  # Import the setup_logging function

# Set up logging
logger = setup_logging(__name__)  # Use the setup_logging function

# Define the state
class NarrativeArc(BaseModel):
    """Model representing a narrative arc."""
    Title: str = Field(..., description="The title of the narrative arc")
    Description: str = Field(..., description="A brief description of the narrative arc")
    Progression: List[str] = Field(default_factory=list, description="Key progression points of the narrative arc")
    Type: str = Field(..., description="Type of the arc: 'Episodic' or 'Seasonal'")
    Characters: List[str] = Field(default_factory=list, description="Characters involved in this arc")

class NarrativeState(TypedDict):
    """State representation for the narrative analysis process."""
    season_analysis: str
    episode_arcs: List[NarrativeArc]
    season_arcs: List[NarrativeArc]
    file_paths: Dict[str, str]
    existing_arcs: Dict[str, NarrativeArc]
    episode_narrative_analysis: str  # New field for episode narrative analysis


llm = get_llm("intelligent")

def read_file_content(file_path: str) -> str:
    """Read the content of a file."""
    with open(file_path, 'r') as file:
        return file.read()

def load_existing_arcs(file_path: str) -> Dict[str, NarrativeArc]:
    """Load existing narrative arcs from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return {arc['Title']: NarrativeArc(**arc) for arc in data}
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}. Returning empty existing arcs.")
        return {}

def update_narrative_arcs(existing_arcs: Dict[str, NarrativeArc], new_arcs: List[NarrativeArc]) -> Dict[str, NarrativeArc]:
    """Update existing narrative arcs with new information."""
    for new_arc in new_arcs:
        if new_arc.Title in existing_arcs:
            existing_arc = existing_arcs[new_arc.Title]
            existing_arc.Description = new_arc.Description
            existing_arc.Progression.extend(new_arc.Progression)
            existing_arc.Characters = list(set(existing_arc.Characters + new_arc.Characters))
            logger.info(f"Updated existing arc: {new_arc.Title}")
        else:
            existing_arcs[new_arc.Title] = new_arc
            logger.info(f"Added new arc: {new_arc.Title}")
    return existing_arcs

# Define the nodes
def seasonal_narrative_analysis(state: NarrativeState) -> NarrativeState:
    """Analyze the season plot to identify overarching themes and character development."""
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
        Focus on the specific narrative elements that make this season unique and impactful.
        """
    )
    season_analysis = llm.invoke(prompt.format_messages(season_plot=season_plot))
    logger.info("Season narrative analysis completed.")
    state['season_analysis'] = season_analysis.content

    # Save the seasonal narrative analysis output to a file
    with open(state['file_paths']['seasonal_narrative_analysis_output_path'], 'w') as f:
        f.write(state['season_analysis'])
    logger.info("Seasonal narrative analysis output saved.")

    return state

def episode_narrative_arc_extraction(state: NarrativeState) -> NarrativeState:
    """Analyze the episode plot to identify narrative arcs based on the season analysis."""
    logger.info("Starting episode narrative analysis.")
    
    # Check if the episode analysis output file already exists
    if os.path.exists(state['file_paths']['episode_narrative_arcs_path']):
        logger.info("Episode narrative analysis output already exists. Loading from file.")
        with open(state['file_paths']['episode_narrative_arcs_path'], 'r') as f:
            state['episode_arcs'] = json.load(f)
        return state

    episode_plot = read_file_content(state['file_paths']['episode_plot_path'])
    logger.debug(f"Episode plot content: {episode_plot[:100]}...")  # Log the first 100 characters for brevity

    existing_arcs = state['existing_arcs'] if state['existing_arcs'] else {}
    logger.info(f"Existing arcs before analysis: {list(existing_arcs.keys())}")

    prompt = ChatPromptTemplate.from_template(
        """
        You are tasked with identifying and classifying the different types of narrative arcs present in a TV series episode. These arcs focus on the thematic development of stories, character relationships, and specific journeys tied to the genre of the series. Below are the main types of narrative arcs commonly found in TV shows.
        Types of Narrative Arcs:

            Soap Arc: Focuses on the development of a romantic relationship between characters or explores the dynamics and evolution of family relationships or friendships.
                Example: Jim and Pam's relationship in The Office or the family interactions in This Is Us.
                
            Genre-Specific Arc: This arc is specific to the genre of the show, such as specific professional challenges in a medical drama, family or political dynamics in a fantasy series, or survival in a horror series.
                Example: The power struggles between families in Game of Thrones, or medical challenges in Grey's Anatomy.

            Character Arc: Follows the personal growth or transformation of a character over time.
                Example: Walter White's transformation in Breaking Bad.

            Episodic Arc: A self-contained story that begins and ends within a single episode.
                Example: Standalone cases in Law & Order, medical cases in Grey's Anatomy, or main anthologic plots in Black Mirror.

            Mythology Arc: Focuses on the overarching lore or thematic elements that span across episodes or seasons.
                Example: The alien conspiracy in The X-Files or the Night Walkers plot in Game of Thrones.

        Task:

        For each storyline you identify in a TV series episode, assign it to one of these narrative arc types. Provide a brief explanation for your classification, focusing on how the storyline fits within the chosen narrative type.

        Analyze the following episode plot:

        [EPISODE PLOT]
        {episode_plot}
        [/EPISODE PLOT]
        
        Consider this season analysis for context:

        [SEASON ANALYSIS]
        {season_analysis}
        [/SEASON ANALYSIS]

        Identify specific narrative arcs for this episode:
        1. Focus on individual character storylines and their development.
        2. Identify both episodic arcs and ongoing seasonal arcs.
        3. Be specific about plot points and character motivations.
        4. Avoid broad or vague categories; each arc should be distinct and well-defined.
        5. The title should be specific. For example for a medical drama the episodic content should be "Medical Case: [insert case name here]", for a procedural it should be "Procedural Case: [insert case name here]" and so on.
        
        Return the narrative arcs in the following JSON format:
        [
            {{
                "Title": "Arc Title",
                "Description": "Brief description of the arc",
                "Progression": ["Key point 1", "Key point 2", ...],
                "Type": "Episodic/Seasonal/Potential Series-wide",
                "Characters": ["Character 1", "Character 2", ...]
            }},
            ... more arcs ...
        ]
        Ensure that your response is a valid JSON array containing only the narrative arcs, without any additional text.
        """
    )
    
    logger.info("Sending prompt to LLM for episode narrative analysis.")
    
    episode_analysis = llm.invoke(prompt.format_messages(
        episode_plot=episode_plot,
        season_analysis=state['season_analysis']
    ))
    
    logger.info("Episode narrative analysis completed.")
    
    try:
        cleaned_content = clean_llm_json_response(episode_analysis.content)
        logger.debug(f"Cleaned content: {cleaned_content[:500]}...")  # Log the first 500 characters

        parsed_arcs = json.loads(cleaned_content)
        
        if not isinstance(parsed_arcs, list):
            logger.error("Parsed content is not a list. Raw content: {}".format(cleaned_content))
            raise ValueError("Parsed content is not a list")

        new_arcs = []
        for arc in parsed_arcs:
            if not isinstance(arc, dict):
                logger.warning(f"Skipping invalid arc: {arc}")
                continue
            try:
                new_arc = NarrativeArc(**arc)
                new_arcs.append(new_arc)
            except ValueError as e:
                logger.warning(f"Error creating NarrativeArc: {e}. Skipping this arc: {arc}")

        logger.info(f"New arcs identified: {[arc.Title for arc in new_arcs]}")
        state['episode_arcs'] = new_arcs
        state['existing_arcs'] = update_narrative_arcs(state['existing_arcs'], new_arcs)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Raw response content: {episode_analysis.content}")
    except ValueError as e:
        logger.error(f"Invalid data structure: {e}")
        logger.error(f"Parsed content: {parsed_arcs}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Raw response content: {episode_analysis.content}")
    
    # Save the episode narrative analysis output to a file
    with open(state['file_paths']['episode_narrative_arcs_path'], 'w') as f:
        json.dump([arc.dict() for arc in state['episode_arcs']], f, indent=2)
    logger.info("Episode narrative analysis output saved.")

    return state

def verify_narrative_arcs(state: NarrativeState) -> NarrativeState:
    """Verify the identified narrative arcs for consistency and distinctness."""
    logger.info("Starting verification of narrative arcs.")
    prompt = ChatPromptTemplate.from_template(
        """As a Narrative Verification Specialist, your task is to verify the following narrative arcs:

        {episode_arcs}

        Ensure that:
        1. Each arc is distinct and doesn't overlap significantly with others.
        2. Arc titles are consistent with the broader story and not too episode-specific.
        3. Descriptions and progressions are clear and relevant to the arc.
        4. The categorization (Episodic/Seasonal/Potential Series-wide) is appropriate.

        If you find any issues, please correct them. Return the verified and potentially corrected arcs in the same JSON format.
        Ensure your response contains only the JSON array of narrative arcs, without any additional text or explanations.
        """
    )
    
    verified_arcs = llm.invoke(prompt.format_messages(episode_arcs=json.dumps([arc for arc in state['episode_arcs']])))
    
    try:
        cleaned_content = clean_llm_json_response(verified_arcs.content)
        parsed_arcs = json.loads(cleaned_content)
        state['episode_arcs'] = [NarrativeArc(**arc) for arc in parsed_arcs]
        logger.info("Narrative arcs verified successfully.")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse verified arcs: {e}")
        logger.error(f"Raw response content: {verified_arcs.content}")
        logger.warning("Keeping original arcs due to parsing error")
    
    return state

def manage_season_arcs(state: NarrativeState) -> NarrativeState:
    """Manage and update the season-wide narrative arcs based on episode analysis."""
    logger.info("Managing season arcs.")
    season_arcs = update_narrative_arcs(state['existing_arcs'], state['episode_arcs'])
    state['season_arcs'] = list(season_arcs.values())
    logger.info(f"Updated season arcs: {[arc.Title for arc in state['season_arcs']]}")
    return state

def episode_narrative_analysis(state: NarrativeState) -> NarrativeState:
    """Analyze the episode plot to provide a discoursive narrative analysis based on the season analysis."""
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
        1. Key themes and motifs present in the episode.
        2. Character development and interactions.
        3. Narrative techniques and stylistic choices.
        4. How this episode contributes to the overarching narrative of the season.

        Your analysis should be detailed, insightful, and avoid generic or vague statements.
        Focus on the specific narrative elements that make this episode unique and impactful.
        """
    )
    
    episode_analysis = llm.invoke(prompt.format_messages(
        episode_plot=episode_plot,
        season_analysis=state['season_analysis']
    ))
    
    logger.info("Episode narrative analysis completed.")
    state['episode_narrative_analysis'] = episode_analysis.content

    # Save the episode narrative analysis output to a file
    with open(state['file_paths']['episode_narrative_analysis_output_path'], 'w') as f:
        f.write(state['episode_narrative_analysis'])
    logger.info("Episode narrative analysis output saved.")

    return state

# Create the graph
def create_narrative_arc_graph():
    """Create and configure the state graph for narrative arc extraction."""
    workflow = StateGraph(NarrativeState)

    workflow.add_node("seasonal_analysis_node", seasonal_narrative_analysis)
    workflow.add_node("episode_analysis_node", episode_narrative_arc_extraction)
    workflow.add_node("episode_narrative_analysis_node", episode_narrative_analysis)  # New node
    workflow.add_node("verify_arcs_node", verify_narrative_arcs)
    workflow.add_node("manage_season_arcs_node", manage_season_arcs)

    workflow.set_entry_point("seasonal_analysis_node")
    workflow.add_edge("seasonal_analysis_node", "episode_analysis_node")
    workflow.add_edge("episode_analysis_node", "episode_narrative_analysis_node")  # New edge
    workflow.add_edge("episode_narrative_analysis_node", "verify_arcs_node")
    workflow.add_edge("verify_arcs_node", "manage_season_arcs_node")
    workflow.add_edge("manage_season_arcs_node", END)
    
    logger.info("Narrative arc graph workflow created successfully.")

    return workflow.compile()

# Function to run the graph
def extract_narrative_arcs(file_paths: Dict[str, str]) -> Dict[str, List[NarrativeArc]]:
    """Extract narrative arcs from the provided file paths and save the results."""
    existing_arcs = load_existing_arcs(file_paths.get("season_narrative_arcs_path", ""))
    graph = create_narrative_arc_graph()
    initial_state = NarrativeState(
        season_analysis="",
        episode_arcs=[],
        season_arcs=[],
        file_paths=file_paths,
        existing_arcs=existing_arcs,
        episode_narrative_analysis="" 
    )
    result = graph.invoke(initial_state)

    # Save the updated narrative arcs
    with open(file_paths["season_narrative_arcs_path"], 'w') as f:
        json.dump([arc.dict() for arc in result['season_arcs']], f, indent=2)

    # Save the seasonal narrative analysis output
    with open(file_paths['seasonal_narrative_analysis_output_path'], 'w') as f:
        f.write(result['season_analysis'])

    # Save the episode narrative analysis
    with open(file_paths['episode_narrative_arcs_path'], 'w') as f:
        json.dump([arc.dict() for arc in result['episode_arcs']], f, indent=2)

    # Save the episode narrative analysis
    with open(file_paths['episode_narrative_analysis_output_path'], 'w') as f:
        f.write(result['episode_narrative_analysis'])  # Ensure this is included

    return result