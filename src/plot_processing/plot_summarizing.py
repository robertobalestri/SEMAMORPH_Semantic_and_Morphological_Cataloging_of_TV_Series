from typing import List, Dict
from textwrap import dedent
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from src.utils.logger_utils import setup_logging
from src.utils.text_utils import load_text
from src.utils.llm_utils import clean_llm_text_response
import os

logger = setup_logging(__name__)

def summarize_plot(text: str, llm: AzureChatOpenAI, output_path: str) -> str:
    """
    Summarize the plot of a series using the LLM.
    
    Args:
        text (str): The text to summarize
        llm (AzureChatOpenAI): The LLM to use
        output_path (str): Where to save the summary
        
    Returns:
        str: The summarized text
    """
    prompt = dedent(f"""You are an expert at summarizing the main narrative arcs and plot progression of TV series.
    You will receive a plot description for a TV series and need to summarize it, focusing on the key narrative arcs and progression, while preserving the storyline without extraneous details.
    Your summary should be linear, capturing the primary events and shifts in the plot in a detailed, chronological manner without skipping around the text or adding commentary.
    Avoid conclusions or personal interpretation.
    Please summarize the following text:\n{text}""")
    
    response = llm.invoke([HumanMessage(content=prompt)])
    summary = clean_llm_text_response(response.content.strip())
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as output_file:
        output_file.write(summary)
    
    return summary

def create_season_summary(episode_summarized_plots_paths: List[str], llm: AzureChatOpenAI, season_summary_path: str) -> str:
    """
    Create a cumulative season summary by combining previous episode summaries with new episode content.
    
    This function implements a sequential summary system where:
    - Previously summarized content is preserved (not heavily reduced)
    - New episode content is appropriately summarized and integrated
    - Each episode builds upon the previous cumulative summary
    
    Args:
        episode_summarized_plots_paths (List[str]): List of paths to episode plot files
        llm (AzureChatOpenAI): The LLM to use for summarization
        season_summary_path (str): Path where the cumulative summary will be saved
        
    Returns:
        str: The cumulative season summary
    """
    try:
        if not episode_summarized_plots_paths:
            logger.warning("No episode plots provided for season summary")
            return ""
        
        # Load all episode plots
        episode_plots = []
        for plot_path in episode_summarized_plots_paths:
            if os.path.exists(plot_path):
                plot_content = load_text(plot_path)
                episode_plots.append(plot_content)
                logger.info(f"Loaded episode plot from: {plot_path}")
            else:
                logger.warning(f"Episode plot file not found: {plot_path}")
        
        if not episode_plots:
            logger.error("No valid episode plots found")
            return ""
        
        # For the first episode, just use its content as the initial summary
        if len(episode_plots) == 1:
            cumulative_summary = episode_plots[0]
            logger.info("Created initial season summary from first episode")
        else:
            # For multiple episodes, create cumulative summary
            # Previous episodes (already summarized) + new episode (full detail)
            previous_summary = "\n\n".join(episode_plots[:-1])  # All but the last episode
            new_episode_plot = episode_plots[-1]  # Latest episode
            
            prompt = dedent(f"""You are an expert at creating cumulative narrative summaries for TV series.

You will receive TWO types of content that require different treatment:

1. PREVIOUSLY SUMMARIZED CONTENT: Already condensed material from past episodes
   - Do NOT heavily cut or reduce this content further
   - Preserve the existing narrative flow and key details
   - This content should be maintained with minimal changes

2. NEW EPISODE DETAILED PLOT: Fresh, full-detail content from the latest episode
   - This should be appropriately summarized and integrated
   - Focus on key narrative arcs, character development, and plot progression
   - Maintain chronological flow

Your task is to create a cumulative summary that:
- Preserves the narrative continuity from previous episodes
- Integrates the new episode content seamlessly
- Maintains chronological order
- Provides context for future episode generation without replacing detailed plot generation
- Avoids over-summarization of already condensed content

PREVIOUSLY SUMMARIZED CONTENT (preserve with minimal changes):
{previous_summary}

NEW EPISODE DETAILED PLOT (summarize and integrate):
{new_episode_plot}

Create a cumulative summary that combines both sections while respecting their different treatment requirements:""")
            
            response = llm.invoke([HumanMessage(content=prompt)])
            cumulative_summary = clean_llm_text_response(response.content.strip())
            logger.info(f"Created cumulative season summary combining {len(episode_plots)-1} previous episodes with 1 new episode")
        
        # Save the cumulative summary
        os.makedirs(os.path.dirname(season_summary_path), exist_ok=True)
        with open(season_summary_path, "w", encoding="utf-8") as output_file:
            output_file.write(cumulative_summary)
        
        logger.info(f"Saved cumulative season summary to: {season_summary_path}")
        return cumulative_summary
        
    except Exception as e:
        logger.error(f"Error creating season summary: {e}")
        return ""

def create_episode_summary(episode_plot_path: str, llm: AzureChatOpenAI, episode_summary_path: str) -> str:
    """
    Create a summary of a single episode's plot for use in cumulative season summaries.
    
    This function creates a condensed version of an episode's plot that will be used
    as input for subsequent cumulative summaries. The summary focuses on:
    - Key narrative arcs and plot progression
    - Character development and relationships
    - Important story beats and turning points
    - Setting up context for future episodes
    
    Args:
        episode_plot_path (str): Path to the detailed episode plot file
        llm (AzureChatOpenAI): The LLM to use for summarization
        episode_summary_path (str): Path where the episode summary will be saved
        
    Returns:
        str: The episode summary
    """
    try:
        if not os.path.exists(episode_plot_path):
            logger.error(f"Episode plot file not found: {episode_plot_path}")
            return ""
        
        # Load the episode plot
        episode_plot = load_text(episode_plot_path)
        
        if not episode_plot.strip():
            logger.warning(f"Episode plot file is empty: {episode_plot_path}")
            return ""
        
        prompt = dedent(f"""You are an expert at summarizing TV episode plots for narrative continuity purposes.

Create a concise but comprehensive summary of this episode that will serve as context for future episode plot generation.

Your summary should:
- Capture the main narrative arcs and plot progression
- Highlight key character developments and relationship changes
- Include important story beats and turning points
- Maintain chronological flow of events
- Preserve essential details that provide context for future episodes
- Be detailed enough to understand story continuity but concise enough for cumulative use

Focus on narrative elements that will help understand:
- Character motivations and growth
- Ongoing storylines and conflicts
- Relationship dynamics
- World-building and setting details
- Plot threads that may continue in future episodes

Episode Plot to Summarize:
{episode_plot}

Create a detailed but concise episode summary:""")
        
        response = llm.invoke([HumanMessage(content=prompt)])
        episode_summary = clean_llm_text_response(response.content.strip())
        
        # Save the episode summary
        os.makedirs(os.path.dirname(episode_summary_path), exist_ok=True)
        with open(episode_summary_path, "w", encoding="utf-8") as output_file:
            output_file.write(episode_summary)
        
        logger.info(f"Created episode summary and saved to: {episode_summary_path}")
        return episode_summary
        
    except Exception as e:
        logger.error(f"Error creating episode summary: {e}")
        return ""
