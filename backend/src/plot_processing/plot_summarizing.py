from typing import List, Dict
from textwrap import dedent
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from ..utils.logger_utils import setup_logging
from ..utils.text_utils import load_text
from ..utils.llm_utils import clean_llm_text_response
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
    prompt = dedent(f"""You are an expert at summarizing TV series plots as clear, narrative prose.

Create a plot summary that reads like a coherent story narrative.

CRITICAL REQUIREMENTS:
- Always use explicit character names (e.g., "Dr. Smith", "Detective Johnson") instead of pronouns or assumed context
- Write as a flowing narrative plot, NOT as a list or analysis with headers/sections
- Focus purely on what happens in the story, in chronological order
- Include character full names when first mentioned, then use their common names consistently
- Capture the main plot events, character interactions, and story developments
- Write in past tense as if recounting the story
- Do NOT include meta-commentary, analysis sections, or structural headers
- Make the summary self-contained - someone should understand the story without prior knowledge
- IMPORTANT: Only use character names that are explicitly mentioned in the source text. Do not invent or assume character names that are not present in the original material.

The summary should read like: "In this story, [Character Name] does X while [Character Name] faces Y. When Z happens, [Character Name] must decide..." etc.

Text to Summarize:
{text}

Write a clear plot narrative summary:""")
    
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
            
            prompt = dedent(f"""You are an expert at creating cumulative narrative summaries for TV series as flowing plot narratives.

You will receive TWO types of content:

1. PREVIOUSLY SUMMARIZED CONTENT: Already condensed material from past episodes
   - Preserve this content with minimal changes
   - Maintain the existing narrative flow

2. NEW EPISODE PLOT: Fresh, detailed content from the latest episode  
   - Summarize this into clear narrative prose
   - Focus on key story events and character actions

CRITICAL REQUIREMENTS:
- Always use explicit character names (e.g., "Dr. Smith", "Detective Johnson") instead of pronouns
- Write as a single, flowing narrative plot - NOT as sections, lists, or analysis
- Combine both parts into one coherent story progression
- Use past tense throughout as if recounting the story
- Include character full names when first mentioned, then use consistent common names
- Focus purely on plot events and character actions in chronological order
- Do NOT include headers, sections, meta-commentary, or analysis like "Episode Summary:", "Continuity Context:", etc.
- Make the summary self-contained and story-focused
- IMPORTANT: Only use character names that are explicitly mentioned in the source text. Do not invent or assume character names that are not present in the original material.

The result should read like: "In previous episodes, [Character Name] did X. When Y happened, [Character Name] responded by Z. In the latest episode, [Character Name] faces A while [Character Name] deals with B..." etc.

PREVIOUSLY SUMMARIZED CONTENT (preserve with minimal changes):
{previous_summary}

NEW EPISODE DETAILED PLOT (summarize and integrate):
{new_episode_plot}

Create a single, flowing narrative summary that combines both sections as a coherent story:""")
            
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
        
        prompt = dedent(f"""You are an expert at summarizing TV episode plots as clear, narrative prose.

Create a plot summary of this episode that reads like a coherent story narrative.

CRITICAL REQUIREMENTS:
- Always use explicit character names instead of pronouns or assumed context
- Write as a flowing narrative plot, NOT as a list or analysis with headers/sections
- Focus purely on what happens in the story, in chronological order
- Include character full names when first mentioned, then use their common names consistently
- Capture the main plot events, character interactions, and story developments
- Write in past tense as if recounting the story
- Do NOT include meta-commentary, analysis sections, or structural headers like "Episode Summary:", "Continuity Context:", etc.
- Make the summary self-contained - someone should understand the story without prior knowledge
- IMPORTANT: Only use character names that are explicitly mentioned in the source text. Do not invent or assume character names that are not present in the original material.


Episode Plot to Summarize:
{episode_plot}

Write a clear plot narrative summary:""")
        
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

def create_or_update_season_summary(
    episode_plot_path: str,
    season_summary_path: str, 
    episode_summary_path: str,
    llm: AzureChatOpenAI
) -> str:
    """
    Create or update season summary after processing an episode.
    
    This function:
    1. Creates an episode summary from the detailed plot (recap content already filtered)
    2. Updates the cumulative season summary with the new episode
    
    Args:
        episode_plot_path (str): Path to the detailed episode plot file (should be recap-filtered)
        season_summary_path (str): Path to the cumulative season summary file
        episode_summary_path (str): Path where episode summary will be saved
        llm (AzureChatOpenAI): The LLM to use for summarization
        
    Returns:
        str: The updated season summary
    """
    try:
        logger.info("📖 Creating/updating season summary")
        
        # Verify that we're using a recap-filtered plot file
        if "_plot.txt" in episode_plot_path or "possible_speakers" in episode_plot_path:
            logger.info("✅ Using recap-filtered plot content for season summary")
        else:
            logger.warning("⚠️ Episode plot path may contain recap content - consider using filtered version")
        
        # Step 1: Create episode summary
        episode_summary = create_episode_summary(episode_plot_path, llm, episode_summary_path)
        
        if not episode_summary:
            logger.error("Failed to create episode summary")
            return ""
            
        # Step 2: Update or create season summary
        if os.path.exists(season_summary_path):
            existing_summary = load_text(season_summary_path)
            
            if existing_summary.strip():
                # Create cumulative summary with previous context + new episode
                from backend.src.utils.llm_utils import clean_llm_text_response
                from textwrap import dedent
                from langchain_core.messages import HumanMessage
                
                prompt = dedent(f"""You are an expert at creating cumulative narrative summaries for TV series as flowing plot narratives.

You will receive TWO types of content:

1. PREVIOUSLY SUMMARIZED CONTENT: Already condensed material from past episodes
   - Preserve this content with minimal changes
   - Maintain the existing narrative flow

2. NEW EPISODE SUMMARY: Fresh, summarized content from the latest episode
   - Integrate this seamlessly with existing content
   - Maintain chronological flow and narrative continuity

CRITICAL REQUIREMENTS:
- Always use explicit character names instead of pronouns
- Write as a single, flowing narrative plot - NOT as sections, lists, or analysis  
- Combine both parts into one coherent story progression
- Use past tense throughout as if recounting the story
- Include character full names when first mentioned, then use consistent common names
- Focus purely on plot events and character actions in chronological order
- Do NOT include headers, sections, meta-commentary, or analysis like "Episode Summary:", "Continuity Context:", etc.
- Make the summary self-contained and story-focused

PREVIOUSLY SUMMARIZED CONTENT (preserve with minimal changes):
{existing_summary}

NEW EPISODE SUMMARY (integrate seamlessly):
{episode_summary}

Create a single, flowing narrative summary that combines both sections as a coherent story:""")
                
                response = llm.invoke([HumanMessage(content=prompt)])
                cumulative_summary = clean_llm_text_response(response.content.strip())
                logger.info("Updated cumulative season summary with new episode")
            else:
                # Empty existing file, just use new episode summary
                cumulative_summary = episode_summary
                logger.info("Created new season summary from episode summary")
        else:
            # No existing summary, use episode summary as the initial season summary
            cumulative_summary = episode_summary
            logger.info("Created initial season summary from first episode")
        
        # Step 3: Save the updated season summary
        os.makedirs(os.path.dirname(season_summary_path), exist_ok=True)
        with open(season_summary_path, "w", encoding="utf-8") as output_file:
            output_file.write(cumulative_summary)
        
        logger.info(f"Saved updated season summary to: {season_summary_path}")
        return cumulative_summary
        
    except Exception as e:
        logger.error(f"Error creating/updating season summary: {e}")
        return ""
