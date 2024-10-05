from src.utils.logger_utils import setup_logging
from typing import List
from langchain_openai import AzureChatOpenAI  # Updated import
import networkx as nx
from src.plot_processing.plot_processing_models import EntityLink
import re
import json
from langchain_core.messages import HumanMessage

from src.utils.text_utils import split_into_sentences, clean_text
from textwrap import dedent

# Set up colored logging
logger = setup_logging(__name__)


def semantic_split(text: str, llm: AzureChatOpenAI, window_size: int = 15) -> List[str]:
    logger.info("Starting semantic split with [BOS] markers and context from previous splits")

    # Split the text into sentences
    sentences = split_into_sentences(text)
    total_sentences = len(sentences)
    segments = []
    i = 0
    last_bos_text = ''

    while i < total_sentences:
        # Construct the window text
        window_sentences = []
        if last_bos_text:
            window_sentences.append('[BOS] ' + last_bos_text)
        window_end = min(i + window_size, total_sentences)
        window_sentences.extend(sentences[i:window_end])
        window_text = '\n'.join(window_sentences)

        logger.debug(f"Window start index: {i}, Window end index: {window_end}")
        logger.debug(f"Current window text:\n{window_text}")

        prompt = dedent(f"""...""")  # Keep the existing prompt

        logger.info(f"Sending prompt to LLM for segment analysis:\n{prompt}")

        # LLM Interaction with error handling
        for attempt in range(3):  # Retry up to 3 times
            try:
                response = llm.invoke([HumanMessage(content=prompt)])  # Corrected method
                logger.info(f"Received response from LLM:\n{response.content}")  # Corrected to use content
                break  # Exit the retry loop if successful
            except Exception as e:
                logger.error(f"Error during LLM call (attempt {attempt + 1}): {e}")
                if attempt == 2:  # Last attempt
                    raise  # Re-raise the exception after the last attempt

        marked_text = response.content.strip()  # Corrected to use content

        # Split the marked_text into segments at [BOS]
        segments_in_response = re.split(r'\[BOS\]', marked_text)
        # Remove any empty strings and strip whitespace
        segments_in_response = [seg.strip() for seg in segments_in_response if seg.strip()]

        # Append all segments except for the last one to segments
        for seg in segments_in_response[:-1]:
            segments.append(clean_text(seg))
            logger.debug(f"Created segment: {seg[:50]}...")

        # The last segment is stored for the next iteration
        if segments_in_response:
            last_bos_text = clean_text(segments_in_response[-1])
        else:
            last_bos_text = ''

        # Move i forward by window_size
        i = window_end
        logger.debug(f"Moving window forward. New start index: {i}")

    # After the loop, add any remaining text in last_bos_text to segments
    if last_bos_text:
        segments.append(last_bos_text)
        logger.debug(f"Added last segment: {last_bos_text[:50]}...")

    logger.info(f"Semantic split complete. Number of segments: {len(segments)}")
    logger.debug(f"Final segments: {[seg[:50] + '...' for seg in segments]}")

    return segments

def build_character_graph(entities: List[EntityLink], text: str) -> nx.Graph:
    """
    Build a character graph based on co-occurrence of characters in the text.

    Args:
        entities (List[EntityLink]): List of extracted entities.
        text (str): The text to analyze.

    Returns:
        nx.Graph: A graph representing character relationships.
    """
    G = nx.Graph()
    
    # Add nodes for each character
    for entity in entities:
        G.add_node(entity.character)
    
    # Add edges based on co-occurrence in sentences
    sentences = text.split('.')
    for sentence in sentences:
        characters_in_sentence = [entity.character for entity in entities if entity.character in sentence]
        for i in range(len(characters_in_sentence)):
            for j in range(i+1, len(characters_in_sentence)):
                G.add_edge(characters_in_sentence[i], characters_in_sentence[j])
    
    return G

def summarize_segments(segments: List[str], llm: AzureChatOpenAI) -> List[str]:
    """
    Summarize each segment into a concise sentence.

    Args:
        segments (List[str]): List of segments to summarize.
        llm (AzureOpenAI): The language model instance.

    Returns:
        List[str]: List of summarized segments.
    """
    logger.info("Starting segment summarization")
    summaries = []
    for i, segment in enumerate(segments):
        prompt = f"Summarize the following segment in one concise sentence:\n\n{segment}"
        logger.info(f"Sending prompt to LLM for segment summarization:\n{prompt}")
        response = llm.invoke(prompt)  # Updated method
        logger.info(f"Received response from LLM:\n{response.text}")
        summary = response.text.strip()
        summaries.append(summary)
        logger.info(f"Summarized segment {i+1}/{len(segments)}")
    
    logger.info("Segment summarization complete")
    return summaries

