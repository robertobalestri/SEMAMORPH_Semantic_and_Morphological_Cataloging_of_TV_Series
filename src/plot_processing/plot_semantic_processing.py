from src.utils.logger_utils import setup_logging
from typing import List
from langchain_openai import AzureChatOpenAI
import networkx as nx
from src.plot_processing.plot_processing_models import EntityLink
import re
import json
from langchain_core.messages import HumanMessage
import os
from src.utils.text_utils import split_into_sentences, clean_text
from src.utils.llm_utils import clean_llm_text_response
from textwrap import dedent

# Set up colored logging
logger = setup_logging(__name__)

'''def guess_locations(sentences: List[str], llm: AzureChatOpenAI, window_size: int = 20, overlap: int = 5, plot_localized_sentences_path: str = "") -> List[str]:
    
    if os.path.exists(plot_localized_sentences_path):
        with open(plot_localized_sentences_path, "r", encoding='utf-8') as plot_localized_sentences_file:
            locations = json.load(plot_localized_sentences_file)
            return locations
    else:
        locations = []
    
    total_sentences = len(sentences)
    context = []
     
    for i in range(0, total_sentences, window_size - overlap):
        window_end = min(i + window_size, total_sentences)
        window_sentences = sentences[i:window_end]
        
        # Use only the last 'overlap' number of localized sentences as context
        context_sentences = locations[-overlap:] if len(locations) >= overlap else locations
        
       
        
        if len(locations) > 0:
            # Remove the already used locations from the sentences to be analyzed
            sentences_to_analyze = window_sentences[overlap:]
            context = '\n'.join(sentence for sentence in context_sentences)  # Join the context sentences with newlines
        else:
            sentences_to_analyze = window_sentences
        
        sentences_to_analyze_text = clean_llm_text_response('\n'.join(sentences_to_analyze))

        prompt = dedent(f"""
        Analyze the following sentences and guess the location for each, speculating based on the context of the sentences.
        Insert a [LOC: n] tag at the beginning of each sentence, where n is a numbered identifier for the location change.
        If the location changes, use a new numbered identifier. Identifiers have a validity, once used for a location change, it should not be reused.
        Depending on the context, the location can be a room or a larger area like a square or a building. Since we are doing these for a plot, try to understand if visually the location can be different.
        If it is said that a character is leaving a location towards another location, the [LOC] tag is still the one from which the character is leaving.
        
        For example:
        [LOC: 1] Bob orders the food.
        [LOC: 1] The waiter breaks a glass.
        [LOC: 2] Laura arrives at the hospital.
        [LOC: 2] Laura meets John.
        [LOC: 3] Bob is in the kitchen talking to John.
        [LOC: 4] Rebecca is in the living room cutting vegetables.
        [LOC: 4] Rebecca exits the living room towards the kitchen.
        [LOC: 5] Rebecca reaches Bob and John in the kitchen.

        Context from previous sentences (if any):
        {context if context else 'No context'}

        Sentences to analyze:
        {sentences_to_analyze_text}

        Please provide the sentences with [LOC: n] tags inserted at the beginning of each:
        """)

        logger.info(f"Sending prompt to LLM for location guessing:\n{prompt}")

        # LLM Interaction with error handling
        for attempt in range(3):  # Retry up to 3 times
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                logger.info(f"Received response from LLM for location guessing:\n{response.content}")
                
                # Parse and validate the localized sentences
                localized_sentences = []
                localized_sentences = split_into_sentences((response.content.strip()))
                
                for localized_sentence in localized_sentences:
                    locations.append(localized_sentence)

                break  # Exit the retry loop if successful
            except Exception as e:
                logger.error(f"Error during LLM call for location guessing (attempt {attempt + 1}): {e}")
                if attempt == 2:  # Last attempt
                    raise  # Re-raise the exception after the last attempt

    with open(plot_localized_sentences_path, "w", encoding='utf-8') as plot_localized_sentences_file:
        json.dump(locations, plot_localized_sentences_file, indent=2, ensure_ascii=False)
    
    return locations'''

def semantic_split(text: str, llm: AzureChatOpenAI, window_size: int = 20, plot_localized_sentences_path: str = "") -> List[str]:
    logger.info("Starting semantic split with <BOS> markers, context from previous splits, and location guessing")

    # Split the text into sentences
    sentences = split_into_sentences(text)
    # Guess locations for the entire text
    '''sentences_with_locations = guess_locations(
                                sentences = sentences, 
                                llm = llm, 
                                plot_localized_sentences_path = plot_localized_sentences_path
                            )'''
    
    
    
    segments = []
    i = 0
    last_bos_text = ''
    
    sentences_with_locations = sentences
    
    total_sentences = len(sentences_with_locations)
    
    sentences_with_locations[0] = '<BOS> ' + sentences_with_locations[0]

    while i < total_sentences:
        # Construct the window text
        window_sentences = []
        if last_bos_text:
            window_sentences.append('<BOS> ' + last_bos_text)
        window_end = min(i + window_size, total_sentences)
        window_sentences.extend(sentences_with_locations[i:window_end])
        window_text = '\n'.join(window_sentences)

        logger.debug(f"Window start index: {i}, Window end index: {window_end}")
        logger.debug(f"Current window text:\n{window_text}")

        prompt = dedent(f"""
                    Analyze the following text and insert <BOS> (Begin of Scene) tags only where a new semantic scene begins.

                    Guidelines:

                        - **Scene shifts**: A new scene usually starts when there is a major change in time or location, or a change of narrative or thematicfocus during the same situation.
                        
                        - **Dialogue continuity**: Do not add a <BOS> tag if the scene remains the same during an ongoing conversation, unless there is a change in the focus of the discourse.

                        - **Minor transitions**: Small movements within the same setting (e.g., a character moving to another room) should not trigger a new scene tag unless the overall narrative shifts (e.g., new characters, locations, or significant plot developments).

                        - **Sentence integrity**: Only place <BOS> tags at the beginning of sentences that introduce a new scene. Do not break sentences unnecessarily.

                        - **Emotional focus**: A typical scene revolves around one predominant emotion. 
                        
                        - **Event and Action focus**: Only one event or action usually happens per scene.

                        - **Thematic shifts**: Continuous narration can consist of multiple scenes if the thematic focus changes. For example, an encounter between two characters can transition from the meeting itself to a discussion about political matters or a recent natural disaster.
                        
                        - **Time and Place words**: The plot might contain words similar to 'Afterward', 'In another place', 'The next day', 'The following week', or other words indicating where and when the action is happening. These are indicators that a scene might change.
                        
                        - **Difference between anticipation and resolution**: The anticipation of an event can be in one scene, and the resolution in another. For example, a chracter might indicates that he's going to do something or order others to do something, but the action takes time and so the actual resolution happens after some time in a later scene.
                        
                    Text to analyze:

                    {window_text}

                    Please return the text with <BOS> tags inserted only where significant narrative shifts occur.
                    """)

        logger.info(f"Sending prompt to LLM for segment analysis:\n{prompt}")

        # LLM Interaction with error handling
        for attempt in range(3):  # Retry up to 3 times
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                logger.info(f"Received response from LLM:\n{response.content}")
                break  # Exit the retry loop if successful
            except Exception as e:
                logger.error(f"Error during LLM call (attempt {attempt + 1}): {e}")
                if attempt == 2:  # Last attempt
                    raise  # Re-raise the exception after the last attempt

        marked_text = clean_llm_text_response(response.content.strip())

        # Split the marked_text into segments at <BOS>
        segments_in_response = re.split(r'<BOS>', marked_text)
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