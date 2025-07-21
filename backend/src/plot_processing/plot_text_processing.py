import re
from typing import List
from nltk.tokenize import sent_tokenize
from functools import lru_cache
from langchain_openai import AzureChatOpenAI  # Updated import
from ..utils.logger_utils import setup_logging
from ..utils.text_utils import split_into_sentences, remove_duplicates
from ..config import config
import json
from textwrap import dedent
from ..utils.llm_utils import clean_llm_text_response
from langchain_core.messages import HumanMessage

# Set up colored logging
logger = setup_logging(__name__)

def replace_pronouns_with_names(text: str, intelligent_llm: AzureChatOpenAI, cheap_llm: AzureChatOpenAI, chunk_size: int = None, context_size: int = None) -> str:
    """
    Replace pronouns and generic references in the text with specific character names.

    Args:
        text (str): The text to modify.
        intelligent_llm (AzureChatOpenAI): The intelligent language model instance.
        cheap_llm (AzureChatOpenAI): The cheap language model instance.
        chunk_size (int, optional): Number of sentences to process in each batch. If None, uses config value.
        context_size (int, optional): Number of sentences to provide as context. If None, uses config value.

    Returns:
        str: The modified text with pronouns replaced by names.
    """
    # Use config values if not provided
    if chunk_size is None:
        chunk_size = config.pronoun_replacement_batch_size
    if context_size is None:
        context_size = config.pronoun_replacement_context_size
    
    logger.info(f"Using batch size: {chunk_size}, context size: {context_size}")
    
    sentences = split_into_sentences(text)
    named_sentences = []

    for i in range(0, len(sentences), chunk_size):
        end_index = min(i + chunk_size, len(sentences))
        chunk = sentences[i:end_index]
        
        logger.info(f"Processing {len(chunk)} sentences (index {i} to {end_index-1}).")

        prompt = (
            "Rewrite the following text, replacing pronouns and generic references with specific character names. "
            "Follow these guidelines:\n"
            "1. Replace all pronouns (he, she, they, etc.) with the appropriate character name. Pay attention to the context to understand who the pronoun refers to, don't just replace the pronoun with the nearest names you find.\n"
            "2. Replace generic references like 'the woman', 'the boy', 'the old man' with specific character names found in the provided text.\n"
            "3. Clarify possessive pronouns (his, her, their) when the reference is unclear.\n"
            "4. If you think that a sentence can be rephrased to be more clear, do so.\n"
            "5. Do not add or remove any information beyond the replacements.\n"
            "6. Separate each sentence with a newline character.\n"
            "7. You will be given the previous context (if any) of the text you are modifying.\n"
            "8. You should only output the rewritten [TEXT TO MODIFY] text. You should never output the text included in the [PREVIOUS CONTEXT] tags.\n\n"
            "9. CRITICAL: Do not make up character names. Only replace names that are explicitly mentioned in the text provided. If you don't read the name of a character in the text, do not replace it.\n"
            "10. If you cannot determine who a pronoun refers to with certainty, keep the pronoun or use a generic reference like 'the speaker' or 'a character' rather than guessing a specific name.\n"
            "11. When multiple characters are present but the referent is unclear, use generic references like 'one of the characters' or 'a character' instead of assuming a specific name."
        )
        
        if i > 0:
            # Use the last 'context_size' sentences from the previous chunk as context
            context = '\n'.join(named_sentences[-context_size:])
            prompt += f"[PREVIOUS CONTEXT]\n{context}\n[/PREVIOUS CONTEXT]\n\n"
        
        prompt += f"[TEXT TO MODIFY]\n{' '.join(chunk)}\n[/TEXT TO MODIFY]\n\nRewritten text:"
        
        logger.info(f"Sending prompt to LLM for pronoun and reference replacement:\n{prompt}\n")
        response = intelligent_llm.invoke([HumanMessage(content=prompt)])  # Updated method
        logger.info(f"Received response from LLM:\n{response.content}\n")  # Corrected to use content
        
        response_sentences = [s.strip() for s in response.content.split('\n') if s.strip()]  # Corrected to use content
        
        # If this is not the first chunk, remove any sentences that are duplicates from the context
        if i > 0:
            response_sentences = [s for s in response_sentences if s not in named_sentences[-context_size:]]
        
        named_sentences.extend(response_sentences)
    
    named_sentences = remove_duplicates(named_sentences)
    return '\n'.join(named_sentences)

def simplify_text(text: str, llm: AzureChatOpenAI, window_size: int = None) -> str:
    """
    Call the LLM to simplify the text using a sliding window approach. We ask it to rephrase the sentences in a simpler way, preferring shorter sentences and a dot at the end of each sentence.

    Args:
        text (str): The text to simplify.
        llm (AzureChatOpenAI): The language model instance.
        window_size (int, optional): Number of sentences to process in each batch. If None, uses config value.

    Returns:
        str: The simplified text.
    """
    # Use config value if not provided
    if window_size is None:
        window_size = config.text_simplification_batch_size
    
    logger.info(f"Using simplification batch size: {window_size}")
    
    sentences = split_into_sentences(text)
    simplified_sentences = []

    for i in range(0, len(sentences), window_size):
        window = sentences[i:i+window_size]
        window_text = ' '.join(window)
        
        prompt = dedent(f"""You are a text simplification and clarification expert. You will be given a text and you will need to simplify and clarify it.
        You will be rephrasing the sentences in a simpler way, preferring shorter periods, but very clear and precise, avoiding ambiguities and maintaining all the details about place, time, characters, and actions.
        You will try to avoid direct quotes and instead use a indirect quotation.
        You will split the text in a way that each period is focused on a single character or event.
        You will not add any other text or commentary, nor backticks or markdown formatting.
        Please simplify the following text:\n{window_text}""")
        
        logger.info(f"Sending prompt to LLM for text simplification:\n{prompt}\n")
        
        for attempt in range(3):  # Retry up to 3 times
            try:
                response = llm.invoke([HumanMessage(content=prompt)])  # Corrected method
                logger.info(f"Received response from LLM:\n{response.content}\n")
                break  # Exit the retry loop if successful
            except Exception as e:
                logger.error(f"Error during LLM call for text simplification (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    raise
        
        simplified_sentences.extend(split_into_sentences(response.content.strip()))
    
    simplified_text = '\n'.join(simplified_sentences)
    return simplified_text
