from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
import logging
import json
import re
from src.utils.logger_utils import setup_logging
from typing import List, Dict, Union
import re

load_dotenv(override=True)

# Set up logging
logger = setup_logging(__name__)

# Global variables to store LLM instances
_intelligent_llm = None
_cheap_llm = None

def _initialize_llm(intelligent_or_cheap: str) -> AzureChatOpenAI:
    """
    Initialize and return an instance of AzureChatOpenAI LLM.

    Args:
        intelligent_or_cheap (str): Specify whether to initialize the 'intelligent' or 'cheap' LLM.

    Returns:
        AzureChatOpenAI: An instance of the AzureChatOpenAI LLM.
    """
    try:
        if intelligent_or_cheap == "intelligent":
            return AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME_INTELLIGENT"),
                model=os.getenv("AZURE_OPENAI_LLM_MODEL_NAME_INTELLIGENT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=0.2,
            )
        elif intelligent_or_cheap == "cheap":
            return AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME_CHEAP"),
                model=os.getenv("AZURE_OPENAI_LLM_MODEL_NAME_CHEAP"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=0.2,
            )
        else:
            raise ValueError(f"Invalid LLM type: {intelligent_or_cheap}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

def get_llm(intelligent_or_cheap: str) -> AzureChatOpenAI:
    """
    Get the initialized LLM instance. If not initialized, initialize it first.

    Args:
        intelligent_or_cheap (str): Specify whether to get the 'intelligent' or 'cheap' LLM.

    Returns:
        AzureChatOpenAI: An instance of the AzureChatOpenAI LLM.
    """
    global _intelligent_llm, _cheap_llm

    if intelligent_or_cheap == "intelligent":
        if _intelligent_llm is None:
            _intelligent_llm = _initialize_llm("intelligent")
        return _intelligent_llm
    elif intelligent_or_cheap == "cheap":
        if _cheap_llm is None:
            _cheap_llm = _initialize_llm("cheap")
        return _cheap_llm
    else:
        raise ValueError(f"Invalid LLM type: {intelligent_or_cheap}")

def clean_llm_json_response(response: str) -> List[Dict]:
    """
    Clean and extract a valid JSON array from the LLM response.
    
    Args:
        response (str): The raw response from the LLM, expected to contain JSON.
    
    Returns:
        List[Dict]: A cleaned list of dictionaries containing the entities.
    
    Raises:
        ValueError: If no valid JSON object or array found in the response.
    """
    
    # Remove from the string ```json, ```plaintext, ```markdown and ```
    response_cleaned = re.sub(r"```(json|plaintext|markdown)?", "", response)
    
    # Remove any comments
    response_cleaned = re.sub(r'//.*?$|/\*.*?\*/', '', response_cleaned, flags=re.MULTILINE | re.DOTALL)
    
    # Try to extract a JSON object or array from the cleaned response
    json_match = re.search(r'(\{|\[)[\s\S]*(\}|\])', response_cleaned)
    
    if json_match:
        json_str = json_match.group(0)
        
        # Remove any trailing commas inside objects and arrays
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Ensure all keys are properly quoted
        json_str = re.sub(r'(\w+)(?=\s*:)', r'"\1"', json_str)
        
        # Fix common JSON issues
        json_str = re.sub(r'(\w+)\s*:\s*([a-zA-Z0-9_]+)', r'"\1": "\2"', json_str)  # Ensure values are quoted
        
        # Handle nested objects and arrays
        json_str = re.sub(r'([\[{])\s*,\s*', r'\1', json_str)  # Remove leading commas in arrays/objects
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)  # Remove trailing commas in arrays/objects
        json_str = re.sub(r"’", "'", json_str)  # Replace all curly apostrophies with regular apostrophies
        
        
        # Wrap the response in an array if it contains a single object
        if json_str.startswith('{'):
            json_str = f'[{json_str}]'  # Wrap single object in an array
        
        # Try to parse the JSON
        try:
            parsed_json = json.loads(json_str)
            
            # If the parsed JSON is a dictionary, convert it to a list
            if isinstance(parsed_json, dict):
                return [parsed_json]  # Wrap the single entity in a list
            elif isinstance(parsed_json, list):
                return parsed_json  # Return the parsed list of dictionaries
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse cleaned JSON: {e}")
            logger.error(f"Cleaned JSON string: {json_str}")
            logger.error(f"Full response: {response}")
            raise ValueError(f"Failed to parse cleaned JSON: {e}")
    
    # If no valid JSON is found, raise an error
    raise ValueError("No valid JSON object or array found in the response")

def clean_llm_text_response(response: str) -> str:
    """
    Clean and extract meaningful text from a markdown or plaintext LLM response.
    
    Args:
        response (str): The raw response from the LLM, expected to be plaintext or markdown.
    
    Returns:
        str: A cleaned version of the text.
    """
    
    # Remove any formatting indicators like ```plaintext, ```markdown, and ```
    response_cleaned = re.sub(r"```(plaintext|markdown|html)?", "", response)
    response_cleaned = re.sub(r"```", "", response_cleaned)  # Remove all triple backticks
    response_cleaned = re.sub(r"’", "'", response_cleaned)  # Replace all curly apostrophies with regular apostrophies
    # Strip extra whitespace
    response_cleaned = response_cleaned.strip()
    
    return response_cleaned