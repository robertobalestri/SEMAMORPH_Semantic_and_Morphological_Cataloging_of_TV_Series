import spacy
from typing import List, Dict, Tuple
from collections import defaultdict
from src.plot_processing.plot_processing_models import EntityLink
from src.utils.logger_utils import setup_logging
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from src.utils.llm_utils import clean_llm_json_response
import json
import os
import re
from textwrap import dedent
import logging

titles_list = ["Dr", "Dr.", "Mr", "Mr.", "Ms", "Ms.", "Miss", "Miss.", "Mrs", "Mrs.", "Mme", "Mme.", "Prof", "Prof.", "Doc", "Doctor", "Sir", "Lady", "Dame", "Duke", "Duchess", "Count", "Countess", "Baron", "Baroness", "King", "Queen", "Prince", "Princess", "Earl", "Don", "Sister", "Brother", "Colonel", "Major", "Captain", "Lieutenant", "Sergeant", "Private", "Doctorate", "PhD", "MD", "MA", "BA", "BS", "BSc", "Bachelor", "Master", "Masters", "M.A.", "M.B.A.", "M.D.", "M.S.", "M.Sc.", "M.B.A.", "M.D.", "M.S.", "M.Sc."]

logger = setup_logging(__name__)
nlp = spacy.load("en_core_web_trf")

def extract_entities_with_spacy(text: str) -> List[str]:
    doc = nlp(text)
    entities = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities.add(ent.text)
    return list(entities)

def extract_entities_with_spacy(text: str) -> List[str]:
    """
    Extract entities from the text using spaCy's NER, including titles.

    Args:
        text (str): The input text to process.

    Returns:
        List[str]: A list of entity mentions, including titles.
    """
    doc = nlp(text)
    entities = set()

    for ent in doc.ents:
        if ent.label_ == "PERSON":  # Only process PERSON entities
            # Check if the previous token is a title
            #se gli ultimi due caratteri sono 's allora li rimuove
            entity_text = ent.text if not ent.text.endswith("'s") else ent.text[:-2]
            
            prev_token = doc[ent.start - 1] if ent.start > 0 else None
            if prev_token and prev_token.text in titles_list:
                # Combine title with the entity text
                entities.add(f"{prev_token.text} {entity_text}")
            else:
                entities.add(entity_text)

    return list(entities)

def refine_entities(entities: List[str], plot: str, llm: AzureChatOpenAI, existing_entities: List[EntityLink]) -> List[EntityLink]:
    existing_info = "Existing characters:\n" + "\n".join([f"{e.entity_name}" for e in existing_entities])
    new_entities = "New characters to refine:\n" + "\n".join(entities)

    prompt = dedent(f"""Given the following characters extracted from the plot, please refine them:
                Plot: {plot}
                {existing_info}
                
                {new_entities}
                
                For each character:
                1. Provide a primary name in the format "name_surname" (lowercase with underscores). If a character has only one name or surname with a title, use the title and the name (e.g. "dr_johnson").
                2. In appellations, list ALL names, titles and nicknames used for the character in the plot.
                3. Choose from the appellations found the best appellation for the character, we will use it as the entity name. When possible, choose the appellation with name and surname, or with title and surname.
                4. Include full names, first names, last names, titles, and nicknames in appellations.
                5. For families or groups, use "the_surname" format. Families and members of the family are different entities. They should be added separately and the family members should not be listed in the appellations of the family.
                6. Treat individuals with the same surname (e.g., Mr. Smith and Mrs. Smith) as separate entities unless there's clear evidence they're the same person.
                6. Exclude generic appellatives (e.g. "the baby", "the old man", "the woman", "she", "he", etc.) and non-character entities (like name of Universities, Companies, etc.), also exclude names that are not characters (for example in "Bob looks like Dumbo", Dumbo is not a character).
                7. Sometimes some characters might be referred only with a generic term like "the dragon", "the mage", "the robot", etc. In those cases use the generic term as the entity name.
                8. If a character is not in the existing entities list, you can still add it if you are sure that it's a new character.
                
                Respond with a JSON list of objects in this format:
                [
                    {{
                        "entity_name": "primary_name",
                        "best_appellation": "Chosen Appellation",
                        "appellations": ["Appellation1", "Appellation2", ...]
                    }},
                    ...
                ]
                """)

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        refined_entities = clean_llm_json_response(response.content)
        logger.info(f"Refined entities: {refined_entities}")
        
        if isinstance(refined_entities, list) and len(refined_entities) > 0:
            return [EntityLink(**entity) for entity in refined_entities]
        else:
            logger.error(f"Unexpected format for refined entities: {refined_entities}")
            return []
    except ValueError as e:
        logger.error(f"Failed to clean and parse LLM response: {e}")
        return []

def merge_entities(existing_entities: List[EntityLink], new_entities: List[EntityLink], llm: AzureChatOpenAI) -> List[EntityLink]:
    merged = defaultdict(lambda: {"entity_name": "", "best_appellation": "","appellations": set()})  # Added best_appellation
    
    # First, process existing entities
    for entity in existing_entities:
        key = entity.entity_name.lower()
        merged[key]["entity_name"] = entity.entity_name
        merged[key]["best_appellation"] = entity.best_appellation
        merged[key]["appellations"].update(entity.appellations)
        
    
    # Then, process new entities
    for new_entity in new_entities:
        lower_name = new_entity.entity_name.lower()
        
        # Check if this entity already exists
        if lower_name in merged:
            merged[lower_name]["appellations"].update(new_entity.appellations)
            merged[lower_name]["best_appellation"] = new_entity.best_appellation
        else:
            # Check for shared appellations
            shared_appellation_entities = []
            for existing_key, existing_data in merged.items():
                shared_appellations = set(existing_data["appellations"]) & set(new_entity.appellations)
                if shared_appellations:
                    shared_appellation_entities.append((existing_key, shared_appellations))
            
            if shared_appellation_entities:
                # Call LLM to disambiguate
                should_merge, merge_with = disambiguate_entities(new_entity, shared_appellation_entities, merged, llm)
                if should_merge:
                    merged[merge_with]["appellations"].update(new_entity.appellations)
                    merged[merge_with]["best_appellation"] = new_entity.best_appellation  # Update best_appellation
                    logger.info(f"Merged '{new_entity.entity_name}' with existing entity '{merged[merge_with]['entity_name']}' based on LLM disambiguation")
                else:
                    merged[lower_name]["entity_name"] = new_entity.entity_name
                    merged[lower_name]["appellations"].update(new_entity.appellations)
                    merged[lower_name]["best_appellation"] = new_entity.best_appellation  # Store best_appellation
                    logger.info(f"Added '{new_entity.entity_name}' as a new entity based on LLM disambiguation")
            else:
                # No shared appellations, add as a new entity
                merged[lower_name]["entity_name"] = new_entity.entity_name
                merged[lower_name]["appellations"].update(new_entity.appellations)
                merged[lower_name]["best_appellation"] = new_entity.best_appellation  # Store best_appellation
    
    # Convert back to EntityLink objects
    return [EntityLink(entity_name=data["entity_name"], appellations=list(data["appellations"]), best_appellation=data["best_appellation"]) for data in merged.values()]  # Include best_appellation

def disambiguate_entities(new_entity: EntityLink, shared_appellation_entities: List[Tuple[str, set]], merged: Dict, llm: AzureChatOpenAI) -> Tuple[bool, str]:
    prompt = f"""I need to determine if the following entity should be merged with any existing entities or kept separate:

New Entity: {new_entity.entity_name}
Appellations: {', '.join(new_entity.appellations)}

Existing entities with shared appellations:
"""
    for existing_key, shared_appellations in shared_appellation_entities:
        prompt += f"- {merged[existing_key]['entity_name']} (Shared appellations: {', '.join(shared_appellations)})\n"
        prompt += f"  All appellations: {', '.join(merged[existing_key]['appellations'])}\n"

    prompt += """
Based on this information, should the new entity be merged with one of the existing entities, or should it be kept as a separate entity?
If it should be merged, specify which entity it should be merged with.

Respond in the following JSON format:
{
    "should_merge": true/false,
    "merge_with": "entity_name_to_merge_with_or_null_if_not_merging",
    "explanation": "Brief explanation of the decision"
}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        result = json.loads(clean_llm_json_response(response.content))
        logger.info(f"LLM disambiguation result: {result}")
        return result["should_merge"], result["merge_with"]
    except Exception as e:
        logger.error(f"Failed to parse LLM response for entity disambiguation: {e}")
        return False, ""

def extract_and_refine_entities(text: str, llm: AzureChatOpenAI, refined_output_path: str, raw_output_path: str, season_entities_path: str) -> List[EntityLink]:
   
    # Extract new entities
    extracted_entities = extract_entities_with_spacy(text)
    
    with open(raw_output_path, "w", encoding="utf-8") as raw_file:
        json.dump(extracted_entities, raw_file, indent=2)

    # Load existing season entities
    existing_entities = []
    if os.path.exists(season_entities_path):
        with open(season_entities_path, 'r') as f:
            existing_entities = [EntityLink(**entity) for entity in json.load(f)]
        logger.info(f"Loaded {len(existing_entities)} existing entities from season file")

    # Pre-process text with existing entities
    preprocessed_text = substitute_appellations_with_names(text, existing_entities, llm)
    logger.info("Pre-processed text with existing season entities")

    # Refine new entities, considering existing ones
    refined_entities = refine_entities(extracted_entities, preprocessed_text, llm, existing_entities)
    
    # Merge new entities with existing ones, prioritizing season-level information
    merged_entities = merge_entities(existing_entities, refined_entities, llm)
    
    # Update season entities file with new information
    with open(season_entities_path, "w", encoding="utf-8") as season_file:
        json.dump([entity.model_dump() for entity in merged_entities], season_file, indent=2)
    
    with open(refined_output_path, "w", encoding="utf-8") as refined_file:
        json.dump([entity.model_dump() for entity in refined_entities], refined_file, indent=2)
    
    return merged_entities

def substitute_appellations_with_names(text: str, entities: List[EntityLink], llm: AzureChatOpenAI) -> str:
    # Sort entities by the length of their longest appellation
    sorted_entities = sorted(entities, key=lambda e: max(len(a) for a in e.appellations), reverse=True)
    
    # Keep track of substitutions
    substitutions: List[Tuple[str, str, int, int]] = []
    ambiguous_substitutions: List[Tuple[str, List[EntityLink], int, int]] = []

    for entity in sorted_entities:
        # Sort appellations by length in descending order
        sorted_appellations = sorted(entity.appellations, key=len, reverse=True)
        
        for appellation in sorted_appellations:
            # Use word boundaries and capture surrounding whitespace, optional possessive form, and following punctuation
            pattern = r'(\s|^)(' + re.escape(appellation) + r"(?:'s)?)(\s|[.,!?;]|$)"
            
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                start, end = match.span(2)  # Get the span of the actual appellation (group 2)
                
                # Check if this span overlaps with any previous substitution
                if not any(s <= start < e or s < end <= e for _, _, s, e in substitutions):
                    # Check if this appellation is ambiguous (belongs to multiple entities)
                    ambiguous_entities = [e for e in entities if appellation.lower() in [a.lower() for a in e.appellations]]
                    if len(ambiguous_entities) > 1:
                        ambiguous_substitutions.append((appellation, ambiguous_entities, start, end))
                    else:
                        # Preserve possessive form if present
                        replacement = f"[{entity.entity_name}]" + match.group(2)[len(appellation):]
                        substitutions.append((match.group(2), replacement, start, end))

    # Resolve ambiguous substitutions using LLM
    for appellation, ambiguous_entities, start, end in ambiguous_substitutions:
        context = text[max(0, start-100):min(len(text), end+100)]  # Get surrounding context
        entity_name = disambiguate_appellation(appellation, ambiguous_entities, context, llm)
        if entity_name:
            # Preserve possessive form if present
            replacement = f"[{entity_name}]" + text[start:end][len(appellation):]
            substitutions.append((text[start:end], replacement, start, end))

    # Apply substitutions in reverse order of their position in the text
    for original, replacement, start, end in sorted(substitutions, key=lambda x: x[2], reverse=True):
        logger.debug(f"Substituting '{original}' with '{replacement}' at positions {start}:{end}")
        prefix = text[:start]
        suffix = text[end:]
        text = f"{prefix}{replacement}{suffix}"

    return text

def disambiguate_appellation(appellation: str, entities: List[EntityLink], context: str, llm: AzureChatOpenAI) -> str:
    prompt = f"""Given the following context and a list of possible entities, determine which entity the appellation refers to:

Context: "{context}"

Appellation: "{appellation}"

Possible entities:
{', '.join([e.entity_name for e in entities])}

Respond with the name of the entity that the appellation most likely refers to in this context. If you're unsure, respond with "UNCERTAIN".

Your response should be in the following JSON format:
{{
    "entity_name": "chosen_entity_name_or_UNCERTAIN",
    "explanation": "Brief explanation of your decision"
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        result = json.loads(clean_llm_json_response(response.content))
        logger.info(f"LLM disambiguation result for appellation '{appellation}': {result}")
        return result["entity_name"] if result["entity_name"] != "UNCERTAIN" else None
    except Exception as e:
        logger.error(f"Failed to parse LLM response for appellation disambiguation: {e}")
        logger.error(f"Raw response: {response.content}")
        return None

def normalize_entities_names_to_best_appellation(text: str, entities: List[EntityLink]) -> str:
    #substitute the [entity_name] with the best_appellation
    for entity in entities:
        text = text.replace(f"[{entity.entity_name}]", entity.best_appellation)
    return text
