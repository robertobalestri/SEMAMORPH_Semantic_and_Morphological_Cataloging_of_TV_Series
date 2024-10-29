import spacy
from typing import List, Tuple
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
from src.storage.character_storage import CharacterStorage
from src.storage.database import DatabaseManager

logger = setup_logging(__name__)
nlp = spacy.load("en_core_web_trf")
db_manager = DatabaseManager()
character_storage = CharacterStorage()

titles_list = ["Dr", "Dr.", "Mr", "Mr.", "Ms", "Ms.", "Miss", "Miss.", "Mrs", "Mrs.", "Mme", "Mme.", "Prof", "Prof.", "Doc", "Doctor", "Sir", "Lady", "Dame", "Duke", "Duchess", "Count", "Countess", "Baron", "Baroness", "King", "Queen", "Prince", "Princess", "Earl", "Don", "Sister", "Brother", "Colonel", "Major", "Captain", "Lieutenant", "Sergeant", "Private", "Doctorate", "PhD", "MD", "MA", "BA", "BS", "BSc", "Bachelor", "Master", "Masters", "M.A.", "M.B.A.", "M.D.", "M.S.", "M.Sc.", "M.B.A.", "M.D.", "M.S.", "M.Sc."]

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

    # Process entities in spans
    current_entity = []
    for token in doc:
        # If we find a PERSON entity
        if token.ent_type_ == "PERSON":
            current_entity.append(token.text)
        else:
            # If we have collected a person entity, process it
            if current_entity:
                entity_text = " ".join(current_entity).strip()
                
                # Remove possessive 's if present
                if entity_text.endswith("'s"):
                    entity_text = entity_text[:-2].strip()
                
                # Check for title
                prev_token = doc[token.i - len(current_entity) - 1] if token.i - len(current_entity) > 0 else None
                if prev_token and prev_token.text in titles_list:
                    entity_text = f"{prev_token.text} {entity_text}"

                # Add entity if it's valid
                if len(entity_text.strip()) > 1 and not all(c in {';', ',', '.', ' '} for c in entity_text):
                    entities.add(entity_text)
                
                current_entity = []

    # Handle any remaining entity at the end of the text
    if current_entity:
        entity_text = " ".join(current_entity).strip()
        if len(entity_text) > 1 and not all(c in {';', ',', '.', ' '} for c in entity_text):
            entities.add(entity_text)

    return list(entities)

def refine_entities(entities: List[str], plot: str, llm: AzureChatOpenAI, existing_entities: List[EntityLink]) -> List[EntityLink]:
    """
    Refine the extracted entities using a language model to ensure accuracy and completeness.

    Args:
        entities (List[str]): The list of extracted entities.
        plot (str): The plot text for context.
        llm (AzureChatOpenAI): The language model for refining entities.
        existing_entities (List[EntityLink]): The list of existing entities for reference.

    Returns:
        List[EntityLink]: A list of refined entities as EntityLink objects.
    """
    existing_info = "Existing characters:\n" + "\n".join([f"{e.entity_name}" for e in existing_entities])
    new_entities = "New characters to refine:\n" + "\n".join(entities)

    prompt = dedent(f"""Given the following characters extracted from the plot, please refine them:
                Plot: {plot}
                {existing_info}
                
                {new_entities}
                
                For each character:
                1. Provide a primary name in the format "name_surname" (lowercase with underscores). If a character has only one name or surname with a title, use the title and the name (e.g. "dr_johnson").
                2. In appellations, list ALL names, titles and nicknames used for the character in the plot.
                3. Include full names, first names, last names, titles, and nicknames in appellations.
                4. For families or groups, use "the_surname" format. Families and members of the family are different entities. They should be added separately and the family members should not be listed in the appellations of the family.
                5. Treat individuals with the same surname (e.g., Mr. Smith and Mrs. Smith) as separate entities unless there's clear evidence they're the same person.
                6. Exclude generic appellatives (e.g. "the baby", "the old man", "the woman", "she", "he", etc.) and non-character entities (like name of Universities, Companies, etc.), also exclude names that are not characters (for example in "Bob looks like Dumbo", Dumbo is not a character).
                7. Sometimes some characters might be referred only with a generic term like "the dragon", "the mage", "the robot", etc. In those cases use the generic term as the entity name.
                8. If a character is not in the existing entities list, you can still add it if you are sure that it's a new character.
                9. To choose the best appellation follow this rule: if name and surname are present ALWAYS use just them without title. If only one of the two is present, but it's present also a title, use the appellation that comprehends also the title.

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

def extract_and_refine_entities(text: str, series: str, llm: AzureChatOpenAI, refined_output_path: str, raw_output_path: str, season_entities_path: str) -> List[EntityLink]:
    """
    Extract and refine entities from the provided text, saving results to specified paths.
    Also ensures all entities are stored in the database.
    """
    # Extract new entities
    extracted_entities = extract_entities_with_spacy(text)
    
    # Save raw entities
    with open(raw_output_path, "w", encoding="utf-8") as raw_file:
        json.dump(extracted_entities, raw_file, indent=2)

    # Load existing entities
    existing_entities = []
    if os.path.exists(season_entities_path):
        with open(season_entities_path, 'r') as f:
            existing_entities = [EntityLink(**entity) for entity in json.load(f)]
        logger.info(f"Loaded {len(existing_entities)} existing entities from season file")

    # Pre-process and refine entities
    preprocessed_text = substitute_appellations_with_names(text, existing_entities, llm)
    refined_entities = refine_entities(extracted_entities, preprocessed_text, llm, existing_entities)

    # Store all entities in the database
    with db_manager.session_scope() as session:
        for entity in refined_entities:
            character_storage.get_or_create_character(entity, series, session)

    # Save refined entities
    with open(refined_output_path, "w", encoding="utf-8") as refined_file:
        json.dump([entity.model_dump() for entity in refined_entities], refined_file, indent=2)
    
    # Update season entities file
    with open(season_entities_path, "w", encoding="utf-8") as season_file:
        json.dump([entity.model_dump() for entity in refined_entities], season_file, indent=2)
    
    return refined_entities

def substitute_appellations_with_names(text: str, entities: List[EntityLink], llm) -> str:
    """
    Substitute appellations in the text with their corresponding best appellations.
    Sorts entities by appellation length to handle longer appellations first.
    """
    if not entities:
        return text

    # Sort entities by appellation length (longest first) to avoid partial replacements
    sorted_entities = sorted(entities, key=lambda e: len(e.entity_name), reverse=True)
    
    # Create a copy of the text to modify
    modified_text = text

    for entity in sorted_entities:
        # Replace entity name with best appellation
        if entity.entity_name and len(entity.entity_name.strip()) > 1:  # Skip empty or single-char names
            modified_text = modified_text.replace(entity.entity_name, entity.best_appellation)

    return modified_text

def normalize_entities_names_to_best_appellation(text: str, entities: List[EntityLink]) -> str:
    """Normalize entity names in the text to their best appellations."""
    for entity in entities:
        text = text.replace(f"[{entity.entity_name}]", entity.best_appellation)
    return text

def normalize_names(text: str, entities: List[EntityLink], llm: AzureChatOpenAI) -> str:
    """Normalize names in the text using substitution and best appellations."""
    substituted_text = substitute_appellations_with_names(text, entities, llm)
    normalized_text = normalize_entities_names_to_best_appellation(substituted_text, entities)
    return normalized_text
