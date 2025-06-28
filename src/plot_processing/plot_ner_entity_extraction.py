# plot_ner_entity_extraction.py

import spacy
import re
import json
import os
import logging
from textwrap import dedent
from typing import List, Dict, Set, Any
from collections import defaultdict

from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate

from src.utils.logger_utils import setup_logging
from src.utils.llm_utils import clean_llm_json_response
from src.utils.character_entities_utils import TITLES, normalize_entity_name, has_conflicting_gender_titles, extract_surname_from_appellation
from src.narrative_storage_management.repositories import DatabaseSessionManager, CharacterRepository
from src.narrative_storage_management.character_service import CharacterService
from src.plot_processing.plot_processing_models import EntityLink, EntityLinkEncoder

logger = setup_logging(__name__)
nlp = spacy.load("en_core_web_trf")

def extract_entities_with_spacy(text: str) -> List[str]:
    """
    Extract PERSON entities using spaCy NER and attach preceding titles if found.
    
    Args:
        text: The input text to process
        
    Returns:
        List of entity mentions, including titles
    """
    logger.info(f"Extracting entities from text of length: {len(text)}")
    doc = nlp(text)
    entities: Set[str] = set()
    current_entity_tokens = []

    # Debug: Print all entities detected by spaCy
    all_entities = [(ent.text, ent.label_) for ent in doc.ents]
    logger.debug(f"All entities detected by spaCy: {all_entities[:10]}...")  # Show first 10

    for token in doc:
        if token.ent_type_ == "PERSON":
            current_entity_tokens.append(token)
        else:
            if current_entity_tokens:
                # Build candidate entity string
                entity_text = " ".join(tok.text for tok in current_entity_tokens).strip()
                if entity_text.endswith("'s"):
                    entity_text = entity_text[:-2].strip()
                    
                # Check for a preceding title
                first_token_index = current_entity_tokens[0].i
                if first_token_index - 1 >= 0:
                    prev_token = doc[first_token_index - 1]
                    if prev_token.text in TITLES:
                        entity_text = f"{prev_token.text} {entity_text}"
                        
                if len(entity_text) > 1 and not all(c in {';', ',', '.', ' '} for c in entity_text):
                    entities.add(entity_text)
                    logger.debug(f"Added entity: {entity_text}")
                current_entity_tokens = []
                
    if current_entity_tokens:
        entity_text = " ".join(tok.text for tok in current_entity_tokens).strip()
        if len(entity_text) > 1:
            entities.add(entity_text)
            logger.debug(f"Added final entity: {entity_text}")
    
    logger.info(f"Extracted {len(entities)} entities")
    return list(entities)

def extract_and_refine_entities(
    named_plot: str,
    series: str,
    llm: AzureChatOpenAI,
    refined_output_path: str,
    raw_output_path: str,
    season_entities_path: str
) -> List[EntityLink]:
    """
    Main pipeline:
      1. Extract entities with spaCy.
      2. Save raw extractions if paths provided.
      3. Load existing season entities if path provided.
      4. Pre-substitute known appellations in the text.
      5. Refine new entities using the LLM.
      6. Remove conflicting surname appellations between male and female characters.
      7. Merge refined entities with existing ones using improved logic.
      8. Merge duplicate entities.
      9. Save the final entities to the database and output files if requested.
      
    Args:
        named_plot: The plot text to process
        series: The series name for database storage
        llm: The language model to use for refinement
        refined_output_path: Path to save refined entities
        raw_output_path: Path to save raw spaCy entities
        season_entities_path: Path to season entities file
        
    Returns:
        List of final extracted and refined EntityLink objects
    """
    logger.info(f"Starting entity extraction and refinement for series: {series}")

    # Log the pipeline steps
    logger.info("🔍 ENTITY PROCESSING PIPELINE:")
    logger.info("   1. Extract entities with SpaCy")
    logger.info("   2. Load existing season entities")
    logger.info("   3. Preprocess text with known appellations")
    logger.info("   4. Refine entities with LLM")
    logger.info("   5. Remove conflicting surname appellations")
    logger.info("   6. Merge with existing entities")
    logger.info("   7. Deduplicate entities")
    logger.info("   8. Save to database and files")
    logger.info("=" * 60)

    # 1. Extract entities
    logger.info("🔍 STEP 1: EXTRACTING ENTITIES WITH SPACY")
    # Check if raw entities file already exists
    if os.path.exists(raw_output_path):
        try:
            logger.info(f"Loading raw entities from existing file: {raw_output_path}")
            with open(raw_output_path, "r", encoding="utf-8") as f:
                extracted_entities = json.load(f)
            logger.info(f"✅ Loaded {len(extracted_entities)} entities from existing raw file")
            logger.info(f"📋 Raw entities: {extracted_entities}")
        except Exception as e:
            logger.error(f"Error loading raw entities from file: {e}")
            # Extract entities if loading fails
            extracted_entities = extract_entities_with_spacy(named_plot)
            
            # Save raw extractions if path provided
            if raw_output_path:
                os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
                with open(raw_output_path, "w", encoding="utf-8") as f:
                    json.dump(extracted_entities, f, indent=2)
                logger.info(f"Saved raw spaCy entities to {raw_output_path}")
    else:
        # Extract entities with spaCy
        logger.info("🔍 Extracting new entities with SpaCy...")
        extracted_entities = extract_entities_with_spacy(named_plot)
        logger.info(f"✅ SpaCy extracted {len(extracted_entities)} entities")
        logger.info(f"📋 Raw entities: {extracted_entities}")
        
        # Save raw extractions if path provided
        if raw_output_path:
            os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
            with open(raw_output_path, "w", encoding="utf-8") as f:
                json.dump(extracted_entities, f, indent=2)
            logger.info(f"Saved raw spaCy entities to {raw_output_path}")

    # 3. Load existing entities if available
    logger.info("🔍 STEP 2: LOADING EXISTING SEASON ENTITIES")
    existing_entities: List[EntityLink] = []
    if season_entities_path and os.path.exists(season_entities_path):
        try:
            with open(season_entities_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                existing_entities = [EntityLink(**entity) for entity in loaded]
            logger.info(f"✅ Loaded {len(existing_entities)} existing entities from previous episodes")
            for entity in existing_entities:
                logger.info(f"   📋 Existing: {entity.entity_name} → {entity.best_appellation} ({entity.appellations})")
        except Exception as e:
            logger.error(f"Error loading existing entities: {e}")

    # 4. Preprocess text using known appellations
    logger.info("🔍 STEP 3: PREPROCESSING TEXT WITH KNOWN APPELLATIONS")
    preprocessed_text = named_plot
    if existing_entities:
        # Create a mapping of appellations to entity names for substitution
        appellation_to_entity = {}
        
        for entity in existing_entities:
            for appellation in entity.appellations:
                if appellation and len(appellation.strip()) > 1:
                    appellation_to_entity[appellation.strip()] = entity.entity_name
            # Also add the best appellation
            if entity.best_appellation and len(entity.best_appellation.strip()) > 1:
                appellation_to_entity[entity.best_appellation.strip()] = entity.entity_name

        # Sort appellations by length (longest first) to avoid partial matches
        # This ensures "Meredith Grey" is processed before "Meredith" or "Grey"
        sorted_appellations = sorted(appellation_to_entity.keys(), key=len, reverse=True)
        
        logger.debug(f"Preprocessing with {len(sorted_appellations)} appellations in order: {sorted_appellations[:5]}...")
        
        # Substitute appellations with entity names
        for appellation in sorted_appellations:
            # Use word boundaries to avoid partial word matches, case-insensitive
            pattern = r'\b' + re.escape(appellation) + r'\b'
            entity_name = appellation_to_entity[appellation]
            
            # Count matches before substitution for debugging
            matches_before = len(re.findall(pattern, preprocessed_text, re.IGNORECASE))
            if matches_before > 0:
                logger.debug(f"Preprocessing: '{appellation}' -> '[{entity_name}]' ({matches_before} matches)")
                preprocessed_text = re.sub(pattern, f"[{entity_name}]", preprocessed_text, flags=re.IGNORECASE)
        
        logger.info(f"✅ Preprocessed text with {len(sorted_appellations)} known appellations")

    # 5. Refine new entities via LLM
    logger.info("🔍 STEP 4: REFINING ENTITIES WITH LLM")
    refined_entities = []
    if extracted_entities:
        # Create prompt for LLM to refine entities
        prompt_template = """
        You are an expert in character identification and name standardization.

        Here is the text:
        ```
        {named_plot}
        ```

        Here are the entities extracted by a named entity recognition system:
        {entities}

        For each entity, provide:
        1. The standardized entity name (lowercase with underscores)
        2. The best appellation (most complete/formal version of the name)
        3. All possible appellations (variations of the name)

        IMPORTANT GUIDELINES:
        - Be careful with names that share the same surname (like "Derek Shepherd" and "Addison Shepherd"). These are often different characters who are related or married, NOT the same person.
        - Characters with different titles but the same surname (like "Mr. Johnson" and "Mrs. Johnson") are almost always DIFFERENT people.
        - Standardize entity names by removing titles and using lowercase with underscores.
        - ALWAYS use the real name for the entity_name, NOT nicknames (e.g., use "miranda_bailey" not "the_nazi").
        - For the best_appellation, ALWAYS prefer the most formal version with title if available (e.g., "Dr. Miranda Bailey" is better than "Miranda Bailey" or "The Nazi").
        - If a character has a nickname, include it as an appellation but NEVER as the entity_name.
        - Split compound appellations like "Mr. and Mrs. Johnson" into separate entities: "mr_johnson" and "mrs_johnson".
        - NEVER mix gender-specific titles (Mr./Mrs./Miss/Ms.) for the same entity. If you see "Mr. Johnson" and "Mrs. Johnson", these are different people.
        - CRITICAL: NEVER include a common appellation (like just "Johnson") for entities with different gender-specific titles, as this could create confusion between different people with the same surname.
        - List all possible variations of the name as appellations.
        - If you're unsure about an entity, include it anyway with your best guess.

        Format your response as a JSON array of objects with the following structure:
        [
            {{
                "entity_name": "standardized_name",
                "best_appellation": "Best Formal Name",
                "appellations": ["Variation 1", "Variation 2", ...]
            }},
            ...
        ]

        Only include the JSON array in your response, nothing else.
        """
        
        prompt = PromptTemplate.from_template(prompt_template)
        formatted_prompt = prompt.format(
            named_plot=preprocessed_text,
            entities=", ".join(extracted_entities)
        )
        
        # Get response from LLM
        response = llm.invoke([HumanMessage(content=formatted_prompt)])
        
        # Save raw LLM response to file for debugging
        debug_response_path = os.path.join(os.path.dirname(raw_output_path), "llm_raw_response.txt")
        try:
            os.makedirs(os.path.dirname(debug_response_path), exist_ok=True)
            with open(debug_response_path, "w", encoding="utf-8") as f:
                f.write(response.content)
            logger.info(f"Saved raw LLM response to {debug_response_path}")
        except Exception as e:
            logger.error(f"Error saving raw LLM response: {e}")
        
        # Parse response
        try:
            # Clean the response and get parsed entities (this returns a Python list, not a JSON string)
            parsed_entities = clean_llm_json_response(response.content)
            
            # Convert to EntityLink objects
            for entity_data in parsed_entities:
                entity_name = entity_data.get("entity_name", "").strip()
                best_appellation = entity_data.get("best_appellation", "").strip()
                appellations = entity_data.get("appellations", [])
                
                # Skip empty entities
                if not entity_name:
                    continue
                    
                # Create EntityLink object
                entity_link = EntityLink(
                    entity_name=entity_name,
                    best_appellation=best_appellation if best_appellation else entity_name,
                    appellations=appellations if appellations else [entity_name]
                )
                
                refined_entities.append(entity_link)
                
            logger.info(f"✅ LLM refined {len(refined_entities)} entities")
            for entity in refined_entities:
                logger.info(f"   📋 LLM Result: {entity.entity_name} → {entity.best_appellation} ({entity.appellations})")
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            # Fallback: Create basic EntityLink objects for each extracted entity
            for entity_name in extracted_entities:
                entity_link = EntityLink(
                    entity_name=normalize_entity_name(entity_name),
                    best_appellation=entity_name,
                    appellations=[entity_name]
                )
                refined_entities.append(entity_link)
            logger.info(f"⚠️ Created {len(refined_entities)} basic entities after LLM parsing failure")
            for entity in refined_entities:
                logger.info(f"   📋 Fallback: {entity.entity_name} → {entity.best_appellation} ({entity.appellations})")

    # 6. Remove conflicting surname appellations between male and female characters
    logger.info("🔍 STEP 5: CHECKING FOR CONFLICTING SURNAME APPELLATIONS")
    if refined_entities:
        logger.info(f"📊 Before surname conflict check: {len(refined_entities)} entities")
        for entity in refined_entities:
            logger.info(f"   📋 Before: {entity.entity_name} → {entity.best_appellation} ({entity.appellations})")
        
        refined_entities = remove_conflicting_surname_appellations(refined_entities, llm)
        
        logger.info(f"✅ After surname conflict check: {len(refined_entities)} entities")
        for entity in refined_entities:
            logger.info(f"   📋 After: {entity.entity_name} → {entity.best_appellation} ({entity.appellations})")
    else:
        logger.info("⚠️ No entities to check for surname conflicts")

    # 7. Prepare entities for database - Include existing entities for proper merging
    logger.info("🔍 STEP 6: PREPARING ENTITIES FOR DATABASE PROCESSING")
    entities_for_database = list(refined_entities)  # Start with new LLM-processed entities
    
    # Add existing entities to database processing so they can be properly merged
    entities_for_database.extend(existing_entities)
    logger.info(f"📊 Entities for database: {len(entities_for_database)} = {len(refined_entities)} new + {len(existing_entities)} existing")
    
    logger.info("📋 All entities going to database processing:")
    for i, entity in enumerate(entities_for_database):
        source = "NEW" if i < len(refined_entities) else "EXISTING"
        logger.info(f"   {source}: {entity.entity_name} → {entity.best_appellation} ({entity.appellations})")
    
    # 8. Prepare entities for season file (merge with existing to preserve other episodes' data)
    logger.info("🔍 STEP 7: PREPARING ENTITIES FOR SEASON FILE")
    merged_entities = list(refined_entities)  # Start with refined entities (LLM results are authoritative)
    
    # Add existing entities that don't conflict with refined entities (for season file only)
    refined_entity_names = {normalize_entity_name(e.entity_name) for e in refined_entities}
    for entity in existing_entities:
        normalized_name = normalize_entity_name(entity.entity_name)
        if normalized_name not in refined_entity_names:
            merged_entities.append(entity)
            logger.debug(f"Added existing entity for season file: {entity.entity_name}")
        else:
            logger.debug(f"Skipping existing entity that conflicts with refined entity: {entity.entity_name}")
    
    # 9. Merge duplicate entities for database entities only - IMPROVED LOGIC
    logger.info("🔍 STEP 8: DEDUPLICATING ENTITIES FOR DATABASE")
    logger.info(f"📊 Before deduplication: {len(entities_for_database)} entities")
    for entity in entities_for_database:
        logger.info(f"   📋 Before dedup: {entity.entity_name} → {entity.best_appellation} ({entity.appellations})")
    
    final_entities_for_database = merge_duplicate_entities_improved(entities_for_database, named_plot, llm)
    
    logger.info(f"✅ After deduplication: {len(final_entities_for_database)} entities")
    for entity in final_entities_for_database:
        logger.info(f"   📋 After dedup: {entity.entity_name} → {entity.best_appellation} ({entity.appellations})")
    
    # 10. Merge duplicate entities for season file - IMPROVED LOGIC
    logger.info("🔍 STEP 9: DEDUPLICATING ENTITIES FOR SEASON FILE")
    final_entities_for_season = merge_duplicate_entities_improved(merged_entities, named_plot, llm)
    logger.info(f"✅ Season file entities after deduplication: {len(final_entities_for_season)}")
    
    # 11. Save ONLY LLM-processed entities to the database
    logger.info("🔍 STEP 10: SAVING TO DATABASE")
    if series:
        try:
            db_manager = DatabaseSessionManager()
            with db_manager.session_scope() as session:
                character_repository = CharacterRepository(session)
                character_service = CharacterService(character_repository)
                
                # Store ONLY LLM-processed entities in the database
                logger.info(f"📊 FINAL ENTITIES GOING TO DATABASE: {len(final_entities_for_database)}")
                for entity in final_entities_for_database:
                    logger.info(f"   → {entity.entity_name}: {entity.best_appellation} (appellations: {entity.appellations})")
                
                processed_characters = character_service.process_entities(final_entities_for_database, series, named_plot, llm)
                logger.info(f"✅ Saved {len(processed_characters)} characters to the database")
                
                # Log what was actually saved
                for char in processed_characters:
                    appellations_list = [app.appellation for app in char.appellations] if hasattr(char, 'appellations') else []
                    logger.info(f"   💾 SAVED: {char.entity_name} → {char.best_appellation} ({appellations_list})")
                    
        except Exception as e:
            logger.error(f"❌ Error saving to database: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
    else:
        logger.warning("⚠️ No series specified, skipping database save")

    # 12. Save to output files
    if refined_output_path:
        try:
            os.makedirs(os.path.dirname(refined_output_path), exist_ok=True)
            with open(refined_output_path, "w", encoding="utf-8") as refined_file:
                json.dump([EntityLinkEncoder().default(ent) for ent in refined_entities], refined_file, indent=2)
            logger.info(f"Saved refined entities to {refined_output_path}")
        except Exception as e:
            logger.error(f"Error saving refined entities: {e}")
    
    if season_entities_path:
        try:
            os.makedirs(os.path.dirname(season_entities_path), exist_ok=True)
            with open(season_entities_path, "w", encoding="utf-8") as season_file:
                json.dump([EntityLinkEncoder().default(ent) for ent in final_entities_for_season], season_file, indent=2)
            logger.info(f"Saved final entities for season to {season_entities_path}")
        except Exception as e:
            logger.error(f"Error saving final entities: {e}")

    return final_entities_for_database  # Return only LLM-processed entities

def extract_and_refine_entities_with_path_handler(
    path_handler,
    series: str = None
) -> List[EntityLink]:
    """
    Enhanced pipeline that uses PathHandler to manage all file paths.
    
    Args:
        path_handler: PathHandler object with all necessary path methods
        series: Optional series name (will be extracted from path_handler if not provided)
        
    Returns:
        List of final extracted and refined EntityLink objects
    """
    from src.ai_models.ai_models import get_llm, LLMType
    
    # Get series name from path_handler if not provided
    if series is None:
        series = path_handler.get_series()
    
    # Get the LLM
    llm = get_llm(LLMType.INTELLIGENT)
    
    # Load the named plot
    named_plot_path = path_handler.get_named_plot_file_path()
    try:
        with open(named_plot_path, "r", encoding="utf-8") as f:
            named_plot = f.read()
        logger.info(f"Loaded named plot with length: {len(named_plot)}")
    except Exception as e:
        logger.error(f"Error loading named plot from {named_plot_path}: {e}")
        raise
    
    # Call the main extraction function
    return extract_and_refine_entities(
        named_plot,
        series,
        llm,
        path_handler.get_episode_refined_entities_path(),
        path_handler.get_episode_raw_spacy_entities_path(),
        path_handler.get_season_extracted_refined_entities_path()
    )

def remove_conflicting_surname_appellations(entities: List[EntityLink], llm: AzureChatOpenAI) -> List[EntityLink]:
    """
    Removes conflicting surname appellations between male and female characters.
    
    Args:
        entities: List of EntityLink objects to check
        llm: LLM to use for verification
        
    Returns:
        List of EntityLink objects with conflicting appellations removed
    """
    if not entities or len(entities) <= 1:
        return entities
    
    logger.info(f"Checking for conflicting surname appellations among {len(entities)} entities")
    
    # Create a prompt template for the LLM
    prompt_template = """
    You are an expert in character identification and appellation standardization.
    
    I have a list of characters and their appellations. I need to identify and remove any common surname 
    appellations shared between characters with different gender titles (like Mr./Mrs./Ms./Miss).
    
    Here are the characters and their appellations:
    {entities_str}
    
    Your task:
    1. Identify any characters that have different gender-specific titles (Mr./Mrs./Ms./Miss) but share the same surname.
    2. For each such case, identify the common surname appellation that could cause confusion.
    3. Format your response as a JSON list of objects with the structure [{{"entity1": "entity_name1", "entity2": "entity_name2", "common_appellation": "shared_surname"}}]
    
    Example: If there's "Mr. Johnson" and "Mrs. Johnson" who both have "Johnson" as an appellation, 
    you would identify "Johnson" as a conflicting appellation that should be removed from both.
    
    Only output the JSON array with your findings, nothing else. If there are no conflicts, output an empty array [].    
    """
    
    # Format the entities for the prompt
    entities_str = ""
    for entity in entities:
        entities_str += f"Entity: {entity.entity_name}\n"
        entities_str += f"Best Appellation: {entity.best_appellation}\n"
        entities_str += f"Appellations: {', '.join(entity.appellations)}\n\n"
    
    prompt = PromptTemplate.from_template(prompt_template)
    formatted_prompt = prompt.format(entities_str=entities_str)
    
    # Get response from LLM
    try:
        logger.info("🔍 MAKING ADDITIONAL LLM CALL: Checking for conflicting surname appellations")
        response = llm.invoke([HumanMessage(content=formatted_prompt)])
        
        # Save this LLM response for debugging
        debug_response_path = "llm_surname_conflicts_response.txt"
        try:
            with open(debug_response_path, "w", encoding="utf-8") as f:
                f.write(response.content)
            logger.info(f"Saved surname conflicts LLM response to {debug_response_path}")
        except Exception as e:
            logger.error(f"Error saving surname conflicts LLM response: {e}")
        
        # Clean and parse the response (this returns a Python list, not a JSON string)
        conflicts = clean_llm_json_response(response.content)
        
        if not conflicts:
            logger.info("No conflicting surname appellations found")
            return entities
        
        logger.info(f"Found {len(conflicts)} potential conflicting surname appellations")
        
        # Create a deep copy of the entities to avoid modifying the originals
        updated_entities = []
        for entity in entities:
            updated_entity = EntityLink(
                entity_name=entity.entity_name,
                best_appellation=entity.best_appellation,
                appellations=entity.appellations.copy()
            )
            updated_entities.append(updated_entity)
        
        # Process each conflict
        for conflict in conflicts:
            entity1_name = conflict.get("entity1")
            entity2_name = conflict.get("entity2")
            common_appellation = conflict.get("common_appellation")
            
            if not all([entity1_name, entity2_name, common_appellation]):
                continue
            
            # Find the entities
            entity1 = next((e for e in updated_entities if e.entity_name == entity1_name), None)
            entity2 = next((e for e in updated_entities if e.entity_name == entity2_name), None)
            
            if not entity1 or not entity2:
                continue
            
            # Remove the common appellation from both entities
            if common_appellation in entity1.appellations:
                entity1.appellations.remove(common_appellation)
                logger.info(f"Removed conflicting appellation '{common_appellation}' from '{entity1.entity_name}'")
            
            if common_appellation in entity2.appellations:
                entity2.appellations.remove(common_appellation)
                logger.info(f"Removed conflicting appellation '{common_appellation}' from '{entity2.entity_name}'")
        
        return updated_entities
    
    except Exception as e:
        logger.error(f"Error checking for conflicting appellations: {e}")
        # If there's an error, return the original entities unchanged
        return entities

def merge_duplicate_entities(entities: List[EntityLink], text: str, llm: AzureChatOpenAI) -> List[EntityLink]:
    """
    Merge duplicate entities based on name similarity and LLM verification.
    
    Args:
        entities: List of EntityLink objects to check for duplicates
        text: Source text for context
        llm: LLM instance for verification
        
    Returns:
        List of EntityLink objects with duplicates merged
    """
    if not entities or len(entities) <= 1:
        return entities
    
    logger.info(f"Checking for duplicate entities among {len(entities)} entities")
    
    merged_entities = entities.copy()
    merged_indices = set()
    final_entities = []

    for i, entity1 in enumerate(merged_entities):
        if i in merged_indices:
            continue
        
        current_entity = EntityLink(
            entity_name=entity1.entity_name,
            best_appellation=entity1.best_appellation,
            appellations=entity1.appellations.copy()
        )

        # Gather any near-duplicates that share some textual similarity
        candidate_indices = set()
        name1 = normalize_entity_name(entity1.entity_name)
        for j, entity2 in enumerate(merged_entities):
            if j <= i or j in merged_indices:
                continue
            name2 = normalize_entity_name(entity2.entity_name)
            # Check for substring matches
            if name1 in name2 or name2 in name1:
                candidate_indices.add(j)

        # LLM decides if they should merge
        for j in candidate_indices:
            entity2 = merged_entities[j]
            if verify_entities_with_llm(current_entity, entity2, text, llm):
                # Merge them
                merged_indices.add(j)
                # Combine appellations
                for app in entity2.appellations:
                    if app not in current_entity.appellations:
                        current_entity.appellations.append(app)
                # If entity2 has a 'longer' name, keep that as the best_appellation
                if len(entity2.best_appellation) > len(current_entity.best_appellation):
                    current_entity.best_appellation = entity2.best_appellation
                logger.info(f"Merged entity '{entity2.entity_name}' into '{current_entity.entity_name}'")
        
        final_entities.append(current_entity)
    
    logger.info(f"Merged entities: original {len(entities)}, final {len(final_entities)}")
    return final_entities

def verify_entities_with_llm(entity1: EntityLink, entity2: EntityLink, text: str, llm: AzureChatOpenAI) -> bool:
    """
    Use the LLM to decide if two names represent the same character.
    
    Args:
        entity1: First EntityLink object
        entity2: Second EntityLink object
        text: Source text for context
        llm: LLM instance for verification
        
    Returns:
        True if entities should be merged, False otherwise
    """
    # Preliminary check - if names are identical after normalization, merge them
    name1 = normalize_entity_name(entity1.entity_name)
    name2 = normalize_entity_name(entity2.entity_name)
    if name1 == name2:
        return True

    # Check for conflicting gender titles - if found, don't merge
    if has_conflicting_gender_titles(entity1.appellations, entity2.appellations):
        logger.debug(f"Conflicting gender titles found between {entity1.entity_name} and {entity2.entity_name}")
        return False

    # Use LLM for complex cases
    prompt = f"""
    Determine if these two names in the following text are the same person or different people.
    
    Text context:
    {text[:2000]}...
    
    Name1: {entity1.entity_name} (appellations: {entity1.appellations})
    Name2: {entity2.entity_name} (appellations: {entity2.appellations})
    
    Consider:
    - Are they the same character referred to by different names/titles?
    - Are they family members with the same surname but different people?
    - Are they completely different characters?
    
    Respond ONLY with "Yes" if they are the same person, or "No" if different people.
    """

    try:
        logger.info(f"🔍 MAKING ADDITIONAL LLM CALL: Verifying if {entity1.entity_name} and {entity2.entity_name} are the same person")
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        result = ("yes" in response) and ("no" not in response)
        logger.info(f"LLM verification result: {entity1.entity_name} vs {entity2.entity_name} = {result} (response: '{response}')")
        return result
    except Exception as e:
        logger.error(f"LLM error during entity verification: {e}")
        # If LLM fails, default to not merging to be safe
        return False

# Legacy function wrappers for backward compatibility
def substitute_appellations_with_names(
    text: str,
    entities: List[EntityLink],
    llm: AzureChatOpenAI
) -> str:
    """
    Substitute appellations in the text with their corresponding best appellations.
    Uses a smart approach that finds all entity mentions first, then replaces them
    to avoid double-substitution issues.
    
    Args:
        text: The text to process
        entities: List of EntityLink objects
        llm: Language model (unused but kept for compatibility)
        
    Returns:
        Text with appellations substituted
    """
    if not entities:
        return text

    # Create a reverse mapping: appellation -> entity
    appellation_to_entity = {}
    for entity in entities:
        for appellation in entity.appellations:
            if appellation and len(appellation.strip()) > 1:
                appellation_to_entity[appellation.strip().lower()] = entity

    # Find all entity mentions in the text with their positions
    mentions = []  # List of (start, end, entity, original_text)
    
    # Sort appellations by length (longest first) to find longer matches first
    sorted_appellations = sorted(appellation_to_entity.keys(), key=len, reverse=True)
    
    logger.debug(f"Looking for {len(sorted_appellations)} appellations in text")
    
    # Find all mentions
    for appellation in sorted_appellations:
        pattern = r'\b' + re.escape(appellation) + r'\b'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.span()
            entity = appellation_to_entity[appellation]
            original_text = match.group()
            
            # Check if this position is already covered by a longer mention
            already_covered = False
            for existing_start, existing_end, _, _ in mentions:
                if start >= existing_start and end <= existing_end:
                    already_covered = True
                    logger.debug(f"Skipping '{original_text}' at {start}-{end} (covered by longer mention)")
                    break
            
            if not already_covered:
                mentions.append((start, end, entity, original_text))
                logger.debug(f"Found mention: '{original_text}' -> '{entity.best_appellation}' at {start}-{end}")
    
    # Sort mentions by position (from end to start to avoid position shifts)
    mentions.sort(key=lambda x: x[0], reverse=True)
    
    # Apply substitutions
    modified_text = text
    substitutions_made = 0
    
    for start, end, entity, original_text in mentions:
        if entity.best_appellation.strip() != original_text.strip():
            logger.debug(f"Replacing '{original_text}' with '{entity.best_appellation}' at position {start}-{end}")
            modified_text = modified_text[:start] + entity.best_appellation + modified_text[end:]
            substitutions_made += 1
        else:
            logger.debug(f"Keeping '{original_text}' (already best appellation)")
    
    logger.debug(f"Made {substitutions_made} substitutions")
    return modified_text

def normalize_entities_names_to_best_appellation(
    text: str,
    entities: List[EntityLink]
) -> str:
    """
    Normalize entity names in the text to their best appellations.
    Processes entities by longest entity name first to avoid partial matches.
    
    Args:
        text: The text to process
        entities: List of EntityLink objects
        
    Returns:
        Text with entity names normalized to best appellations
    """
    # Sort entities by entity_name length (longest first) to handle longer names first
    sorted_entities = sorted(entities, key=lambda e: len(e.entity_name), reverse=True)
    
    modified_text = text
    for entity in sorted_entities:
        # Replace bracketed entity names with best appellations
        pattern = f"[{entity.entity_name}]"
        if pattern in modified_text:
            logger.debug(f"Normalizing [{entity.entity_name}] -> {entity.best_appellation}")
            modified_text = modified_text.replace(pattern, entity.best_appellation)
    
    return modified_text

def normalize_names(
    text: str,
    entities: List[EntityLink],
    llm: AzureChatOpenAI
) -> str:
    """
    Normalize names in the text using substitution and best appellations.
    
    Args:
        text: The text to process
        entities: List of EntityLink objects
        llm: Language model (unused but kept for compatibility)
        
    Returns:
        Text with names normalized
    """
    substituted_text = substitute_appellations_with_names(text, entities, llm)
    normalized_text = normalize_entities_names_to_best_appellation(substituted_text, entities)
    return normalized_text

def force_reextract_entities_with_path_handler(
    path_handler,
    series: str = None
) -> List[EntityLink]:
    """
    Enhanced pipeline that forces re-extraction by deleting existing files first,
    then uses PathHandler to manage all file paths.
    
    Args:
        path_handler: PathHandler object with all necessary path methods
        series: Optional series name (will be extracted from path_handler if not provided)
        
    Returns:
        List of final extracted and refined EntityLink objects
    """
    # Get series name from path_handler if not provided
    if series is None:
        series = path_handler.get_series()
    
    # Remove existing files to force re-extraction
    files_to_remove = [
        path_handler.get_episode_refined_entities_path(),
        path_handler.get_episode_raw_spacy_entities_path(),
        # Note: We don't remove season entities path as it contains data from other episodes
    ]
    
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed existing file to force re-extraction: {file_path}")
        except Exception as e:
            logger.error(f"Error removing file {file_path}: {e}")
    
    # Now run the normal extraction process
    return extract_and_refine_entities_with_path_handler(path_handler, series)

def merge_duplicate_entities_improved(entities: List[EntityLink], text: str, llm: AzureChatOpenAI) -> List[EntityLink]:
    """
    Improved merge function that respects the original LLM's entity structuring decisions.
    
    The key insight: if the original LLM created "derek_shepherd" with appellations ["Dr. Shepherd", "Derek"],
    we should NOT split these into separate entities. The LLM already made the decision that these belong together.
    
    This function only merges:
    1. Entities with identical normalized names 
    2. Entities where one is clearly a subset/variation of another (with LLM confirmation)
    3. Cross-episode character matching (existing characters from previous episodes)
    
    Args:
        entities: List of EntityLink objects to check for duplicates
        text: Source text for context  
        llm: LLM instance for verification
        
    Returns:
        List of EntityLink objects with appropriate duplicates merged
    """
    if not entities or len(entities) <= 1:
        return entities
    
    logger.info(f"🔄 Improved merging: Checking {len(entities)} entities for legitimate duplicates")
    
    # Log all input entities first
    logger.info("📋 INPUT ENTITIES TO MERGE:")
    for i, entity in enumerate(entities):
        logger.info(f"   {i+1}. {entity.entity_name} → {entity.best_appellation} ({entity.appellations})")
    
    # Track entities that have been merged
    merged_indices = set()
    final_entities = []

    for i, entity1 in enumerate(entities):
        if i in merged_indices:
            logger.debug(f"⏭️ Skipping entity {i} (already merged): {entity1.entity_name}")
            continue
        
        logger.info(f"🔍 Processing entity {i+1}/{len(entities)}: {entity1.entity_name}")
        
        current_entity = EntityLink(
            entity_name=entity1.entity_name,
            best_appellation=entity1.best_appellation,
            appellations=entity1.appellations.copy()
        )

        # Look for entities that should definitely be merged
        merge_candidates = []
        for j in range(i + 1, len(entities)):
            if j in merged_indices:
                continue
                
            entity2 = entities[j]
            logger.debug(f"   🔍 Comparing with entity {j+1}: {entity2.entity_name}")
            
            # Rule 1: Identical normalized names - definitely merge
            name1_norm = normalize_entity_name(entity1.entity_name)
            name2_norm = normalize_entity_name(entity2.entity_name)
            
            if name1_norm == name2_norm:
                logger.info(f"✅ RULE 1: Identical normalized names: {entity1.entity_name} = {entity2.entity_name}")
                merge_candidates.append((j, entity2, "identical_names"))
                continue
            
            # Rule 2: Check if one entity name is contained in the other's appellations
            should_merge = False
            merge_reason = ""
            
            # Check if entity1's name appears in entity2's appellations
            if any(normalize_entity_name(app) == name1_norm for app in entity2.appellations):
                should_merge = True
                merge_reason = f"{entity1.entity_name} found in {entity2.entity_name}'s appellations"
                logger.info(f"✅ RULE 2A: {merge_reason}")
            
            # Check if entity2's name appears in entity1's appellations  
            elif any(normalize_entity_name(app) == name2_norm for app in entity1.appellations):
                should_merge = True
                merge_reason = f"{entity2.entity_name} found in {entity1.entity_name}'s appellations"
                logger.info(f"✅ RULE 2B: {merge_reason}")
            
            # Rule 3: Check for substring relationships (but be very careful)
            elif (name1_norm in name2_norm or name2_norm in name1_norm) and len(name1_norm) > 3 and len(name2_norm) > 3:
                logger.info(f"🤔 RULE 3: Potential substring match: {name1_norm} vs {name2_norm}")
                # Only merge if no conflicting gender titles
                if not has_conflicting_gender_titles(entity1.appellations, entity2.appellations):
                    logger.info(f"   ✅ No conflicting gender titles, asking LLM...")
                    # Ask LLM for verification, but with better context
                    if verify_entities_with_improved_context(entity1, entity2, text, llm):
                        should_merge = True
                        merge_reason = f"LLM confirmed {entity1.entity_name} and {entity2.entity_name} are the same"
                        logger.info(f"✅ RULE 3: {merge_reason}")
                    else:
                        logger.info(f"❌ RULE 3: LLM rejected merge of {entity1.entity_name} and {entity2.entity_name}")
                else:
                    logger.info(f"❌ RULE 3: Conflicting gender titles, not merging")
            
            if should_merge:
                merge_candidates.append((j, entity2, merge_reason))

        # Process all merge candidates
        if merge_candidates:
            logger.info(f"🔀 Merging {len(merge_candidates)} entities into {entity1.entity_name}:")
            for j, entity2, reason in merge_candidates:
                logger.info(f"   ➕ Merging {entity2.entity_name} ({reason})")
                merged_indices.add(j)
                
                # Merge appellations
                for app in entity2.appellations:
                    if app not in current_entity.appellations:
                        current_entity.appellations.append(app)
                        logger.debug(f"     ➕ Added appellation: {app}")
                
                # Use the more complete entity name (longer is usually better)
                if len(entity2.entity_name) > len(current_entity.entity_name):
                    old_name = current_entity.entity_name
                    current_entity.entity_name = entity2.entity_name
                    logger.info(f"     🔄 Updated entity name: {old_name} → {entity2.entity_name}")
                
                # Use the better appellation (prefer titles)
                if len(entity2.best_appellation) > len(current_entity.best_appellation):
                    old_appellation = current_entity.best_appellation
                    current_entity.best_appellation = entity2.best_appellation
                    logger.info(f"     🔄 Updated best appellation: {old_appellation} → {entity2.best_appellation}")
        else:
            logger.info(f"   ✅ No merges needed for {entity1.entity_name}")
        
        final_entities.append(current_entity)
        logger.info(f"   ✅ Final entity: {current_entity.entity_name} → {current_entity.best_appellation} ({current_entity.appellations})")
    
    logger.info(f"🔄 Improved merging complete: {len(entities)} → {len(final_entities)} entities")
    logger.info("📋 FINAL MERGED ENTITIES:")
    for i, entity in enumerate(final_entities):
        logger.info(f"   {i+1}. {entity.entity_name} → {entity.best_appellation} ({entity.appellations})")
    
    return final_entities


def verify_entities_with_improved_context(entity1: EntityLink, entity2: EntityLink, text: str, llm: AzureChatOpenAI) -> bool:
    """
    Improved LLM verification that provides better context and is more conservative.
    """
    # Don't merge if there are conflicting gender titles
    if has_conflicting_gender_titles(entity1.appellations, entity2.appellations):
        logger.debug(f"❌ Conflicting gender titles: {entity1.entity_name} vs {entity2.entity_name}")
        return False

    # More conservative prompt that biases toward NOT merging unless very confident
    prompt = f"""
    Context: I have an AI system that extracts characters from TV show episode summaries.
    
    The original AI identified these as potentially the same character:
    Character A: "{entity1.entity_name}" (known as: {entity1.appellations})  
    Character B: "{entity2.entity_name}" (known as: {entity2.appellations})
    
    Text excerpt:
    {text[:1500]}...
    
    IMPORTANT: Only merge if you are VERY CONFIDENT they refer to the same person.
    
    Consider:
    - Are these clearly the same character with different name forms?
    - Could these be family members with similar names? (DON'T merge if so)
    - Could these be different characters who just share a surname? (DON'T merge if so)
    
    If there's ANY doubt, say "No".
    
    Answer ONLY "Yes" (merge) or "No" (keep separate).
    """

    try:
        logger.info(f"🤔 LLM verification: {entity1.entity_name} vs {entity2.entity_name}")
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        result = response.startswith("yes") and not response.startswith("no")
        logger.info(f"🤔 LLM decision: {result} (response: '{response}')")
        return result
    except Exception as e:
        logger.error(f"❌ LLM verification error: {e}")
        # Conservative default: don't merge if LLM fails
        return False
