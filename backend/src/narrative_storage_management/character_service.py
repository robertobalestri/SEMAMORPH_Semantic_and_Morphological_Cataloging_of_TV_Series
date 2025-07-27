# character_service.py

from sqlite3 import IntegrityError
from typing import List, Optional, Union, Dict
from ..narrative_storage_management.narrative_models import Character, CharacterAppellation, NarrativeArc, ArcProgression
from ..narrative_storage_management.repositories import CharacterRepository
from ..plot_processing.plot_processing_models import EntityLink
import logging

from ..utils.logger_utils import setup_logging
logger = setup_logging(__name__)

class CharacterService:
    """Service to manage character-related operations."""

    def __init__(self, character_repository: CharacterRepository):
        self.character_repository = character_repository

    def add_or_update_character(self, entity: EntityLink, series: str) -> Optional[Union[Character, List[Character]]]:
        """Add a new character or update an existing one based on the EntityLink."""
        logger.info(f"🔧 ADD_OR_UPDATE_CHARACTER: Starting for entity '{entity.entity_name}' in series '{series}'")
        
        # Skip invalid entity names
        if len(entity.entity_name.strip()) <= 1:
            logger.warning(f"Skipping invalid entity name: '{entity.entity_name}'")
            return None

        try:
            logger.info(f"🔍 Validating appellations for entity: {entity.entity_name}")
            # Normalize and validate appellations first
            validated_appellations = set()
            for appellation in entity.appellations:
                if appellation and len(appellation.strip()) > 1:
                    validated_appellations.add(appellation.strip())
            
            logger.info(f"📊 Validated appellations: {validated_appellations}")
            
            # Ensure best_appellation is valid and included
            if entity.best_appellation and len(entity.best_appellation.strip()) > 1:
                validated_appellations.add(entity.best_appellation.strip())
                logger.info(f"✅ Best appellation '{entity.best_appellation}' is valid")
            elif validated_appellations:
                # If no valid best_appellation, use the first valid appellation
                entity.best_appellation = next(iter(validated_appellations))
                logger.info(f"🔄 Using first appellation as best: '{entity.best_appellation}'")
            else:
                logger.warning(f"No valid appellations found for entity: {entity.entity_name}")
                return None

            logger.info(f"🔍 Looking for existing character by appellations...")
            # CRITICAL: Find existing character by ANY appellation first to get canonical entity_name
            existing_character = None
            canonical_entity_name = None
            
            # First check if we already have this character by any appellation
            for appellation in validated_appellations:
                logger.info(f"🔍 Checking appellation: '{appellation}'")
                existing_character = self.character_repository.get_character_by_appellation(
                    appellation, 
                    series
                )
                if existing_character:
                    canonical_entity_name = existing_character.entity_name
                    logger.info(f"🔗 Found existing character by appellation '{appellation}': {canonical_entity_name}")
                    break
                else:
                    logger.info(f"❌ No existing character found for appellation: '{appellation}'")
            
            # If not found by appellation, try by normalized entity_name
            if not existing_character:
                logger.info(f"🔍 No existing character found by appellations, checking by entity_name...")
                normalized_entity_name = self._normalize_name(entity.entity_name)
                logger.info(f"🔍 Normalized entity_name: '{entity.entity_name}' → '{normalized_entity_name}'")
                existing_character = self.character_repository.get_by_entity_name(normalized_entity_name, series)
                if existing_character:
                    canonical_entity_name = existing_character.entity_name
                    logger.info(f"🔗 Found existing character by entity_name: {canonical_entity_name}")
                else:
                    # No existing character found, use normalized name as canonical
                    canonical_entity_name = normalized_entity_name
                    logger.info(f"🆕 Creating new character with canonical entity_name: {canonical_entity_name}")
            
            # IMPORTANT: Always use the canonical entity_name for consistency
            if canonical_entity_name is None:
                logger.error(f"Failed to determine canonical entity_name for {entity.entity_name}")
                return None
            entity.entity_name = canonical_entity_name
            logger.info(f"✅ Using canonical entity_name: '{canonical_entity_name}'")

            if existing_character:
                logger.info(f"🔄 Updating existing character: {existing_character.entity_name}")
                # Update existing character
                # Clear existing appellations to avoid duplicates
                existing_character.appellations.clear()
                logger.info(f"🧹 Cleared existing appellations")
                
                # Update best appellation
                existing_character.best_appellation = entity.best_appellation
                logger.info(f"📝 Updated best_appellation to: '{entity.best_appellation}'")
                
                # Update biological sex
                existing_character.biological_sex = entity.biological_sex  # NEW: Update biological sex
                logger.info(f"📝 Updated biological_sex to: '{entity.biological_sex}'")

                # Add all validated appellations
                for appellation in validated_appellations:
                    logger.info(f"🔍 Checking if appellation '{appellation}' is used by another character...")
                    # Check if appellation is used by another character
                    other_character = self.character_repository.get_character_by_appellation(appellation, series)
                    if other_character and other_character.entity_name != existing_character.entity_name:
                        logger.warning(
                            f"Appellation '{appellation}' is already used by character "
                            f"'{other_character.entity_name}', skipping."
                        )
                        continue
                    
                    logger.info(f"✅ Adding appellation '{appellation}' to existing character")
                    new_appellation = CharacterAppellation(
                        appellation=appellation,
                        character_id=existing_character.entity_name
                    )
                    existing_character.appellations.append(new_appellation)
                
                logger.info(f"💾 Saving updated character to database...")
                self.character_repository.update(existing_character)
                logger.info(f"✅ Successfully updated character: {existing_character.entity_name}")
                return existing_character
            else:
                # Create new character with canonical entity_name
                logger.info(f"🆕 Creating new character: {entity.entity_name}")
                new_character = Character(
                    entity_name=entity.entity_name,  # Using the validated canonical entity_name
                    best_appellation=entity.best_appellation,
                    series=series,
                    biological_sex=entity.biological_sex  # NEW: Store biological sex
                )
                logger.info(f"📝 Created Character object: entity_name='{new_character.entity_name}', best_appellation='{new_character.best_appellation}', series='{new_character.series}', biological_sex='{new_character.biological_sex}'")

                # Add all validated appellations
                logger.info(f"🔍 Adding {len(validated_appellations)} appellations to new character...")
                for appellation in validated_appellations:
                    logger.info(f"🔍 Checking if appellation '{appellation}' is already used...")
                    # Check if appellation is already used
                    existing_char = self.character_repository.get_character_by_appellation(appellation, series)
                    if existing_char:
                        logger.warning(
                            f"Appellation '{appellation}' is already used by character "
                            f"'{existing_char.entity_name}', skipping."
                        )
                        continue

                    logger.info(f"✅ Adding appellation '{appellation}' to new character")
                    new_appellation = CharacterAppellation(
                        appellation=appellation,
                        character_id=new_character.entity_name
                    )
                    new_character.appellations.append(new_appellation)

                if not new_character.appellations:
                    logger.warning(f"No valid appellations for character: {entity.entity_name}")
                    return None

                logger.info(f"💾 Saving new character to database...")
                self.character_repository.add(new_character)
                logger.info(f"✅ Successfully created new character: {new_character.entity_name} with appellations: {[app.appellation for app in new_character.appellations]}")
                return new_character

        except IntegrityError as e:
            logger.error(f"Database integrity error when processing character {entity.entity_name}: {e}")
            # Session rollback is handled by session_scope() context manager
            raise
        except Exception as e:
            logger.error(f"Unexpected error when processing character {entity.entity_name}: {e}")
            # Session rollback is handled by session_scope() context manager
            raise

    def get_characters_by_appellations(self, appellations: List[str], series: str) -> List[Character]:
        """Get characters by matching their appellations."""
        # Clean and split any semicolon-separated appellations
        cleaned_appellations = []
        for appellation in appellations:
            if isinstance(appellation, str):  # Ensure we're working with a string
                # Split by semicolon and clean each part
                if ';' in appellation:
                    cleaned_appellations.extend([
                        name.strip() 
                        for name in appellation.split(';') 
                        if name.strip()
                    ])
                else:
                    cleaned_appellations.append(appellation.strip())

        # Get characters by appellations
        characters = self.character_repository.get_by_appellations(cleaned_appellations, series)
        
        if not characters:
            logger.warning(f"No characters found for appellations: {cleaned_appellations}")
        else:
            logger.info(f"Found {len(characters)} characters for appellations: {cleaned_appellations}")
        
        return characters

    def link_characters_to_arc(self, characters: List[Character], arc: Optional[NarrativeArc]):
        if not arc:
            return  # Handle linking within add_arc

        existing_ids = {c.entity_name for c in arc.main_characters}
        new_characters = [c for c in characters if c.entity_name not in existing_ids]
        if new_characters:
            arc.main_characters.extend(new_characters)
            for character in new_characters:
                if arc not in character.main_narrative_arcs:
                    character.main_narrative_arcs.append(arc)
            logger.info(f"Linked {len(new_characters)} characters to arc '{arc.title}'")

    def link_characters_to_progression(self, characters: List[Character], progression: ArcProgression):
        existing_ids = {c.entity_name for c in progression.interfering_characters}
        new_characters = [c for c in characters if c.entity_name not in existing_ids]
        if new_characters:
            progression.interfering_characters.extend(new_characters)
            for character in new_characters:
                if progression not in character.interfering_progressions:
                    character.interfering_progressions.append(progression)
            logger.info(f"Linked {len(new_characters)} characters to progression in S{progression.season}E{progression.episode}")

    def delete_character(self, entity_name: str, series: str) -> bool:
        """
        Delete a character and update all related arcs and progressions.
        Returns True if successful, False otherwise.
        """
        try:
            character = self.character_repository.get_by_entity_name(entity_name, series)
            if not character:
                return False

            # Remove character from main characters in arcs
            for arc in character.main_narrative_arcs:
                arc.main_characters.remove(character)

            # Remove character from interfering characters in progressions
            for progression in character.interfering_progressions:
                progression.interfering_characters.remove(character)

            # Delete the character
            self.character_repository.delete(character)
            return True

        except Exception as e:
            logger.error(f"Error deleting character {entity_name}: {e}")
            raise

    def merge_characters(
        self,
        character1_id: str,
        character2_id: str,
        series: str,
        keep_character: str = 'character1'
    ) -> bool:
        """
        Merge two characters, keeping the data from the specified character.
        
        Args:
            character1_id: Entity name of the first character
            character2_id: Entity name of the second character
            series: Series identifier
            keep_character: Which character to keep ('character1' or 'character2')
        
        Returns:
            bool: True if merge was successful, False otherwise
        """
        # Get both characters
        char1 = self.character_repository.get_by_entity_name(character1_id, series)
        char2 = self.character_repository.get_by_entity_name(character2_id, series)

        if not char1 or not char2:
            return False

        # Determine which character to keep based on keep_character parameter
        kept_char = char1 if keep_character == 'character1' else char2
        merged_char = char2 if keep_character == 'character1' else char1

        # Merge appellations
        merged_appellations = set([app.appellation for app in kept_char.appellations])
        merged_appellations.update([app.appellation for app in merged_char.appellations])

        # Update the kept character with merged appellations
        kept_char.appellations = [
            CharacterAppellation(
                appellation=app,
                character_id=kept_char.entity_name
            )
            for app in merged_appellations
        ]

        # Update any references from narrative arcs
        for arc in merged_char.main_narrative_arcs:
            if kept_char not in arc.main_characters:
                arc.main_characters.append(kept_char)

        # Update any references from progressions
        for prog in merged_char.interfering_progressions:
            if kept_char not in prog.interfering_characters:
                prog.interfering_characters.append(kept_char)

        # Delete the other character
        self.character_repository.delete(merged_char)

        return True

    def _normalize_name(self, name: str) -> str:
        """Helper method to normalize character names."""
        return name.lower().replace(' ', '_')
    
    def get_canonical_entity_name(self, character_name_or_appellation: str, series: str) -> Optional[str]:
        """
        Get the canonical entity_name for any character name or appellation.
        
        This is CRITICAL for system consistency - all parts of the system should use this
        to ensure they reference the same canonical entity_name for character identification.
        
        Args:
            character_name_or_appellation: Any name or appellation for the character
            series: The series to search in
            
        Returns:
            The canonical entity_name if found, None otherwise
        """
        if not character_name_or_appellation or not character_name_or_appellation.strip():
            return None
            
        character_name_or_appellation = character_name_or_appellation.strip()
        
        # First try to find by appellation (most reliable)
        existing_character = self.character_repository.get_character_by_appellation(
            character_name_or_appellation, 
            series
        )
        if existing_character:
            logger.debug(f"🔗 Found canonical entity_name by appellation '{character_name_or_appellation}': {existing_character.entity_name}")
            return existing_character.entity_name
        
        # Try by normalized entity_name
        normalized_name = self._normalize_name(character_name_or_appellation)
        existing_character = self.character_repository.get_by_entity_name(normalized_name, series)
        if existing_character:
            logger.debug(f"🔗 Found canonical entity_name by entity_name '{character_name_or_appellation}': {existing_character.entity_name}")
            return existing_character.entity_name
        
        # Not found in database - return normalized name as potential canonical name
        logger.debug(f"🆕 Character not found in database, using normalized name as canonical: {normalized_name}")
        return normalized_name
    
    def build_character_name_mapping(self, series: str) -> Dict[str, str]:
        """
        Build a comprehensive mapping from any character name/appellation to canonical entity_name.
        
        This is essential for ensuring consistent character identification across all systems
        (face clustering, speaker identification, etc.).
        
        Args:
            series: The series to build mapping for
            
        Returns:
            Dictionary mapping all character names/appellations to canonical entity_names
        """
        mapping = {}
        
        try:
            # Get all characters for this series
            characters = self.character_repository.get_by_series(series)
            
            for character in characters:
                canonical_name = character.entity_name
                
                # Map the entity_name itself
                mapping[canonical_name] = canonical_name
                
                # Map the best_appellation
                if character.best_appellation:
                    mapping[character.best_appellation.lower()] = canonical_name
                    mapping[character.best_appellation] = canonical_name  # Case-sensitive version too
                
                # Map all appellations
                for appellation_obj in character.appellations:
                    if appellation_obj.appellation:
                        appellation = appellation_obj.appellation
                        mapping[appellation.lower()] = canonical_name
                        mapping[appellation] = canonical_name  # Case-sensitive version too
            
            logger.info(f"🗺️ Built character name mapping for series '{series}': {len(mapping)} mappings for {len(characters)} characters")
            
            # Log sample mappings for debugging
            for i, (name, canonical) in enumerate(mapping.items()):
                if i < 5:  # Show first 5 mappings
                    logger.debug(f"   📝 '{name}' → '{canonical}'")
                elif i == 5:
                    logger.debug(f"   📝 ... and {len(mapping) - 5} more mappings")
                    break
            
            return mapping
            
        except Exception as e:
            logger.error(f"❌ Error building character name mapping: {e}")
            return {}

    def process_entities(self, entities: List[EntityLink], series: str, plot: str = "", llm = None) -> List[Character]:
        """
        Process a list of entities and save them to the database.
        
        Args:
            entities: List of EntityLink objects to process
            series: The series these entities belong to
            plot: The plot text for context (used for LLM verification)
            llm: Optional LLM instance for verification
            
        Returns:
            List of processed Character objects
        """
        logger.info(f"🏭 CHARACTER SERVICE: Starting process_entities")
        logger.info(f"📊 DEBUG: Input parameters - entities_count={len(entities)}, series='{series}', plot_length={len(plot)}, llm={llm is not None}")
        
        if not entities:
            logger.warning("⚠️ No entities to process")
            return []
            
        logger.info(f"🏭 CHARACTER SERVICE: Processing {len(entities)} entities for series {series}")
        processed_characters = []
        
        for i, entity in enumerate(entities):
            try:
                logger.info(f"🔍 [{i+1}/{len(entities)}] Processing entity: {entity.entity_name} → {entity.best_appellation} ({entity.appellations})")
                
                # Skip invalid entities
                if not entity.entity_name or len(entity.entity_name.strip()) <= 1:
                    logger.warning(f"⚠️ Skipping invalid entity: {entity.entity_name}")
                    continue
                    
                logger.info(f"🔧 Calling add_or_update_character for entity: {entity.entity_name}")
                # Add or update the character
                result = self.add_or_update_character(entity, series)
                
                if result:
                    if isinstance(result, list):
                        processed_characters.extend(result)
                        logger.info(f"✅ Processed entity {entity.entity_name} → {len(result)} characters")
                    else:
                        processed_characters.append(result)
                        logger.info(f"✅ Processed entity {entity.entity_name} → 1 character")
                else:
                    logger.warning(f"⚠️ Failed to process entity: {entity.entity_name}")
            except Exception as e:
                logger.error(f"❌ Error processing entity {entity.entity_name}: {e}")
                import traceback
                logger.error(f"❌ Traceback: {traceback.format_exc()}")
                
        logger.info(f"✅ CHARACTER SERVICE: Completed processing. Total characters: {len(processed_characters)}")
        logger.info(f"📊 DEBUG: Returning {len(processed_characters)} processed characters")
        return processed_characters

    def get_episode_entities(self, series: str) -> List[Character]:
        """
        Get entities/characters for speaker validation.
        
        Args:
            series: The series to get characters for
            
        Returns:
            List of Character objects for the series
        """
        try:
            # Get all characters for this series - they could all potentially appear in any episode
            characters = self.character_repository.get_by_series(series)
            logger.info(f"📚 Loaded {len(characters)} characters from database for speaker validation")
            return characters
        except Exception as e:
            logger.warning(f"⚠️ Could not load episode entities from database: {e}")
            return []
