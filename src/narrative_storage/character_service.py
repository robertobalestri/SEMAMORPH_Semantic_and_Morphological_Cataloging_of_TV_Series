# character_service.py

from sqlite3 import IntegrityError
from typing import List, Optional, Union
from src.narrative_storage.narrative_models import Character, CharacterAppellation, NarrativeArc, ArcProgression
from src.narrative_storage.repositories import CharacterRepository
from src.plot_processing.plot_processing_models import EntityLink
import logging

from src.utils.logger_utils import setup_logging
logger = setup_logging(__name__)

class CharacterService:
    """Service to manage character-related operations."""

    def __init__(self, character_repository: CharacterRepository):
        self.character_repository = character_repository

    def add_or_update_character(self, entity: EntityLink, series: str) -> Optional[Union[Character, List[Character]]]:
        """Add a new character or update an existing one based on the EntityLink."""
        # Skip invalid entity names
        if len(entity.entity_name.strip()) <= 1:
            logger.warning(f"Skipping invalid entity name: '{entity.entity_name}'")
            return None

        try:
            # Ensure entity_name is normalized
            normalized_entity_name = self._normalize_name(entity.entity_name)
            entity.entity_name = normalized_entity_name

            # Normalize and validate appellations
            validated_appellations = set()
            for appellation in entity.appellations:
                if appellation and len(appellation.strip()) > 1:
                    validated_appellations.add(appellation.strip())
            
            # Ensure best_appellation is valid and included
            if entity.best_appellation and len(entity.best_appellation.strip()) > 1:
                validated_appellations.add(entity.best_appellation.strip())
            elif validated_appellations:
                # If no valid best_appellation, use the first valid appellation
                entity.best_appellation = next(iter(validated_appellations))
            else:
                logger.warning(f"No valid appellations found for entity: {entity.entity_name}")
                return None

            # First try to find existing character by normalized entity_name
            existing_character = self.character_repository.get_by_entity_name(normalized_entity_name, series)
            
            # If not found by entity_name, try to find by any appellation
            if not existing_character:
                for appellation in validated_appellations:
                    existing_character = self.character_repository.get_character_by_appellation(
                        appellation, 
                        series
                    )
                    if existing_character:
                        break

            if existing_character:
                # Update existing character
                # Clear existing appellations to avoid duplicates
                existing_character.appellations.clear()
                
                # Update best appellation
                existing_character.best_appellation = entity.best_appellation

                # Add all validated appellations
                for appellation in validated_appellations:
                    # Check if appellation is used by another character
                    other_character = self.character_repository.get_character_by_appellation(appellation, series)
                    if other_character and other_character.entity_name != existing_character.entity_name:
                        logger.warning(
                            f"Appellation '{appellation}' is already used by character "
                            f"'{other_character.entity_name}', skipping."
                        )
                        continue
                    
                    new_appellation = CharacterAppellation(
                        appellation=appellation,
                        character_id=existing_character.entity_name
                    )
                    existing_character.appellations.append(new_appellation)

                self.character_repository.update(existing_character)
                logger.info(
                    f"Updated existing character: {existing_character.entity_name} "
                    f"with appellations: {[app.appellation for app in existing_character.appellations]}"
                )
                return existing_character
            else:
                # Create new character
                new_character = Character(
                    entity_name=normalized_entity_name,
                    best_appellation=entity.best_appellation,
                    series=series
                )

                # Add all validated appellations
                for appellation in validated_appellations:
                    # Check if appellation is already used
                    existing_char = self.character_repository.get_character_by_appellation(appellation, series)
                    if existing_char:
                        logger.warning(
                            f"Appellation '{appellation}' is already used by character "
                            f"'{existing_char.entity_name}', skipping."
                        )
                        continue

                    new_appellation = CharacterAppellation(
                        appellation=appellation,
                        character_id=new_character.entity_name
                    )
                    new_character.appellations.append(new_appellation)

                if not new_character.appellations:
                    logger.warning(f"No valid appellations for character: {entity.entity_name}")
                    return None

                self.character_repository.add(new_character)
                logger.info(
                    f"Added new character: {new_character.entity_name} "
                    f"with appellations: {[app.appellation for app in new_character.appellations]}"
                )
                return new_character

        except IntegrityError as e:
            logger.error(f"Database integrity error when processing character {entity.entity_name}: {e}")
            self.character_repository.session.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error when processing character {entity.entity_name}: {e}")
            self.character_repository.session.rollback()
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

    def merge_characters(self, character1_id: str, character2_id: str, series: str) -> bool:
        """
        Merge two characters, updating all related arcs and progressions.
        The first character (character1_id) will be kept, the second will be deleted.
        """
        try:
            char1 = self.character_repository.get_by_entity_name(character1_id, series)
            char2 = self.character_repository.get_by_entity_name(character2_id, series)

            if not char1 or not char2:
                return False

            # Merge appellations
            existing_appellations = {app.appellation for app in char1.appellations}
            for app in char2.appellations:
                if app.appellation not in existing_appellations:
                    app.character = char1
                    existing_appellations.add(app.appellation)

            # Update main characters in arcs
            for arc in char2.main_narrative_arcs:
                if char1 not in arc.main_characters:
                    arc.main_characters.append(char1)
                arc.main_characters.remove(char2)

            # Update interfering characters in progressions
            for progression in char2.interfering_progressions:
                if char1 not in progression.interfering_characters:
                    progression.interfering_characters.append(char1)
                progression.interfering_characters.remove(char2)

            # Delete the second character
            self.character_repository.delete(char2)
            return True

        except Exception as e:
            logger.error(f"Error merging characters {character1_id} and {character2_id}: {e}")
            raise

    def _normalize_name(self, name: str) -> str:
        """Helper method to normalize character names."""
        return name.lower().replace(' ', '_')
