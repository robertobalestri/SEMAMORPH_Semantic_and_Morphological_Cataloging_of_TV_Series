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

        # Process single character
        return self._add_or_update_single_character(entity, series)

    def _add_or_update_single_character(self, entity: EntityLink, series: str) -> Optional[Character]:
        # Skip invalid entity names
        if len(entity.entity_name.strip()) <= 1:
            logger.warning(f"Skipping invalid entity name: '{entity.entity_name}'")
            return None

        try:
            # Try to find existing character
            character = self.character_repository.get_by_entity_name(entity.entity_name, series)
            if character:
                # Update existing character with new appellations
                existing_appellations = {app.appellation for app in character.appellations}
            else:
                existing_appellations = set()

            new_appellations = set()
            for app in entity.appellations:
                app = app.strip()
                if len(app) > 1:
                    # Check if appellation is already used by another character
                    existing_character = self.character_repository.get_character_by_appellation(app, series)
                    if existing_character and existing_character.entity_name != entity.entity_name:
                        logger.warning(f"Appellation '{app}' is already used by character '{existing_character.entity_name}', skipping.")
                    else:
                        new_appellations.add(app)

            all_appellations = existing_appellations.union(new_appellations)

            if not all_appellations:
                logger.warning(f"No valid appellations for character: {entity.entity_name}")
                return None

            # Update or create character
            if character:
                # Update best appellation if provided
                if entity.best_appellation and len(entity.best_appellation.strip()) > 1:
                    character.best_appellation = entity.best_appellation

                # Update appellations
                character.appellations = [
                    CharacterAppellation(appellation=app, character_id=character.entity_name)
                    for app in all_appellations
                ]
                self.character_repository.update(character)
                logger.info(f"Updated existing character: {character.entity_name}")
            else:
                # Create new character
                character = Character(
                    entity_name=entity.entity_name,
                    best_appellation=entity.best_appellation,
                    series=series,
                    appellations=[
                        CharacterAppellation(appellation=app, character_id=entity.entity_name)
                        for app in all_appellations
                    ]
                )
                self.character_repository.add(character)
                logger.info(f"Added new character: {character.entity_name}")

            return character

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
            if ';' in appellation:
                cleaned_appellations.extend([name.strip() for name in appellation.split(';') if name.strip()])
            else:
                cleaned_appellations.append(appellation.strip())

        # Add explicit series filter to ensure we only get characters from this series
        return self.character_repository.get_by_appellations(cleaned_appellations, series)

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
