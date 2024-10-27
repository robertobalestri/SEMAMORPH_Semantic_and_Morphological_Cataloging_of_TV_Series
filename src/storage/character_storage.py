from typing import List, Optional
from sqlmodel import Session, select
from src.storage.narrative_models import Character
from src.plot_processing.plot_processing_models import EntityLink
from src.utils.logger_utils import setup_logging

logger = setup_logging(__name__)

class CharacterStorage:
    def get_or_create_character(self, entity: EntityLink, series: str, session: Session) -> Optional[Character]:
        """Get an existing character or create a new one."""
        # Process character names from semicolon-separated string
        if isinstance(entity.entity_name, str) and ';' in entity.entity_name:
            character_names = [name.strip() for name in entity.entity_name.split(';') if name.strip()]
            characters = []
            for name in character_names:
                char_entity = EntityLink(
                    entity_name=name,
                    best_appellation=name,
                    appellations=[name]
                )
                character = self._get_or_create_single_character(char_entity, series, session)
                if character:
                    characters.append(character)
            return characters[0] if characters else None
        else:
            return self._get_or_create_single_character(entity, series, session)

    def _get_or_create_single_character(self, entity: EntityLink, series: str, session: Session) -> Optional[Character]:
        """Internal method to get or create a single character."""
        # Skip invalid entity names
        if len(entity.entity_name.strip()) <= 1:
            logger.warning(f"Skipping invalid entity name: '{entity.entity_name}'")
            return None
            
        # Try to find existing character
        character = session.exec(
            select(Character).where(
                Character.entity_name == entity.entity_name,
                Character.series == series
            )
        ).first()

        if character:
            # Update existing character with new appellations
            existing_appellations = character.appellations
            new_appellations = entity.appellations
            all_appellations = list(set(existing_appellations + new_appellations))
            character.appellations = all_appellations
            character.best_appellation = entity.best_appellation
            logger.info(f"Updated existing character: {character.entity_name}")
        else:
            # Create new character
            character = Character(
                entity_name=entity.entity_name,
                best_appellation=entity.best_appellation,
                appellations=entity.appellations,
                series=series
            )
            session.add(character)
            session.flush()  # Ensure the character is in the session
            logger.info(f"Created new character: {character.entity_name}")

        return character

    def get_characters_by_names(self, character_names: List[str], series: str, session: Session) -> List[Character]:
        """Get characters by their entity names."""
        if not character_names:
            return []
            
        # Handle semicolon-separated strings
        processed_names = []
        for name in character_names:
            if ';' in name:
                processed_names.extend([n.strip() for n in name.split(';') if n.strip()])
            else:
                processed_names.append(name.strip())

        return session.exec(
            select(Character).where(
                Character.entity_name.in_(processed_names),
                Character.series == series
            )
        ).all()

    def get_characters_by_appellations(self, appellations: List[str], series: str, session: Session) -> List[Character]:
        """Get characters by their appellations."""
        characters = []
        for appellation in appellations:
            character = session.exec(
                select(Character).where(
                    Character.best_appellation == appellation,
                    Character.series == series
                )
            ).first()
            if character:
                characters.append(character)
        return characters
