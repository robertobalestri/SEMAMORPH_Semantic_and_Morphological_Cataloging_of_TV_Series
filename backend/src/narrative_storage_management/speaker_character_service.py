"""
Speaker character service for managing speaker-character database operations.
"""

import logging
from typing import List, Dict, Optional
from .repositories import DatabaseSessionManager, CharacterRepository
from .character_service import CharacterService
from .narrative_models import Character
from ..plot_processing.plot_processing_models import EntityLink

logger = logging.getLogger(__name__)

class SpeakerCharacterService:
    """Service for managing speaker-character relationships."""

    def __init__(self):
        self.db_manager = DatabaseSessionManager()
    
    def get_all_characters_as_data(self, series: str) -> List[Dict]:
        """
        Get all characters from database as plain data (no ORM objects).
        This avoids session binding issues.
        
        Args:
            series: The series to get characters for
            
        Returns:
            List of character data dictionaries with entity_name, best_appellation, appellations
        """
        with self.db_manager.session_scope() as session:
            character_service = CharacterService(CharacterRepository(session))
            db_characters = character_service.get_episode_entities(series)
            
            # Convert to plain data while session is active
            characters_data = []
            for char in db_characters:
                # Force load and extract data while session is active
                appellations_data = [app.appellation for app in char.appellations]
                
                # Create a simple data structure that doesn't depend on the session
                char_data = {
                    'entity_name': char.entity_name,
                    'best_appellation': char.best_appellation,
                    'appellations': appellations_data,
                    'series': char.series,
                    'biological_sex': char.biological_sex  # Include biological sex
                }
                characters_data.append(char_data)
        
        logger.info(f"📚 Loaded {len(characters_data)} characters from database for speaker validation")
        return characters_data
    
    def build_appellation_mapping(self, characters_data: List[Dict]) -> Dict[str, Dict]:
        """
        Build a mapping from all appellations to character data.
        
        Args:
            characters_data: List of character data dictionaries
            
        Returns:
            Dictionary mapping lowercase appellations to character data
        """
        mapping = {}
        for char_data in characters_data:
            # Add best appellation
            if char_data['best_appellation']:
                mapping[char_data['best_appellation'].lower()] = char_data
            
            # Add all appellations
            for appellation in char_data['appellations']:
                if appellation:
                    mapping[appellation.lower()] = char_data
                    
        return mapping
    
    def find_character_by_entity_name(self, entity_name: str, characters_data: List[Dict]) -> Optional[Dict]:
        """
        Find a character data dict by entity name.
        
        Args:
            entity_name: The entity name to search for
            characters_data: List of character data dictionaries
            
        Returns:
            Character data dict if found, None otherwise
        """
        for char_data in characters_data:
            if char_data["entity_name"] == entity_name:
                return char_data
        return None
    
    def add_appellation_to_character(self, entity_name: str, new_appellation: str, series: str) -> bool:
        """
        Add a new appellation to an existing character by entity name.
        
        Args:
            entity_name: The entity name of the character to update
            new_appellation: The new appellation to add
            series: The series the character belongs to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.db_manager.session_scope() as session:
                character_service = CharacterService(CharacterRepository(session))
                
                # Get the character fresh from the database within this session
                character = character_service.character_repository.get_by_entity_name(entity_name, series)
                if not character:
                    logger.error(f"❌ Character {entity_name} not found in database")
                    return False
                
                # Create EntityLink to use existing add_or_update logic
                appellations = [app.appellation for app in character.appellations]
                appellations.append(new_appellation)
                
                entity_link = EntityLink(
                    entity_name=character.entity_name,
                    best_appellation=character.best_appellation,
                    appellations=appellations,
                    biological_sex=character.biological_sex  # Preserve biological sex
                )
                
                result = character_service.add_or_update_character(entity_link, series)
                if result:
                    logger.info(f"✅ Added appellation '{new_appellation}' to character {entity_name}")
                    return True
                else:
                    logger.error(f"❌ Failed to add appellation '{new_appellation}' to character {entity_name}")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Error adding appellation to character: {e}")
            return False
    
    def create_new_characters(self, characters_data: List[Dict], series: str) -> bool:
        """
        Create new characters in the database.
        
        Args:
            characters_data: List of character data dictionaries to create
            series: The series these characters belong to
            
        Returns:
            True if all characters created successfully, False otherwise
        """
        if not characters_data:
            return True
            
        try:
            with self.db_manager.session_scope() as session:
                character_service = CharacterService(CharacterRepository(session))
                
                success_count = 0
                for char_data in characters_data:
                    entity_link = EntityLink(
                        entity_name=char_data["entity_name"],
                        best_appellation=char_data["best_appellation"],
                        appellations=char_data["appellations"],
                        biological_sex=char_data.get("biological_sex")  # Include biological sex
                    )
                    
                    result = character_service.add_or_update_character(entity_link, series)
                    if result:
                        logger.info(f"✅ Created new character: {char_data['best_appellation']}")
                        success_count += 1
                    else:
                        logger.warning(f"⚠️ Failed to create character: {char_data['best_appellation']}")
                
                logger.info(f"📊 Created {success_count}/{len(characters_data)} new characters")
                return success_count == len(characters_data)
                        
        except Exception as e:
            logger.error(f"❌ Error creating new characters: {e}")
            return False
    
    def build_characters_info_for_llm(self, characters_data: List[Dict]) -> List[Dict]:
        """
        Format character data for LLM prompts.
        
        Args:
            characters_data: List of character data dictionaries
            
        Returns:
            List of formatted character info for LLM
        """
        characters_info = []
        for char_data in characters_data:
            characters_info.append({
                "entity_name": char_data["entity_name"],
                "best_appellation": char_data["best_appellation"],
                "appellations": char_data["appellations"],
                "biological_sex": char_data.get("biological_sex")  # Include biological sex
            })
        return characters_info
