from typing import List, Optional, Dict
from src.narrative_storage.repositories import DatabaseSessionManager
from src.narrative_storage.narrative_models import NarrativeArc, ArcProgression, Character
from src.narrative_storage.narrative_arc_service import NarrativeArcService
from src.narrative_storage.arc_progression_service import ArcProgressionService
from src.narrative_storage.character_service import CharacterService
from src.narrative_storage.llm_service import LLMService
from src.narrative_storage.vector_store_service import VectorStoreService
from sqlmodel import select
import uuid
from src.utils.logger_utils import setup_logging

logger = setup_logging(__name__)

class APIService:
    def __init__(self):
        self.db_manager = DatabaseSessionManager()
        self.llm_service = LLMService()
        self.vector_store_service = VectorStoreService()

    def _init_services(self, session):
        """Initialize all required services with the current session."""
        arc_repository = self.db_manager.get_arc_repository(session)
        progression_repository = self.db_manager.get_progression_repository(session)
        character_repository = self.db_manager.get_character_repository(session)
        
        character_service = CharacterService(character_repository)
        progression_service = ArcProgressionService(progression_repository, character_service)
        
        return (
            NarrativeArcService(
                arc_repository=arc_repository,
                progression_repository=progression_repository,
                character_service=character_service,
                llm_service=self.llm_service,
                vector_store_service=self.vector_store_service,
                session=session
            ),
            progression_service,
            character_service
        )

    def get_all_series(self) -> List[str]:
        """Get all unique series names."""
        with self.db_manager.session_scope() as session:
            arc_service, _, _ = self._init_services(session)
            return arc_service.get_all_series()

    def get_arcs_by_series(self, series: str) -> List[NarrativeArc]:
        """Get all narrative arcs for a specific series."""
        with self.db_manager.session_scope() as session:
            arc_service, _, _ = self._init_services(session)
            return arc_service.get_arcs_by_series(series)

    def get_episodes_by_series(self, series: str) -> List[Dict[str, str]]:
        """Get all episodes for a series."""
        with self.db_manager.session_scope() as session:
            _, progression_service, _ = self._init_services(session)
            return progression_service.get_episodes_by_series(series)

    def get_arcs_by_episode(self, series: str, season: str, episode: str) -> List[NarrativeArc]:
        """Get all narrative arcs that have progressions in a specific episode."""
        with self.db_manager.session_scope() as session:
            arc_service, _, _ = self._init_services(session)
            return arc_service.get_arcs_by_episode(series, season, episode)

    def merge_arcs(self, arc_ids: List[str], series: str) -> Dict[str, str]:
        """Merge multiple arcs into one."""
        with self.db_manager.session_scope() as session:
            arc_service, _, _ = self._init_services(session)
            target_arc_id = arc_ids[0]
            source_arc_ids = arc_ids[1:]
            
            # Get arcs
            target_arc = arc_service.get_arc_by_id(target_arc_id)
            source_arcs = [arc_service.get_arc_by_id(id) for id in source_arc_ids]
            
            if not all([target_arc] + source_arcs):
                raise ValueError("One or more arcs not found")
            
            # Use LLM to decide on merging
            merge_decision = self.llm_service.decide_arc_merging(source_arcs[0], target_arc)
            
            if merge_decision.get("same_arc", False):
                arc_service.merge_arcs(target_arc, source_arcs, merge_decision)
                return {"message": "Arcs merged successfully"}
            else:
                return {"message": "Arcs were determined to be different and were not merged"}

    def delete_arc(self, arc_id: str) -> Dict[str, str]:
        """Delete an arc and all its progressions."""
        with self.db_manager.session_scope() as session:
            arc_service, _, _ = self._init_services(session)
            arc = arc_service.get_arc_by_id(arc_id)
            if not arc:
                raise ValueError("Arc not found")
            
            arc_service.delete_arc(arc)
            return {"message": "Arc deleted successfully"}

    def add_progression(self, arc_id: str, progression_data: Dict) -> Dict[str, str]:
        """Add a new progression to an arc."""
        with self.db_manager.session_scope() as session:
            arc_service, progression_service, character_service = self._init_services(session)
            
            arc = arc_service.get_arc_by_id(arc_id)
            if not arc:
                raise ValueError("Arc not found")
            
            # Create new progression
            progression = ArcProgression(
                id=str(uuid.uuid4()),
                content=progression_data["content"],
                series=progression_data["series"],
                season=progression_data["season"],
                episode=f"E{progression_data['episode'].zfill(2)}",
                main_arc_id=arc_id
            )
            
            # Handle interfering characters
            if "interfering_characters" in progression_data:
                char_names = [
                    name.strip() 
                    for name in progression_data["interfering_characters"].split(";")
                    if name.strip()
                ]
                characters = character_service.get_characters_by_appellations(char_names, arc.series)
                character_service.link_characters_to_progression(characters, progression)
            
            progression_service.add_or_update_progression(arc, progression, arc.series, progression.season, progression.episode)
            return {"message": "Progression added successfully"}

    def update_progression(self, progression_id: str, progression_data: Dict) -> Dict[str, str]:
        """Update an existing progression."""
        with self.db_manager.session_scope() as session:
            _, progression_service, character_service = self._init_services(session)
            
            progression = progression_service.get_progression_by_id(progression_id)
            if not progression:
                raise ValueError("Progression not found")
            
            # Update fields
            progression.content = progression_data["content"]
            progression.episode = f"E{progression_data['episode'].zfill(2)}"
            
            # Update interfering characters
            if "interfering_characters" in progression_data:
                char_names = [
                    name.strip() 
                    for name in progression_data["interfering_characters"].split(";")
                    if name.strip()
                ]
                characters = character_service.get_characters_by_appellations(char_names, progression.series)
                progression.interfering_characters.clear()
                character_service.link_characters_to_progression(characters, progression)
            
            progression_service.add_or_update_progression(
                progression.narrative_arc,
                progression,
                progression.series,
                progression.season,
                progression.episode
            )
            return {"message": "Progression updated successfully"}