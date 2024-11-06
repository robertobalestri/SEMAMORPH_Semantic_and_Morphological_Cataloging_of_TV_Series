# arc_progression_service.py

from typing import List, Optional
from src.narrative_storage.narrative_models import ArcProgression, NarrativeArc
from src.narrative_storage.repositories import ArcProgressionRepository
from src.narrative_storage.character_service import CharacterService
import logging
from sqlalchemy import func, select

logger = logging.getLogger(__name__)

class ArcProgressionService:
    """Service to manage arc progressions."""

    def __init__(
        self,
        progression_repository: ArcProgressionRepository,
        character_service: CharacterService
    ):
        self.progression_repository = progression_repository
        self.character_service = character_service
        self.session = progression_repository.session

    def add_or_update_progression(
        self,
        arc: NarrativeArc,
        progression: ArcProgression,
        series: str,
        season: str,
        episode: str
    ) -> ArcProgression:
        """Add or update a progression for an arc in a specific episode."""
        try:
            existing_progression = self.progression_repository.get_single(
                arc.id, series, season, episode
            )

            if existing_progression:
                # Update existing progression
                existing_progression.content = progression.content
                existing_progression.interfering_characters = progression.interfering_characters
                self.progression_repository.add_or_update(existing_progression)
                logger.info(f"Updated progression for arc '{arc.title}' in S{season}E{episode}")
                return existing_progression
            else:
                # Create new progression
                progression.main_arc_id = arc.id
                progression.series = series
                progression.season = season
                progression.episode = episode
                
                # Get next ordinal position within a transaction
                with self.session.begin_nested():
                    progression.ordinal_position = self._calculate_ordinal_position(arc)
                    arc.progressions.append(progression)
                    self.progression_repository.add_or_update(progression)
                
                logger.info(f"Added new progression for arc '{arc.title}' in S{season}E{episode} with ordinal position {progression.ordinal_position}")
                return progression

        except Exception as e:
            logger.error(f"Error in add_or_update_progression: {e}")
            raise

    def _calculate_ordinal_position(self, arc: NarrativeArc) -> int:
        """Calculate the ordinal position based on the total count of progressions for this arc."""
        # Get the maximum ordinal position for this arc
        max_position = self.session.query(func.max(ArcProgression.ordinal_position))\
            .filter(ArcProgression.main_arc_id == arc.id)\
            .scalar() or 0
        
        # Return next position
        return max_position + 1

    def get_arc_progression_count(self, arc_id: str) -> int:
        """Get the total number of progressions for an arc."""
        return self.session.exec(
            select(func.count(ArcProgression.id))
            .where(ArcProgression.main_arc_id == arc_id)
        ).scalar() or 0

    def resequence_ordinal_positions(self, arc_id: str):
        """Resequence ordinal positions to ensure they are consecutive."""
        progressions = self.progression_repository.get_by_arc_id(arc_id)
        for i, prog in enumerate(sorted(progressions, key=lambda p: p.ordinal_position), 1):
            if prog.ordinal_position != i:
                prog.ordinal_position = i
                self.progression_repository.add_or_update(prog)
