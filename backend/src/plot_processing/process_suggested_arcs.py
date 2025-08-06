# narrative_storage_main.py

from ..narrative_storage_management.repositories import (
    DatabaseSessionManager,
    NarrativeArcRepository,
    ArcProgressionRepository,
    CharacterRepository
)
from ..narrative_storage_management.character_service import CharacterService
from ..narrative_storage_management.narrative_arc_service import NarrativeArcService
from ..narrative_storage_management.llm_service import LLMService
from ..narrative_storage_management.vector_store_service import VectorStoreService
import logging
import json
from ..narrative_storage_management.narrative_models import NarrativeArc, ArcProgression
from typing import List
import uuid

from ..utils.logger_utils import setup_logging
logger = setup_logging(__name__)

def process_suggested_arcs(suggested_arcs_path: str, series: str, season: str, episode: str):
    # Initialize services
    db_manager = DatabaseSessionManager()
    llm_service = LLMService()
    vector_store_service = VectorStoreService()

    with db_manager.session_scope() as session:
        # Initialize repositories and services
        arc_repository = NarrativeArcRepository(session)
        progression_repository = ArcProgressionRepository(session)
        character_repository = CharacterRepository(session)
        character_service = CharacterService(character_repository)
        narrative_arc_service = NarrativeArcService(
            arc_repository=arc_repository,
            progression_repository=progression_repository,
            character_service=character_service,
            llm_service=llm_service,
            vector_store_service=vector_store_service,
            session=session
        )

        # Process arcs from JSON file
        with open(suggested_arcs_path, 'r') as f:
            suggested_arcs_data = json.load(f)

        logger.info(f"🔍 Processing {len(suggested_arcs_data)} suggested arcs from {suggested_arcs_path}")

        added_arcs = []
        for i, arc_data in enumerate(suggested_arcs_data):
            logger.info(f"🔄 Processing arc {i+1}/{len(suggested_arcs_data)}: '{arc_data.get('title', 'Unknown')}'")
            
            # Check if the arc has events
            events_data = arc_data.get('single_episode_progression_events', [])
            
            if events_data:
                events_with_timestamps = [
                    event for event in events_data 
                    if (hasattr(event, 'start_timestamp') and getattr(event, 'start_timestamp') is not None) or
                       (isinstance(event, dict) and event.get('start_timestamp') is not None)
                ]
                
                logger.info(f"   Arc has {len(events_data)} events, {len(events_with_timestamps)} with timestamps")
            else:
                logger.info(f"   Arc has no events")
            
            # Add the arc to the database - timestamps are handled by the main narrative arc graph
            added_arc = narrative_arc_service.add_arc(
                arc_data=arc_data,
                series=series,
                season=season,
                episode=episode
            )
            added_arcs.append(added_arc)
            logger.info(f"   ✅ Added arc '{added_arc.title}' to database")
        
        logger.info(f"✅ Processed {len(added_arcs)} arcs successfully")
        return added_arcs

if __name__ == "__main__":
    process_suggested_arcs('suggested_arcs.json', 'SeriesName', '1', '1')
