# repositories.py

from typing import List, Optional, Dict
from sqlmodel import SQLModel, Session, select
from sqlalchemy.orm import selectinload
from ..narrative_storage_management.narrative_models import NarrativeArc, ArcProgression, Character, CharacterAppellation, Event
from contextlib import contextmanager
import os
from sqlmodel import create_engine
from sqlalchemy import func, UniqueConstraint
from sqlalchemy.exc import IntegrityError

from ..utils.logger_utils import setup_logging
logger = setup_logging(__name__)

class DatabaseSessionManager:
    """Manages the database session."""

    def __init__(self, db_url: str = None):
        # If no db_url provided, try to get it from environment
        if db_url is None:
            db_url = self._get_database_url_from_env()
        
        if db_url is None:
            raise ValueError("No database URL provided and could not find DATABASE_NAME in environment variables")
        
        logger.info(f"🔗 Initializing DatabaseSessionManager with URL: {db_url}")
        
        self.engine = create_engine(
            db_url,
            echo=False,
            connect_args={"timeout": 60},
            pool_size=5,
            max_overflow=10
        )
        self.create_tables()

    def _get_database_url_from_env(self) -> str:
        """Get database URL from environment variables."""
        # Note: Environment variables should be loaded by the main application
        # (api_main_updated.py) before this is called
        
        # Get database name from environment variable
        database_name = os.getenv('DATABASE_NAME', 'narrative_storage/narrative.db')
        
        # Convert relative path to absolute path
        if not os.path.isabs(database_name):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
            database_name = os.path.join(project_root, database_name)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(database_name), exist_ok=True)
        
        db_url = f'sqlite:///{database_name}'
        logger.info(f"🔗 Constructed database URL from environment: {db_url}")
        logger.info(f"🔗 Initializing DatabaseSessionManager with URL: {db_url}")
        return db_url

    def create_tables(self):
        """Create database tables if they don't exist."""
        SQLModel.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get a new session."""
        return Session(self.engine)

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session rollback due to exception: {e}")
            raise
        finally:
            session.close()

class BaseRepository:
    """Base repository class with session management."""
    def __init__(self, session: Session):
        self.session = session

class NarrativeArcRepository(BaseRepository):
    """Repository for NarrativeArc operations."""

    def add_or_update(self, arc: NarrativeArc):
        existing_arc = self.session.get(NarrativeArc, arc.id)
        if existing_arc:
            existing_arc.title = arc.title
            existing_arc.description = arc.description
            existing_arc.arc_type = arc.arc_type
            existing_arc.series = arc.series
            logger.info(f"Updated NarrativeArc: {arc.title}")
        else:
            self.session.add(arc)
            logger.info(f"Added new NarrativeArc: {arc.title}")

    def update_fields(self, arc: NarrativeArc, updated_fields: Dict):
        """Update specific fields of the arc."""
        for field, value in updated_fields.items():
            if hasattr(arc, field):
                setattr(arc, field, value)
                logger.debug(f"Set {field} to {value} for arc ID {arc.id}")
            else:
                logger.warning(f"Field '{field}' does not exist on NarrativeArc.")
        self.session.add(arc)  # SQLAlchemy tracks changes

    def get_by_id(self, arc_id: str) -> Optional[NarrativeArc]:
        arc = self.session.get(NarrativeArc, arc_id)
        if arc:
            self.session.refresh(arc, ['main_characters', 'progressions'])
        return arc

    def get_by_title(self, title: str, series: str) -> Optional[NarrativeArc]:
        """Get a narrative arc by its exact title, case-insensitive."""
        return self.session.query(NarrativeArc)\
            .filter(func.lower(NarrativeArc.title) == title.lower())\
            .filter(NarrativeArc.series == series)\
            .first()

    def get_all(self, series: Optional[str] = None) -> List[NarrativeArc]:
        query = select(NarrativeArc).options(
            selectinload(NarrativeArc.main_characters),
            selectinload(NarrativeArc.progressions)
        )
        if series:
            query = query.where(NarrativeArc.series == series)
        return self.session.exec(query).all()

    def delete(self, arc_id: str):
        """Delete a narrative arc and its progressions."""
        arc = self.session.get(NarrativeArc, arc_id)
        if arc:
            self.session.delete(arc)
            logger.info(f"Deleted NarrativeArc: {arc.title}")

    def get_arcs_by_episode(self, series: str, season: str, episode: str) -> List[NarrativeArc]:
        """Get all narrative arcs that have progressions in a specific episode."""
        query = select(NarrativeArc)\
            .join(ArcProgression)\
            .where(
                NarrativeArc.series == series,
                ArcProgression.season == season,
                ArcProgression.episode == episode
            )\
            .options(
                selectinload(NarrativeArc.main_characters),
                selectinload(NarrativeArc.progressions)
            )
        return self.session.exec(query).all()

class ArcProgressionRepository(BaseRepository):
    """Repository for ArcProgression operations."""

    def add_or_update(self, progression: ArcProgression):
        existing_progression = self.session.exec(
            select(ArcProgression).where(
                ArcProgression.main_arc_id == progression.main_arc_id,
                ArcProgression.series == progression.series,
                ArcProgression.season == progression.season,
                ArcProgression.episode == progression.episode
            )
        ).first()

        if existing_progression:
            existing_progression.content = progression.content
            existing_progression.ordinal_position = progression.ordinal_position
            logger.info(f"Updated ArcProgression in S{progression.season}E{progression.episode}")
        else:
            self.session.add(progression)
            logger.info(f"Added new ArcProgression in S{progression.season}E{progression.episode}")

    def get_by_arc_id(self, arc_id: str) -> List[ArcProgression]:
        query = select(ArcProgression).where(
            ArcProgression.main_arc_id == arc_id
        ).options(
            selectinload(ArcProgression.narrative_arc),
            selectinload(ArcProgression.interfering_characters)
        )
        return self.session.exec(query).all()

    def get_single(self, arc_id: str, series: str, season: str, episode: str) -> Optional[ArcProgression]:
        query = select(ArcProgression).where(
            ArcProgression.main_arc_id == arc_id,
            ArcProgression.series == series,
            ArcProgression.season == season,
            ArcProgression.episode == episode
        ).options(
            selectinload(ArcProgression.interfering_characters)
        )
        return self.session.exec(query).first()

    def get_by_id(self, progression_id: str) -> Optional[ArcProgression]:
        """Get a progression by its ID."""
        return self.session.get(ArcProgression, progression_id)

    def delete(self, progression_id: str):
        """Delete a progression."""
        progression = self.session.get(ArcProgression, progression_id)
        if progression:
            self.session.delete(progression)
            logger.info(f"Deleted progression {progression_id}")

class CharacterRepository(BaseRepository):
    """Repository for Character operations."""

    def get_by_entity_name(self, entity_name: str, series: str) -> Optional[Character]:
        """Get a character by entity_name and series."""
        query = select(Character).where(
            Character.entity_name == entity_name,
            Character.series == series
        ).options(
            selectinload(Character.appellations),
            selectinload(Character.main_narrative_arcs),
            selectinload(Character.interfering_progressions)
        )
        return self.session.exec(query).first()

    def get_by_appellations(self, appellations: List[str], series: str) -> List[Character]:
        """Get characters by matching their appellations."""
        if not appellations:
            return []

        query = select(Character).join(CharacterAppellation).where(
            Character.series == series,
            CharacterAppellation.appellation.in_(appellations)
        ).options(
            selectinload(Character.appellations),
            selectinload(Character.main_narrative_arcs),
            selectinload(Character.interfering_progressions)
        ).distinct()

        return self.session.exec(query).all()

    def get_character_by_appellation(self, appellation: str, series: str) -> Optional[Character]:
        """Get a character by an appellation and series."""
        query = select(Character).join(CharacterAppellation).where(
            Character.series == series,
            CharacterAppellation.appellation == appellation
        ).options(
            selectinload(Character.appellations)
        )
        return self.session.exec(query).first()

    def add(self, character: Character):
        """Add a new character to the database."""
        self.session.add(character)
        self.session.flush()  # Ensure the character is persisted

    def update(self, character: Character):
        """Update an existing character in the database."""
        self.session.add(character)  # Add or update the character
        self.session.flush()

    def get_by_series(self, series: str) -> List[Character]:
        """Get all characters for a series."""
        query = select(Character).where(Character.series == series).options(
            selectinload(Character.appellations)
        )
        return self.session.exec(query).all()

    def delete(self, character: Character):
        """Delete a character from the database."""
        try:
            # Delete all appellations first
            for appellation in character.appellations:
                self.session.delete(appellation)
            
            # Remove character from all arcs and progressions
            for arc in character.main_narrative_arcs:
                arc.main_characters.remove(character)
            
            for progression in character.interfering_progressions:
                progression.interfering_characters.remove(character)
            
            # Finally delete the character
            self.session.delete(character)
            self.session.flush()
            logger.info(f"Successfully deleted character {character.entity_name}")
        except Exception as e:
            logger.error(f"Error deleting character {character.entity_name}: {e}")
            raise

class EventRepository:
    """Repository for Event objects."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, event_id: str) -> Optional[Event]:
        """Get an event by its ID."""
        query = select(Event).where(Event.id == event_id).options(
            selectinload(Event.progression)
        )
        return self.session.exec(query).first()

    def get_by_progression_id(self, progression_id: str) -> List[Event]:
        """Get all events for a specific progression, ordered by ordinal_position."""
        query = select(Event).where(
            Event.progression_id == progression_id
        ).order_by(Event.ordinal_position)
        return self.session.exec(query).all()

    def get_by_episode(self, series: str, season: str, episode: str, include_context: bool = False) -> List[Event]:
        """Get all events for a specific episode."""
        query = select(Event).where(
            Event.series == series,
            Event.season == season,
            Event.episode == episode
        )
        
        if include_context:
            # Load progression and narrative arc relationships
            query = query.options(
                selectinload(Event.progression).selectinload(ArcProgression.narrative_arc)
            )
        
        query = query.order_by(Event.ordinal_position)
        return self.session.exec(query).all()

    def get_by_timestamp_range(self, series: str, season: str, episode: str, 
                              start_time: str, end_time: str) -> List[Event]:
        """Get events within a specific timestamp range."""
        # Convert timestamp strings to seconds for comparison
        def timestamp_to_seconds(timestamp_str: str) -> float:
            if not timestamp_str:
                return 0.0
            try:
                time_part, ms_part = timestamp_str.split(',')
                hours, minutes, seconds = map(int, time_part.split(':'))
                milliseconds = int(ms_part)
                return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
            except:
                return 0.0
        
        start_seconds = timestamp_to_seconds(start_time)
        end_seconds = timestamp_to_seconds(end_time)
        
        # Get all events for the episode and filter by timestamp range
        all_events = self.session.exec(select(Event).where(
            Event.series == series,
            Event.season == season,
            Event.episode == episode
        )).all()
        
        # Filter events that fall within the timestamp range
        filtered_events = []
        for event in all_events:
            if event.start_timestamp and event.end_timestamp:
                event_start_seconds = timestamp_to_seconds(event.start_timestamp)
                event_end_seconds = timestamp_to_seconds(event.end_timestamp)
                
                # Check if event overlaps with the requested time range
                if (event_start_seconds >= start_seconds and event_start_seconds <= end_seconds) or \
                   (event_end_seconds >= start_seconds and event_end_seconds <= end_seconds) or \
                   (event_start_seconds <= start_seconds and event_end_seconds >= end_seconds):
                    filtered_events.append(event)
        
        # Sort by start timestamp
        filtered_events.sort(key=lambda e: timestamp_to_seconds(e.start_timestamp) if e.start_timestamp else 0)
        return filtered_events

    def create(self, event: Event) -> Event:
        """Create a new event."""
        self.session.add(event)
        self.session.flush()
        return event

    def update(self, event: Event) -> Event:
        """Update an existing event."""
        self.session.add(event)
        self.session.flush()
        return event

    def delete(self, event: Event):
        """Delete an event."""
        self.session.delete(event)
        self.session.flush()

    def delete_by_progression_id(self, progression_id: str):
        """Delete all events for a specific progression."""
        events = self.get_by_progression_id(progression_id)
        for event in events:
            self.session.delete(event)
        self.session.flush()

    def get_events_with_low_confidence(self, threshold: float = 0.5) -> List[Event]:
        """Get events with confidence scores below threshold."""
        query = select(Event).where(
            Event.confidence_score < threshold
        ).order_by(Event.confidence_score)
        return self.session.exec(query).all()

    def get_events_by_character(self, character_name: str, series: str) -> List[Event]:
        """Get all events involving a specific character."""
        # This would require joining with the event_characters table
        # For now, we'll use a simple text search in content
        query = select(Event).where(
            Event.series == series,
            Event.content.ilike(f"%{character_name}%")
        ).order_by(Event.start_timestamp)
        return self.session.exec(query).all()

    def get_statistics_by_episode(self, series: str, season: str, episode: str) -> Dict:
        """Get statistics about events for an episode."""
        events = self.get_by_episode(series, season, episode)
        
        if not events:
            return {
                "total_events": 0,
                "average_confidence": 0.0,
                "extraction_methods": {},
                "total_duration": 0.0
            }
        
        total_duration = 0.0
        extraction_methods = {}
        confidence_scores = []
        
        for event in events:
            if event.confidence_score:
                confidence_scores.append(event.confidence_score)
            
            if event.extraction_method:
                extraction_methods[event.extraction_method] = extraction_methods.get(event.extraction_method, 0) + 1
            
            if event.start_timestamp and event.end_timestamp:
                total_duration += (event.end_timestamp - event.start_timestamp)
        
        return {
            "total_events": len(events),
            "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
            "extraction_methods": extraction_methods,
            "total_duration": total_duration
        }