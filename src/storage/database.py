# database.py

from typing import List, Optional, TYPE_CHECKING
from sqlmodel import SQLModel, create_engine, select, Session
from sqlalchemy.orm import selectinload
from src.utils.logger_utils import setup_logging
from contextlib import contextmanager
from typing import Generator


from src.storage.narrative_arc_models import NarrativeArc, ArcProgression


logger = setup_logging(__name__)

class DatabaseManager:
    def __init__(self, db_url: str = 'sqlite:///narrative_arcs.db'):
        self.engine = create_engine(db_url, echo=False)
        SQLModel.metadata.create_all(self.engine)
        logger.info(f"Database engine initialized with URL: {db_url}")

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = Session(self.engine)
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session rollback due to exception: {e}")
            raise
        finally:
            session.close()

    # NarrativeArc Methods
    def add_or_update_narrative_arc(self, arc: "NarrativeArc", session: Optional[Session] = None):
        """Add a new NarrativeArc or update an existing one."""
        external_session = session is not None
        if not external_session:
            with self.session_scope() as session:
                self._add_or_update_narrative_arc(session, arc)
            return

        self._add_or_update_narrative_arc(session, arc)

    def _add_or_update_narrative_arc(self, session: Session, arc: "NarrativeArc"):
        """Internal method to add or update a NarrativeArc."""
        existing_arc = session.get(NarrativeArc, arc.id)
        if existing_arc:
            # Update existing arc
            existing_arc.title = arc.title
            existing_arc.description = arc.description
            existing_arc.arc_type = arc.arc_type
            existing_arc.episodic = arc.episodic
            existing_arc.characters = arc.characters
            existing_arc.series = arc.series
            session.add(existing_arc)
            logger.info(f"Updated existing arc: {arc.title}")
        else:
            # Add new arc
            session.add(arc)
            logger.info(f"Added new arc: {arc.title}")

    def get_narrative_arc_by_id(self, arc_id: str, session: Optional[Session] = None) -> Optional["NarrativeArc"]:
        """Retrieve a NarrativeArc by its ID."""
        if session is None:
            with self.session_scope() as session:
                return session.get(NarrativeArc, arc_id)
        return session.get(NarrativeArc, arc_id)

    def get_all_narrative_arcs(self, series: Optional[str] = None, session: Optional[Session] = None) -> List["NarrativeArc"]:
        """Retrieve all NarrativeArcs, optionally filtered by series."""
        if session is None:
            with self.session_scope() as session:
                return self._get_all_narrative_arcs(session, series)
        return self._get_all_narrative_arcs(session, series)

    def _get_all_narrative_arcs(self, session: Session, series: Optional[str] = None) -> List["NarrativeArc"]:
        """Internal method to retrieve all NarrativeArcs."""
        query = select(NarrativeArc).options(
            selectinload(NarrativeArc.progressions)  # Eager load progressions
        )
        if series:
            query = query.where(NarrativeArc.series == series)
        arcs = session.exec(query).all()
        logger.debug(f"Retrieved {len(arcs)} arcs from the database.")
        return arcs

    def get_narrative_arc_by_title(self, title: str, series: str, session: Optional[Session] = None) -> "NarrativeArc":
        """Retrieve NarrativeArcs by title and series."""
        if session is None:
            with self.session_scope() as session:
                return self._get_narrative_arc_by_title(session, title, series)
        return self._get_narrative_arc_by_title(session, title, series)

    def _get_narrative_arc_by_title(self, session: Session, title: str, series: str) -> Optional["NarrativeArc"]:
        """Internal method to retrieve NarrativeArcs by title and series."""
        query = select(NarrativeArc).where(
            NarrativeArc.title == title,
            NarrativeArc.series == series
        ).options(
            selectinload(NarrativeArc.progressions)
        )
        arc = session.exec(query).first()
        logger.info(f"Looking for arc with title '{title}' in series '{series}': {'Found' if arc else 'Not found'}")
        return arc

    # ArcProgression Methods
    def add_arc_progression(self, progression: "ArcProgression", session: Optional[Session] = None) -> bool:
        """Add a new ArcProgression if it doesn't already exist."""
        external_session = session is not None
        if not external_session:
            with self.session_scope() as session:
                return self._add_arc_progression(session, progression)
        return self._add_arc_progression(session, progression)

    def _add_arc_progression(self, session: Session, progression: "ArcProgression") -> bool:
        """Internal method to add a new ArcProgression."""
        # Check if the progression already exists
        existing_progression = session.exec(select(ArcProgression).where(
            ArcProgression.main_arc_id == progression.main_arc_id,
            ArcProgression.series == progression.series,
            ArcProgression.season == progression.season,
            ArcProgression.episode == progression.episode
        )).first()

        if existing_progression:
            # If it exists, update its content and ordinal position
            existing_progression.content = progression.content
            existing_progression.ordinal_position = progression.ordinal_position
            session.add(existing_progression)
            logger.info(f"Updated existing progression for arc {progression.main_arc_id} in episode {progression.series} S{progression.season}E{progression.episode}.")
        else:
            # If it doesn't exist, add the new progression
            session.add(progression)
            logger.info(f"Added new progression for arc {progression.main_arc_id} in episode {progression.series} S{progression.season}E{progression.episode}.")

        return True

    def get_arc_progressions(self, main_arc_id: str, session: Optional[Session] = None) -> List["ArcProgression"]:
        """Retrieve all ArcProgressions for a given NarrativeArc ID."""
        if session is None:
            with self.session_scope() as session:
                return self._get_arc_progressions(session, main_arc_id)
        return self._get_arc_progressions(session, main_arc_id)

    # database.py

    def _get_arc_progressions(self, session: Session, main_arc_id: str) -> List["ArcProgression"]:
        """Internal method to retrieve ArcProgressions."""
        query = select(ArcProgression).where(
            ArcProgression.main_arc_id == main_arc_id
        )
        progressions = session.exec(query).all() or []  # Ensure it's a list
        sorted_progressions = sorted(progressions, key=lambda p: p.ordinal_position or 0)
        logger.debug(f"Retrieved {len(sorted_progressions)} progressions for arc ID {main_arc_id}.")
        return sorted_progressions

    def get_single_arc_episode_progression(self, arc_id: str, series: str, season: str, episode: str, session: Session) -> Optional["ArcProgression"]:
        """Retrieve a specific ArcProgression based on arc ID, series, season, and episode."""
        query = select(ArcProgression).where(
            ArcProgression.main_arc_id == arc_id,
            ArcProgression.series == series,
            ArcProgression.season == season,
            ArcProgression.episode == episode
        )
        progression = session.exec(query).first()
        return progression
