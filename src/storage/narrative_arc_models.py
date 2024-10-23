from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy.types import String
from sqlalchemy import Column
import uuid

class ArcProgression(SQLModel, table=True):
    """Model representing a specific progression of a narrative arc in an episode."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    content: str
    series: str
    season: str
    episode: str
    ordinal_position: Optional[int] = Field(default=None)
    
    # Relationship
    main_arc_id: Optional[str] = Field(default=None, foreign_key="narrativearc.id")
    narrative_arc: Optional["NarrativeArc"] = Relationship(back_populates="progressions")

    def get_title(self, session) -> Optional[str]:
        """Fetch the title of the associated NarrativeArc."""
        if self.main_arc_id:
            arc = session.get(NarrativeArc, self.main_arc_id)
            return arc.title if arc else None
        return None

class NarrativeArc(SQLModel, table=True):
    """Model representing a narrative arc."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    title: str
    arc_type: str
    description: str
    episodic: bool
    characters: str = Field(default="", sa_column=Column(String, default=""))
    series: str

    # Relationships
    progressions: List["ArcProgression"] = Relationship(
        back_populates="narrative_arc",
        sa_relationship_kwargs={'cascade': 'all, delete-orphan'}
    )

    def __init__(self, **data):
        if 'id' not in data or data['id'] is None:
            data['id'] = str(uuid.uuid4())
        super().__init__(**data)
        if isinstance(data.get('characters'), list):
            self.characters = ", ".join(data['characters'])

    def update_characters(self, new_characters: List[str]):
        """Update the characters attribute with a list of new characters."""
        self.characters = ", ".join(new_characters)
