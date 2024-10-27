from typing import List, Optional, Set
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy.types import String, JSON
from sqlalchemy import Column, Table, ForeignKey
import uuid

# Junction tables for many-to-many relationships
arc_characters = Table(
    "arc_characters",
    SQLModel.metadata,
    Column("arc_id", String, ForeignKey("narrativearc.id"), primary_key=True),
    Column("character_id", String, ForeignKey("character.id"), primary_key=True)
)

progression_characters = Table(
    "progression_characters",
    SQLModel.metadata,
    Column("progression_id", String, ForeignKey("arcprogression.id"), primary_key=True),
    Column("character_id", String, ForeignKey("character.id"), primary_key=True)
)

class Character(SQLModel, table=True):
    """Model representing a character in the series."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    entity_name: str = Field(index=True)
    best_appellation: str
    appellations: List[str] = Field(default=[], sa_column=Column(JSON))  # Fixed: using JSON type
    series: str = Field(index=True)

    # Relationships will be set after NarrativeArc and ArcProgression are defined
    narrative_arcs: List["NarrativeArc"] = []
    progressions: List["ArcProgression"] = []

    def __init__(self, **data):
        if isinstance(data.get('appellations'), str):
            data['appellations'] = [a.strip() for a in data['appellations'].split(",")]
        super().__init__(**data)

class ArcProgression(SQLModel, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)
    main_arc_id: str = Field(foreign_key="narrativearc.id")
    content: str
    series: str
    season: str
    episode: str
    ordinal_position: int = Field(default=0, nullable=False)
    
    interfering_episode_characters: List[str] = Field(default=[], sa_column=Column(JSON))  # Fixed: using JSON type
    characters: List[Character] = []  # Will be set after class definition
    narrative_arc: Optional["NarrativeArc"] = None  # Will be set after class definition

    def get_title(self, session) -> Optional[str]:
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
    series: str
    main_characters: List[str] = Field(default=[], sa_column=Column(JSON))  # Fixed: using JSON type
    
    characters: List[Character] = []  # Will be set after class definition
    progressions: List[ArcProgression] = []  # Will be set after class definition

    def __init__(self, **data):
        if 'id' not in data or data['id'] is None:
            data['id'] = str(uuid.uuid4())
        if isinstance(data.get('main_characters'), str):
            data['main_characters'] = [c.strip() for c in data['main_characters'].split(",")]
        super().__init__(**data)

# Set up relationships after all classes are defined
Character.narrative_arcs = Relationship(
    back_populates="characters",
    link_model=arc_characters,
    sa_relationship_kwargs={"lazy": "selectin"}
)

Character.progressions = Relationship(
    back_populates="characters",
    link_model=progression_characters,
    sa_relationship_kwargs={"lazy": "selectin"}
)

ArcProgression.characters = Relationship(
    back_populates="progressions",
    link_model=progression_characters,
    sa_relationship_kwargs={"lazy": "selectin"}
)

ArcProgression.narrative_arc = Relationship(
    back_populates="progressions",
    sa_relationship_kwargs={"lazy": "selectin"}
)

NarrativeArc.characters = Relationship(
    back_populates="narrative_arcs",
    link_model=arc_characters,
    sa_relationship_kwargs={"lazy": "selectin"}
)

NarrativeArc.progressions = Relationship(
    back_populates="narrative_arc",
    sa_relationship_kwargs={'cascade': 'all, delete-orphan', "lazy": "selectin"}
)
