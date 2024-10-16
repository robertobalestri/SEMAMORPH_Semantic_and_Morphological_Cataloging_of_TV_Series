from pydantic import BaseModel, Field
from typing import List, Optional


class ArcProgression(BaseModel):
    """Model representing a specific progression of a narrative arc in an episode."""
    content: str = Field(..., description="The progression content for this episode")
    series: str = Field(..., description="The series this progression belongs to")
    season: str = Field(..., description="The season this progression appears in")
    episode: str = Field(..., description="The episode this progression appears in")
    ordinal_position: Optional[int] = Field(None, description="The ordinal position of this progression in the arc")
    main_arc_id: Optional[str] = Field(None, description="The ID of the main arc this progression belongs to")


class NarrativeArc(BaseModel):
    """Model representing a narrative arc."""
    title: str = Field(..., description="The title of the narrative arc")
    arc_type: str = Field(..., description="Type of the arc such as 'Soap Arc'/'Genre-Specific Arc'/'Character Arc'/'Episodic Arc'/'Mythology Arc'")
    description: str = Field(..., description="A brief description of the narrative arc")
    duration: str = Field(..., description="Type of the arc: 'Episodic' or 'Seasonal'")
    characters: List[str] = Field(default_factory=list, description="Characters involved in this arc")
    series: str = Field(..., description="The series this arc belongs to")
    progressions: List[ArcProgression] = Field(default_factory=list, description="List of progressions for this arc")
    id: Optional[str] = Field(None, description="Unique identifier for the arc")

    class Config:
        populate_by_name = True
