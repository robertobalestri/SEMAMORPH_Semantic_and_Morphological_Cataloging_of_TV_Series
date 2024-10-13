from pydantic import BaseModel, Field
from typing import List


class NarrativeArc(BaseModel):
    """Model representing a narrative arc."""
    Title: str = Field(..., description="The title of the narrative arc")
    Arc_Type: str = Field(..., description="Type of the arc such as 'Soap Arc'/'Genre-Specific Arc'/'Character Arc'/'Episodic Arc'/'Mythology Arc'")
    Description: str = Field(..., description="A brief description of the narrative arc")
    Progression: List[str] = Field(default_factory=list, description="Key progression points of the narrative arc")
    Duration: str = Field(..., description="Type of the arc: 'Episodic' or 'Seasonal'")
    Characters: List[str] = Field(default_factory=list, description="Characters involved in this arc")
