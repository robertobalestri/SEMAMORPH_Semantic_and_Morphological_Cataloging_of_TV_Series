from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from src.storage.database import DatabaseManager
from src.storage.narrative_arc_models import NarrativeArc, ArcProgression
from src.utils.logger_utils import setup_logging
from sqlmodel import Session

logger = setup_logging(__name__)

app = FastAPI(title="Narrative Arcs Dashboard API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_manager = DatabaseManager()

class ArcProgressionResponse(BaseModel):
    id: str
    content: str
    series: str
    season: str
    episode: str
    ordinal_position: int
    interfering_episode_characters: str

class NarrativeArcResponse(BaseModel):
    id: str
    title: str
    description: str
    arc_type: str
    episodic: bool
    main_characters: str
    series: str
    progressions: List[ArcProgressionResponse]

@app.get("/api/series", response_model=List[str])
async def get_series():
    """Get all unique series names."""
    try:
        with db_manager.session_scope() as session:
            arcs = db_manager.get_all_narrative_arcs(session=session)
            # Convert to list before the session closes
            series_list = list(set(arc.series for arc in arcs))
            return series_list
    except Exception as e:
        logger.error(f"Error getting series: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/arcs/{series}", response_model=List[NarrativeArcResponse])
async def get_arcs_by_series(series: str):
    """Get all narrative arcs for a specific series."""
    try:
        with db_manager.session_scope() as session:
            arcs = db_manager.get_all_narrative_arcs(series=series, session=session)
            # Convert to response model before session closes
            return [NarrativeArcResponse.model_validate(arc.dict()) for arc in arcs]
    except Exception as e:
        logger.error(f"Error getting arcs for series {series}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/episodes/{series}", response_model=List[dict])
async def get_episodes(series: str):
    """Get all episodes for a series."""
    try:
        with db_manager.session_scope() as session:
            arcs = db_manager.get_all_narrative_arcs(series=series, session=session)
            episodes = set()
            for arc in arcs:
                for prog in arc.progressions:
                    # Normalize episode format: remove 'E' prefix and ensure 2 digits
                    episode_num = prog.episode.replace('E', '') if prog.episode.startswith('E') else prog.episode
                    episodes.add((prog.season, episode_num.zfill(2)))
            
            # Convert to list before session closes
            return [{"season": season, "episode": episode} 
                    for season, episode in sorted(episodes)]
    except Exception as e:
        logger.error(f"Error getting episodes for series {series}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/arcs/{series}/{season}/{episode}", response_model=List[NarrativeArcResponse])
async def get_arcs_by_episode(series: str, season: str, episode: str):
    """Get all narrative arcs that have progressions in a specific episode."""
    try:
        logger.debug(f"Fetching arcs for {series} {season} {episode}")
        
        # Normalize episode format
        normalized_episode = f"E{episode.zfill(2)}"
        logger.debug(f"Normalized episode: {normalized_episode}")
        
        with db_manager.session_scope() as session:
            arcs = db_manager.get_all_narrative_arcs(series=series, session=session)
            episode_arcs = []
            
            for arc in arcs:
                episode_progressions = [
                    prog for prog in arc.progressions
                    if (prog.series == series and 
                        prog.season == season and 
                        prog.episode == normalized_episode)
                ]
                if episode_progressions:
                    # Create a new dictionary with the arc data
                    arc_dict = arc.dict()
                    # Replace progressions with only the matching ones
                    arc_dict['progressions'] = [prog.dict() for prog in episode_progressions]
                    episode_arcs.append(arc_dict)
            
            if not episode_arcs:
                logger.warning(f"No arcs found for {series} {season} {normalized_episode}")
            
            # Convert to response model
            return [NarrativeArcResponse.model_validate(arc) for arc in episode_arcs]
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
