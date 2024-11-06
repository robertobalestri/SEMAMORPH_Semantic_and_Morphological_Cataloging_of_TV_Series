from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from src.narrative_storage.repositories import DatabaseSessionManager
from src.narrative_storage.narrative_models import NarrativeArc, ArcProgression
from src.utils.logger_utils import setup_logging
from sqlmodel import Session, select

logger = setup_logging(__name__)

app = FastAPI(title="Narrative Arcs Dashboard API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

db_manager = DatabaseSessionManager()

class ArcProgressionResponse(BaseModel):
    id: str
    content: str
    series: str
    season: str
    episode: str
    ordinal_position: int
    interfering_characters: List[str]

    class Config:
        from_attributes = True

    @classmethod
    def from_progression(cls, prog: ArcProgression):
        return cls(
            id=prog.id,
            content=prog.content,
            series=prog.series,
            season=prog.season,
            episode=prog.episode,
            ordinal_position=prog.ordinal_position,
            interfering_characters=[char.best_appellation for char in prog.interfering_characters]
        )

class NarrativeArcResponse(BaseModel):
    id: str
    title: str
    description: str
    arc_type: str
    main_characters: List[str]
    series: str
    progressions: List[ArcProgressionResponse]

    class Config:
        from_attributes = True

    @classmethod
    def from_arc(cls, arc: NarrativeArc):
        return cls(
            id=arc.id,
            title=arc.title,
            description=arc.description,
            arc_type=arc.arc_type,
            main_characters=[char.best_appellation for char in arc.main_characters],
            series=arc.series,
            progressions=[ArcProgressionResponse.from_progression(prog) for prog in sorted(
                arc.progressions,
                key=lambda x: (x.season, x.episode)
            )]
        )

@app.get("/api/series", response_model=List[str])
async def get_series():
    """Get all unique series names."""
    try:
        with db_manager.session_scope() as session:
            result = session.exec(select(NarrativeArc.series).distinct())
            return list(result)
    except Exception as e:
        logger.error(f"Error getting series: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/arcs/{series}", response_model=List[NarrativeArcResponse])
async def get_arcs_by_series(series: str):
    """Get all narrative arcs for a specific series."""
    try:
        with db_manager.session_scope() as session:
            query = select(NarrativeArc).where(NarrativeArc.series == series)
            arcs = session.exec(query).all()
            return [NarrativeArcResponse.from_arc(arc) for arc in arcs]
    except Exception as e:
        logger.error(f"Error getting arcs for series {series}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/episodes/{series}", response_model=List[dict])
async def get_episodes(series: str):
    """Get all episodes for a series."""
    try:
        with db_manager.session_scope() as session:
            query = select(ArcProgression.season, ArcProgression.episode)\
                .where(ArcProgression.series == series)\
                .distinct()
            episodes = session.exec(query).all()
            
            # Convert to list of dicts and sort
            episode_list = [
                {"season": season, "episode": episode.replace('E', '').zfill(2)}
                for season, episode in episodes
            ]
            return sorted(
                episode_list,
                key=lambda x: (x["season"], x["episode"])
            )
    except Exception as e:
        logger.error(f"Error getting episodes for series {series}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/arcs/{series}/{season}/{episode}", response_model=List[NarrativeArcResponse])
async def get_arcs_by_episode(series: str, season: str, episode: str):
    """Get all narrative arcs that have progressions in a specific episode."""
    try:
        normalized_episode = f"E{episode.zfill(2)}"
        logger.debug(f"Fetching arcs for {series} {season} {normalized_episode}")
        
        with db_manager.session_scope() as session:
            # Get arcs that have progressions in this episode
            query = select(NarrativeArc)\
                .join(ArcProgression)\
                .where(
                    ArcProgression.series == series,
                    ArcProgression.season == season,
                    ArcProgression.episode == normalized_episode
                )\
                .distinct()
            
            arcs = session.exec(query).all()
            
            if not arcs:
                logger.warning(f"No arcs found for {series} {season} {normalized_episode}")
                return []
            
            return [NarrativeArcResponse.from_arc(arc) for arc in arcs]
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
