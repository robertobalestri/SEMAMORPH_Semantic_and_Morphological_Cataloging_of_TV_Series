from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Union
from pydantic import BaseModel
from src.narrative_storage.repositories import DatabaseSessionManager
from src.narrative_storage.narrative_arc_service import NarrativeArcService
from src.narrative_storage.repositories import NarrativeArcRepository, ArcProgressionRepository, CharacterRepository
from src.narrative_storage.character_service import CharacterService
from src.narrative_storage.vector_store_service import VectorStoreService
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

class ArcUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    arc_type: Optional[str] = None
    main_characters: Optional[List[str]] = None

class ProgressionUpdateRequest(BaseModel):
    content: str
    interfering_characters: List[str]

class ProgressionCreateRequest(BaseModel):
    content: str
    arc_id: str
    series: str
    season: str
    episode: str
    interfering_characters: List[str]

class VectorStoreEntry(BaseModel):
    id: str
    content: str
    metadata: dict
    embedding: Optional[List[float]] = None
    distance: Optional[float] = None

class ProgressionMapping(BaseModel):
    season: str
    episode: str
    content: str
    interfering_characters: List[str]

class ArcMergeRequest(BaseModel):
    arc_id_1: str
    arc_id_2: str
    merged_title: str
    merged_description: str
    merged_arc_type: str
    main_characters: List[str]
    progression_mappings: List[ProgressionMapping]

def normalize_season_episode(season: str, episode: str) -> tuple[str, str]:
    """Normalize season and episode format to S01, E01."""
    # Remove any S/SS or E/EE prefix and leading zeros
    season_num = int(season.replace('S', '').replace('s', ''))
    episode_num = int(episode.replace('E', '').replace('e', ''))
    
    # Format with leading zeros
    return f"S{season_num:02d}", f"E{episode_num:02d}"

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
            logger.info(f"Fetching arcs for series: {series}")
            query = select(NarrativeArc).where(NarrativeArc.series == series)
            arcs = session.exec(query).all()
            logger.info(f"Found {len(arcs)} arcs")
            
            # Normalize season/episode format in progressions
            for arc in arcs:
                logger.info(f"Arc '{arc.title}' has {len(arc.progressions)} progressions")
                for prog in arc.progressions:
                    normalized_season, normalized_episode = normalize_season_episode(prog.season, prog.episode)
                    prog.season = normalized_season
                    prog.episode = normalized_episode
                    logger.info(f"  - Progression in {normalized_season}{normalized_episode}")
            
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
            
            # Convert to list of dicts and normalize format
            episode_list = [
                {
                    "season": normalized_season,
                    "episode": normalized_episode
                }
                for season, episode in episodes
                for normalized_season, normalized_episode in [normalize_season_episode(season, episode)]
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

@app.delete("/api/arcs/{arc_id}")
async def delete_arc(arc_id: str):
    """Delete a narrative arc and its progressions."""
    try:
        with db_manager.session_scope() as session:
            arc_repository = NarrativeArcRepository(session)
            progression_repository = ArcProgressionRepository(session)
            character_service = CharacterService(CharacterRepository(session))
            vector_store_service = VectorStoreService()
            
            narrative_arc_service = NarrativeArcService(
                arc_repository=arc_repository,
                progression_repository=progression_repository,
                character_service=character_service,
                llm_service=None,  # Not needed for deletion
                vector_store_service=vector_store_service,
                session=session
            )
            
            result = narrative_arc_service.delete_arc(arc_id)
            if not result:
                raise HTTPException(status_code=404, detail="Arc not found")
            return {"message": "Arc deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting arc: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/arcs/{arc_id}")
async def update_arc(arc_id: str, update_data: ArcUpdateRequest):
    """Update arc details."""
    try:
        with db_manager.session_scope() as session:
            arc_repository = NarrativeArcRepository(session)
            progression_repository = ArcProgressionRepository(session)
            character_service = CharacterService(CharacterRepository(session))
            vector_store_service = VectorStoreService()
            
            narrative_arc_service = NarrativeArcService(
                arc_repository=arc_repository,
                progression_repository=progression_repository,
                character_service=character_service,
                llm_service=None,  # Not needed for update
                vector_store_service=vector_store_service,
                session=session
            )
            
            updated_arc = narrative_arc_service.update_arc_details(
                arc_id=arc_id,
                title=update_data.title,
                description=update_data.description,
                arc_type=update_data.arc_type,
                main_characters=update_data.main_characters
            )
            if not updated_arc:
                raise HTTPException(status_code=404, detail="Arc not found")
            return NarrativeArcResponse.from_arc(updated_arc)
    except Exception as e:
        logger.error(f"Error updating arc: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/progressions/{progression_id}")
async def update_progression(progression_id: str, update_data: ProgressionUpdateRequest):
    """Update progression content and interfering characters."""
    try:
        with db_manager.session_scope() as session:
            arc_repository = NarrativeArcRepository(session)
            progression_repository = ArcProgressionRepository(session)
            character_service = CharacterService(CharacterRepository(session))
            vector_store_service = VectorStoreService()
            
            narrative_arc_service = NarrativeArcService(
                arc_repository=arc_repository,
                progression_repository=progression_repository,
                character_service=character_service,
                llm_service=None,
                vector_store_service=vector_store_service,
                session=session
            )
            
            updated_progression = narrative_arc_service.update_progression(
                progression_id=progression_id,
                content=update_data.content,
                interfering_characters=update_data.interfering_characters
            )
            if not updated_progression:
                raise HTTPException(status_code=404, detail="Progression not found")
            return ArcProgressionResponse.from_progression(updated_progression)
    except Exception as e:
        logger.error(f"Error updating progression: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/progressions", response_model=ArcProgressionResponse)
async def create_progression(progression_data: ProgressionCreateRequest):
    """Create a new progression for an arc."""
    try:
        with db_manager.session_scope() as session:
            arc_repository = NarrativeArcRepository(session)
            progression_repository = ArcProgressionRepository(session)
            character_service = CharacterService(CharacterRepository(session))
            vector_store_service = VectorStoreService()
            
            narrative_arc_service = NarrativeArcService(
                arc_repository=arc_repository,
                progression_repository=progression_repository,
                character_service=character_service,
                llm_service=None,
                vector_store_service=vector_store_service,
                session=session
            )
            
            new_progression = narrative_arc_service.add_progression(
                arc_id=progression_data.arc_id,
                content=progression_data.content,
                series=progression_data.series,
                season=progression_data.season,
                episode=progression_data.episode,
                interfering_characters=progression_data.interfering_characters
            )
            if not new_progression:
                raise HTTPException(status_code=404, detail="Arc not found")
            return ArcProgressionResponse.from_progression(new_progression)
    except Exception as e:
        logger.error(f"Error creating progression: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vector-store/{series}", response_model=List[VectorStoreEntry])
async def get_vector_store_entries(series: str, query: Optional[str] = None):
    """Get vector store entries for a series, optionally filtered by similarity to a query."""
    try:
        vector_store_service = VectorStoreService()
        
        if query:
            # If query provided, return similar documents
            results = vector_store_service.find_similar_documents(
                query=query,
                series=series,
                n_results=10,
                include_embeddings=True
            )
            return results
        else:
            # If no query, return all documents for the series
            results = vector_store_service.get_all_documents(
                series=series,
                include_embeddings=True
            )
            return results
    except Exception as e:
        logger.error(f"Error getting vector store entries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/arcs/merge", response_model=NarrativeArcResponse)
async def merge_arcs(merge_data: ArcMergeRequest):
    """Merge two arcs into a new one."""
    try:
        with db_manager.session_scope() as session:
            arc_repository = NarrativeArcRepository(session)
            progression_repository = ArcProgressionRepository(session)
            character_service = CharacterService(CharacterRepository(session))
            vector_store_service = VectorStoreService()
            
            narrative_arc_service = NarrativeArcService(
                arc_repository=arc_repository,
                progression_repository=progression_repository,
                character_service=character_service,
                llm_service=None,
                vector_store_service=vector_store_service,
                session=session
            )
            
            merged_arc = narrative_arc_service.merge_arcs(
                arc_id_1=merge_data.arc_id_1,
                arc_id_2=merge_data.arc_id_2,
                merged_title=merge_data.merged_title,
                merged_description=merge_data.merged_description,
                merged_arc_type=merge_data.merged_arc_type,
                main_characters=merge_data.main_characters,
                progression_mappings=[
                    {
                        "season": prog.season,
                        "episode": prog.episode,
                        "content": prog.content,
                        "interfering_characters": prog.interfering_characters
                    }
                    for prog in merge_data.progression_mappings
                ]
            )
            
            if not merged_arc:
                raise HTTPException(status_code=404, detail="One or both arcs not found")
            
            return NarrativeArcResponse.from_arc(merged_arc)
            
    except Exception as e:
        logger.error(f"Error merging arcs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
