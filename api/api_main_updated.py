"""
Complete API implementation with direct processing (no subprocess).

This file includes ALL existing endpoints from api_main.py plus direct episode processing.
Run this instead of api_main.py to eliminate subprocess overhead.
"""

import sys
from pathlib import Path
import os

# Add the backend directory to Python path to ensure imports work correctly
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from fastapi import FastAPI, HTTPException, Body, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Union
from pydantic import BaseModel, ValidationError
from backend.src.narrative_storage_management.repositories import DatabaseSessionManager
from backend.src.narrative_storage_management.narrative_arc_service import NarrativeArcService
from backend.src.narrative_storage_management.repositories import NarrativeArcRepository, ArcProgressionRepository, CharacterRepository
from backend.src.narrative_storage_management.character_service import CharacterService
from backend.src.narrative_storage_management.vector_store_service import VectorStoreService
from backend.src.narrative_storage_management.narrative_models import NarrativeArc, ArcProgression, Character
from backend.src.utils.logger_utils import setup_logging
from sqlmodel import Session, select
from backend.src.plot_processing.plot_processing_models import EntityLink

from backend.src.narrative_storage_management.llm_service import LLMService
from backend.src.path_handler import PathHandler
import asyncio
import json
import pandas as pd
import time
from datetime import datetime
from enum import Enum
import logging
from typing import List, Optional, Dict, Union

# DIRECT PROCESSING IMPORTS (delayed imports to avoid credential issues at startup)
from backend.src.plot_processing.plot_text_processing import replace_pronouns_with_names
from backend.src.plot_processing.plot_ner_entity_extraction import (
    extract_and_refine_entities_with_path_handler, 
    substitute_appellations_with_names, 
    normalize_entities_names_to_best_appellation
)
from backend.src.plot_processing.subtitle_processing import (
    parse_srt_file, 
    generate_plot_from_subtitles, 
    save_plot_files,
    load_previous_season_summary,
    map_scenes_to_timestamps_with_boundary_correction,
    save_scene_timestamps,
    PlotScene
)
from backend.src.plot_processing.scene_timestamp_validator import (
    validate_and_fix_scene_timestamps,
    get_scene_coverage_report
)
from backend.src.plot_processing.plot_summarizing import create_or_update_season_summary
from backend.src.utils.text_utils import load_text
from backend.src.ai_models.ai_models import get_llm, LLMType
from backend.src.plot_processing.process_suggested_arcs import process_suggested_arcs
from backend.src.plot_processing.plot_summarizing import create_episode_summary
from backend.src.ai_models.ai_models import get_llm, LLMType
# Note: LangGraph and extract_narrative_arcs imported lazily in function

# SPEAKER IDENTIFICATION IMPORTS
from backend.src.subtitle_speaker_identification import run_speaker_identification_pipeline
from backend.src.config import config

# DIRECT PROCESSING FUNCTION (replaces subprocess)
async def process_episode_directly(series: str, season: str, episode: str, include_speaker_identification: bool = True) -> str:
    """
    Process episode directly without subprocess - this is the key improvement!
    """
    try:
        path_handler = PathHandler(series, season, episode)
        logger.info(f"🚀 DIRECT PROCESSING: {series} {season} {episode}")
        
        # Initialize LLMs
        llm_intelligent = get_llm(LLMType.INTELLIGENT)
        llm_cheap = get_llm(LLMType.CHEAP)
        
        # Step 1: Generate plot from SRT if needed
        raw_plot_path = path_handler.get_raw_plot_file_path()
        if not os.path.exists(raw_plot_path):
            srt_path = path_handler.get_srt_file_path()
            if not os.path.exists(srt_path):
                raise Exception(f"SRT file not found: {srt_path}")
            
            logger.info("📝 Generating plot from SRT")
            subtitles = parse_srt_file(srt_path)
            season_summary_path = path_handler.get_season_summary_path()
            previous_season_summary = load_previous_season_summary(season_summary_path)
            plot_data = generate_plot_from_subtitles(subtitles, llm_intelligent, previous_season_summary)
            
            episode_prefix = f"{series}{season}{episode}"
            episode_dir = os.path.dirname(raw_plot_path)
            save_plot_files(plot_data, episode_dir, episode_prefix)
        
        # Step 1.2: Map scenes to timestamps with boundary correction
        timestamps_path = path_handler.get_scene_timestamps_path()
        if not os.path.exists(timestamps_path):
            logger.info("🕒 Mapping scenes to timestamps with boundary correction")
            try:
                # Load or parse SRT and plot data
                srt_path = path_handler.get_srt_file_path()
                if not os.path.exists(srt_path):
                    raise Exception(f"SRT file not found: {srt_path}")
                
                subtitles = parse_srt_file(srt_path)
                
                # Load plot scenes
                scenes_json_path = path_handler.get_plot_scenes_json_path()
                if not os.path.exists(scenes_json_path):
                    raise Exception(f"Plot scenes JSON not found: {scenes_json_path}")
                
                with open(scenes_json_path, 'r') as f:
                    plot_data = json.load(f)
                
                # Convert to PlotScene objects
                scenes = []
                for scene_data in plot_data.get("scenes", []):
                    scene = PlotScene(
                        scene_number=scene_data.get("scene_number", len(scenes) + 1),
                        plot_segment=scene_data.get("plot_segment", "")
                    )
                    scenes.append(scene)
                
                # Map scenes to timestamps with boundary correction
                mapped_scenes = map_scenes_to_timestamps_with_boundary_correction(scenes, subtitles, llm_cheap)
                
                # Save scene timestamps
                episode_prefix = f"{series}{season}{episode}"
                episode_dir = os.path.dirname(timestamps_path)
                saved_path = save_scene_timestamps(mapped_scenes, episode_dir, episode_prefix)
                
                logger.info(f"✅ Scene timestamps with boundary correction saved to: {saved_path}")
                
            except Exception as e:
                logger.error(f"❌ Scene timestamp mapping failed: {e}")
                # Non-critical error, continue processing
        
        # Step 1.5: Generate episode summary early (MOVED HERE)
        episode_summary_path = path_handler.get_episode_summary_path()
        if not os.path.exists(episode_summary_path):
            logger.info("📖 Generating episode summary")
            create_episode_summary(raw_plot_path, llm_intelligent, episode_summary_path)
        
        # Step 2: Process pronouns and entities
        named_file_path = path_handler.get_named_plot_file_path()
        if not os.path.exists(named_file_path):
            logger.info("🔄 Processing pronouns")
            raw_plot = load_text(raw_plot_path)
            named_plot = replace_pronouns_with_names(text=raw_plot, intelligent_llm=llm_intelligent, cheap_llm=llm_cheap)
            with open(named_file_path, "w") as f:
                f.write(named_plot)
        
        # Step 3: Extract entities
        episode_entities_path = path_handler.get_episode_refined_entities_path()
        if not os.path.exists(episode_entities_path):
            logger.info("🎭 Extracting entities")
            entities = extract_and_refine_entities_with_path_handler(path_handler, series)
        else:
            with open(episode_entities_path, "r") as f:
                entities_data = json.load(f)
                entities = [EntityLink(**entity) for entity in entities_data]



        # Step 4: Entity processing
        entity_substituted_path = path_handler.get_entity_substituted_plot_file_path()
        entity_normalized_path = path_handler.get_entity_normalized_plot_file_path()
        
        with open(named_file_path, "r") as f:
            named_plot = f.read()
        
        if not os.path.exists(entity_substituted_path):
            logger.info("🔀 Substituting entities")
            entity_substituted_plot = substitute_appellations_with_names(named_plot, entities, llm_intelligent)
            with open(entity_substituted_path, "w") as f:
                f.write(entity_substituted_plot)
        
        if not os.path.exists(entity_normalized_path):
            logger.info("📊 Normalizing entities")
            with open(entity_substituted_path, "r") as f:
                entity_substituted_plot = f.read()
            entity_normalized_plot = normalize_entities_names_to_best_appellation(entity_substituted_plot, entities)
            with open(entity_normalized_path, "w") as f:
                f.write(entity_normalized_plot)
        
        # Step 5: Speaker Identification and Face Clustering (MOVED EARLIER)
        if include_speaker_identification:
            logger.info("🎭 Running speaker identification and face clustering")
            try:
                speaker_results = run_speaker_identification_pipeline(
                    series=series,
                    season=season,
                    episode=episode,
                    base_dir="data",
                    force_regenerate=False,
                    # Boolean confidence approach - no threshold needed
                    face_similarity_threshold=config.cosine_similarity_threshold,
                    embedding_model=config.face_embedding_model,
                    face_detector=config.face_detector
                )
                
                # Log speaker identification results
                overall_stats = speaker_results.get('overall_stats', {})
                logger.info(f"🎭 Speaker ID results: {overall_stats.get('high_confidence_dialogue', 0)}/{overall_stats.get('total_dialogue_lines', 0)} confident speakers ({overall_stats.get('final_confidence_rate', 0):.1f}%)")
                
            except Exception as e:
                logger.error(f"❌ Speaker identification failed: {e}")
                # Continue processing even if speaker identification fails


        time.sleep(10000000)
        
        # Step 6: Update season summary
        logger.info("📖 Updating season summary")
        episode_plot_path = path_handler.get_raw_plot_file_path()
        season_summary_path = path_handler.get_season_summary_path()
        episode_summary_path = path_handler.get_episode_summary_path()
        create_or_update_season_summary(episode_plot_path, season_summary_path, episode_summary_path, llm_intelligent)
        
        # Step 7: Extract narrative arcs (MOVED TO END)
        logger.info("📚 Starting narrative arcs extraction...")
        
        suggested_arc_path = path_handler.get_suggested_episode_arc_path()
        if not os.path.exists(suggested_arc_path):
            logger.info("📚 Extracting narrative arcs")
            # Import here to avoid credential issues at startup
            from backend.src.langgraph_narrative_arcs_extraction.narrative_arc_graph import extract_narrative_arcs
            
            file_paths_for_graph = {
                "episode_plot_path": path_handler.get_raw_plot_file_path(),
                "seasonal_narrative_analysis_output_path": path_handler.get_season_narrative_analysis_path(),
                "episode_narrative_analysis_output_path": path_handler.get_episode_narrative_analysis_path(),
                "season_entities_path": path_handler.get_season_extracted_refined_entities_path(),
                "suggested_episode_arc_path": suggested_arc_path
            }
            extract_narrative_arcs(file_paths_for_graph, series, season, episode)
        
        # Step 8: Process arcs and update database
        logger.info("💾 Updating database")
        updated_arcs = process_suggested_arcs(suggested_arc_path, series, season, episode)
        
        logger.info(f"✅ COMPLETED: {series} {season} {episode}")
        return f"Successfully processed {len(updated_arcs)} arcs"
        
    except Exception as e:
        logger.error(f"❌ PROCESSING FAILED: {series} {season} {episode}: {e}")
        raise

# Set up logging
logger = setup_logging(__name__)
root_logger = setup_logging("root")

# Disable uvicorn access log to avoid duplicate logging
import logging
logging.getLogger("uvicorn.access").handlers = []

app = FastAPI(title="Narrative Arcs Dashboard API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:5173"   # Vite dev server
    ],
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
    interfering_characters: Union[str, List[str]]

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

class InitialProgressionData(BaseModel):
    content: str
    season: str
    episode: str
    interfering_characters: List[str]

class ArcCreateRequest(BaseModel):
    title: str
    description: str
    arc_type: str
    main_characters: str
    series: str
    initial_progression: Optional[InitialProgressionData] = None

class CharacterResponse(BaseModel):
    entity_name: str
    best_appellation: str
    series: str
    appellations: List[str]

    class Config:
        from_attributes = True

    @classmethod
    def from_character(cls, character: Character):
        return cls(
            entity_name=character.entity_name,
            best_appellation=character.best_appellation,
            series=character.series,
            appellations=[app.appellation for app in character.appellations]
        )

class CharacterCreateRequest(BaseModel):
    entity_name: str
    best_appellation: str
    series: str
    appellations: List[str]

    class Config:
        from_attributes = True

class CharacterMergeRequest(BaseModel):
    character1_id: str
    character2_id: str
    keep_character: str

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessingJob(BaseModel):
    id: str
    series: str
    season: str
    episodes: List[str]
    status: ProcessingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: Dict[str, str] = {}

class ProcessingRequest(BaseModel):
    series: str
    season: str
    episodes: List[str]
    include_speaker_identification: bool = True  # New parameter to control speaker ID

# SPEAKER IDENTIFICATION MODELS
class SpeakerIdentificationRequest(BaseModel):
    series: str
    season: str
    episode: str
    force_regenerate: bool = False
    face_similarity_threshold: float = config.cosine_similarity_threshold
    embedding_model: str = config.face_embedding_model
    face_detector: str = config.face_detector

# FACE PROCESSING MODELS  
class FaceExtractionRequest(BaseModel):
    series: str
    season: str
    episode: str
    force_extract: bool = False
    detector: str = config.face_detector
    min_confidence: float = config.face_min_confidence
    min_face_area_ratio: float = config.face_min_area_ratio
    blur_threshold: float = config.face_blur_threshold
    enable_eye_validation: bool = config.face_enable_eye_validation
    eye_alignment_threshold: float = config.face_eye_alignment_threshold
    eye_distance_threshold: float = config.face_eye_distance_threshold

class FaceEmbeddingRequest(BaseModel):
    series: str
    season: str
    episode: str
    force_regenerate: bool = False
    embedding_model: str = config.face_embedding_model

class FaceProcessingResponse(BaseModel):
    episode_code: str
    status: str
    faces_extracted: int
    embeddings_generated: int
    processing_time_seconds: Optional[float] = None
    error_message: Optional[str] = None

class SpeakerIdentificationResponse(BaseModel):
    episode_code: str
    status: str
    total_dialogue_lines: int
    faces_extracted: int
    speakers_identified: int
    confident_dialogue: int
    final_confidence_rate: float
    processing_time_seconds: Optional[float] = None
    error_message: Optional[str] = None

class DialogueLineResponse(BaseModel):
    index: int
    start_time: float
    end_time: float
    text: str
    speaker: Optional[str] = None
    speaker_confidence: Optional[float] = None
    scene_number: Optional[int] = None

class FaceDataResponse(BaseModel):
    dialogue_index: int
    face_index: int
    timestamp_seconds: float
    speaker: Optional[str] = None
    speaker_confidence: Optional[float] = None
    detection_confidence: float
    blur_score: float
    image_path: str

class SpeakerStatsResponse(BaseModel):
    episode_code: str
    total_dialogue_lines: int
    total_faces_extracted: int
    speakers_found: List[str]
    confidence_distribution: Dict[str, int]  # ranges like "90-95%": count
    face_cluster_stats: Dict[str, int]  # speaker -> face count

# SCENE VALIDATION MODELS
class SceneValidationRequest(BaseModel):
    series: str
    season: str
    episode: str
    fix_issues: bool = True  # Whether to attempt fixing issues found

class SceneCoverageGap(BaseModel):
    gap_type: str  # "MISSING_SUBTITLES", "OVERLAP", "OUT_OF_ORDER"
    start_seconds: float
    end_seconds: float
    affected_scenes: List[int]
    missing_subtitle_indices: Optional[List[int]] = None
    description: str

class SceneValidationResponse(BaseModel):
    episode_code: str
    is_valid: bool
    coverage_percentage: float
    total_subtitles: int
    covered_subtitles: int
    gaps_found: int
    gap_details: List[SceneCoverageGap]
    corrections_applied: bool = False
    error_message: Optional[str] = None

# In-memory storage for processing jobs
processing_jobs: Dict[str, ProcessingJob] = {}

def normalize_season_episode(season: str, episode: str) -> tuple[str, str]:
    """Normalize season and episode format to S01, E01."""
    season_num = int(season.replace('S', '').replace('s', ''))
    episode_num = int(episode.replace('E', '').replace('e', ''))
    return f"S{season_num:02d}", f"E{episode_num:02d}"

def pad_number(num_str: str) -> str:
    """Pad a number string to at least 2 digits."""
    if len(num_str) == 1:
        return f"0{num_str}"
    return num_str

# ALL EXISTING ENDPOINTS (copied from api_main.py)

@app.get("/api/series", response_model=List[str])
async def get_series():
    """Get all unique series names from both database and filesystem."""
    try:
        series_set = set()
        
        # Get series from database
        try:
            with db_manager.session_scope() as session:
                db_result = session.exec(select(NarrativeArc.series).distinct())
                series_set.update(db_result)
        except Exception as db_error:
            logger.warning(f"Could not fetch series from database: {db_error}")
        
        # Get series from filesystem
        try:
            data_dir = "data"
            if os.path.exists(data_dir):
                for item in os.listdir(data_dir):
                    item_path = os.path.join(data_dir, item)
                    if os.path.isdir(item_path) and not item.startswith('.'):
                        series_set.add(item)
        except Exception as fs_error:
            logger.warning(f"Could not scan filesystem for series: {fs_error}")
        
        # If no series found from database, use filesystem only
        if not series_set:
            logger.info("No series found in database, using filesystem only")
        
        return sorted(list(series_set))
        
    except Exception as e:
        logger.error(f"Error getting series: {str(e)}")
        # Return empty list if everything fails - let the frontend handle it
        return []

@app.get("/api/arcs/series/{series}", response_model=List[NarrativeArcResponse])
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
                for prog in arc.progressions:
                    normalized_season, normalized_episode = normalize_season_episode(prog.season, prog.episode)
                    prog.season = normalized_season
                    prog.episode = normalized_episode
            
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

# NEW PROCESSING ENDPOINTS (no subprocess - direct processing!)

@app.post("/api/processing/episodes", response_model=ProcessingJob)
async def start_episode_processing(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Start processing episodes using DIRECT processing (no subprocess)"""
    try:
        # Generate job ID
        job_id = f"{request.series}_{request.season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create job
        job = ProcessingJob(
            id=job_id,
            series=request.series,
            season=request.season,
            episodes=request.episodes,
            status=ProcessingStatus.PENDING,
            created_at=datetime.now(),
            progress={ep: "pending" for ep in request.episodes}
        )
        
        processing_jobs[job_id] = job
        
        # Start background processing using DIRECT processing
        background_tasks.add_task(
            run_episode_processing_direct,
            job_id,
            request.series,
            request.season,
            request.episodes,
            request.include_speaker_identification
        )
        
        logger.info(f"🚀 Started DIRECT processing job {job_id} for {request.series} {request.season}")
        return job
        
    except Exception as e:
        logger.error(f"Error starting processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_episode_processing_direct(job_id: str, series: str, season: str, episodes: List[str], include_speaker_identification: bool = True):
    """Run episode processing using DIRECT function calls (no subprocess!)"""
    job = processing_jobs[job_id]
    
    try:
        job.status = ProcessingStatus.RUNNING
        job.started_at = datetime.now()
        
        # Process each episode using DIRECT processing
        for episode in episodes:
            job.progress[episode] = "processing"
            
            try:
                # Call the direct processing function
                result = await process_episode_directly(series, season, episode, include_speaker_identification)
                
                job.progress[episode] = "completed"
                logger.info(f"✅ DIRECT processing completed: {series} {season} {episode}")
                
            except Exception as episode_error:
                job.progress[episode] = "failed"
                logger.error(f"❌ DIRECT processing failed: {series} {season} {episode}: {episode_error}")
                # Continue with other episodes even if one fails
        
        # Check if all episodes completed successfully
        failed_episodes = [ep for ep, status in job.progress.items() if status == "failed"]
        if failed_episodes:
            job.status = ProcessingStatus.FAILED
            job.error_message = f"Failed to process episodes: {', '.join(failed_episodes)}"
        else:
            job.status = ProcessingStatus.COMPLETED
        
        job.completed_at = datetime.now()
        logger.info(f"🎉 DIRECT processing job {job_id} completed with status: {job.status}")
        
    except Exception as e:
        job.status = ProcessingStatus.FAILED
        job.error_message = str(e)
        job.completed_at = datetime.now()
        logger.error(f"❌ DIRECT processing job {job_id} failed: {e}")

@app.get("/api/processing/jobs", response_model=List[ProcessingJob])
async def get_processing_jobs():
    """Get all processing jobs"""
    return list(processing_jobs.values())

@app.get("/api/processing/jobs/{job_id}", response_model=ProcessingJob)
async def get_processing_job(job_id: str):
    """Get a specific processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return processing_jobs[job_id]

@app.delete("/api/processing/jobs/{job_id}")
async def delete_processing_job(job_id: str):
    """Delete a processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job.status == ProcessingStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot delete running job")
    
    del processing_jobs[job_id]
    return {"message": "Job deleted successfully"}

@app.get("/api/available-episodes/{series}/{season}", response_model=List[str])
async def get_available_episodes(series: str, season: str):
    """Get list of available episodes for processing"""
    try:
        logger.info(f"Fetching available episodes for {series}/{season}")
        
        # Use PathHandler to find available episodes
        episode_folders = PathHandler.list_episode_folders("data", series, season)
        
        logger.info(f"Found {len(episode_folders)} episodes: {episode_folders}")
        
        # Filter out any invalid episode names and ensure proper format
        valid_episodes = []
        for episode in episode_folders:
            # Ensure episode starts with 'E' and is followed by digits
            if episode.startswith('E') and episode[1:].isdigit():
                valid_episodes.append(episode)
            else:
                logger.warning(f"Skipping invalid episode folder: {episode}")
        
        logger.info(f"Valid episodes: {valid_episodes}")
        return sorted(valid_episodes)
        
    except Exception as e:
        logger.error(f"Error getting available episodes for {series}/{season}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/available-series", response_model=List[str])
async def get_available_series():
    """Get all series available for processing from filesystem."""
    try:
        series_list = []
        data_dir = "data"
        
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    # Check if this directory has season subdirectories
                    has_seasons = False
                    try:
                        for subitem in os.listdir(item_path):
                            if os.path.isdir(os.path.join(item_path, subitem)) and subitem.startswith('S'):
                                has_seasons = True
                                break
                    except:
                        pass
                    
                    if has_seasons:
                        series_list.append(item)
        
        return sorted(series_list)
        
    except Exception as e:
        logger.error(f"Error getting available series: {str(e)}")
        return []

# SPEAKER IDENTIFICATION ENDPOINTS

@app.post("/api/speaker-identification/run", response_model=SpeakerIdentificationResponse)
async def run_speaker_identification(request: SpeakerIdentificationRequest):
    """Run speaker identification pipeline for a specific episode."""
    try:
        start_time = time.time()
        
        logger.info(f"🎭 Starting speaker identification for {request.series}{request.season}{request.episode}")
        
        # Run the speaker identification pipeline
        results = run_speaker_identification_pipeline(
            series=request.series,
            season=request.season,
            episode=request.episode,
            base_dir="data",
            force_regenerate=request.force_regenerate,
            face_similarity_threshold=request.face_similarity_threshold,
            embedding_model=request.embedding_model,
            face_detector=request.face_detector
        )
        
        processing_time = time.time() - start_time
        
        # Extract relevant stats
        overall_stats = results.get('overall_stats', {})
        pipeline_steps = results.get('pipeline_steps', {})
        
        # Check for errors
        failed_steps = [step for step, data in pipeline_steps.items() 
                       if isinstance(data, dict) and data.get('status') == 'failed']
        
        if failed_steps:
            error_msg = f"Failed steps: {', '.join(failed_steps)}"
            return SpeakerIdentificationResponse(
                episode_code=results.get('episode_code', f"{request.series}{request.season}{request.episode}"),
                status="failed",
                total_dialogue_lines=overall_stats.get('total_dialogue_lines', 0),
                faces_extracted=overall_stats.get('faces_extracted', 0),
                speakers_identified=overall_stats.get('speakers_identified', 0),
                confident_dialogue=overall_stats.get('confident_dialogue', 0),
                final_confidence_rate=overall_stats.get('final_confidence_rate', 0.0),
                processing_time_seconds=processing_time,
                error_message=error_msg
            )
        
        return SpeakerIdentificationResponse(
            episode_code=results.get('episode_code', f"{request.series}{request.season}{request.episode}"),
            status="success",
            total_dialogue_lines=overall_stats.get('total_dialogue_lines', 0),
            faces_extracted=overall_stats.get('faces_extracted', 0),
            speakers_identified=overall_stats.get('speakers_identified', 0),
            confident_dialogue=overall_stats.get('high_confidence_dialogue', 0),
            final_confidence_rate=overall_stats.get('final_confidence_rate', 0.0),
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Speaker identification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/speaker-identification/dialogue/{series}/{season}/{episode}", response_model=List[DialogueLineResponse])
async def get_dialogue_with_speakers(series: str, season: str, episode: str):
    """Get dialogue lines with speaker information for an episode."""
    try:
        from backend.src.subtitle_speaker_identification.srt_parser import SRTParser
        
        path_handler = PathHandler(series, season, episode)
        dialogue_json_path = path_handler.get_dialogue_json_path().replace('.json', '_final.json')
        
        # Check if final dialogue file exists
        if not os.path.exists(dialogue_json_path):
            # Fall back to regular dialogue file
            dialogue_json_path = path_handler.get_dialogue_json_path()
            
        if not os.path.exists(dialogue_json_path):
            raise HTTPException(status_code=404, detail="Dialogue data not found. Run speaker identification first.")
        
        # Load dialogue data
        with open(dialogue_json_path, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)
        
        dialogue_lines = dialogue_data.get('dialogue_lines', [])
        
        return [
            DialogueLineResponse(
                index=line.get('index', 0),
                start_time=line.get('start_time', 0.0),
                end_time=line.get('end_time', 0.0),
                text=line.get('text', ''),
                speaker=line.get('speaker'),
                speaker_confidence=line.get('speaker_confidence'),
                scene_number=line.get('scene_number')
            )
            for line in dialogue_lines
        ]
        
    except Exception as e:
        logger.error(f"❌ Error getting dialogue: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/speaker-identification/faces/{series}/{season}/{episode}", response_model=List[FaceDataResponse])
async def get_face_data(series: str, season: str, episode: str):
    """Get face detection data for an episode."""
    try:
        import pandas as pd
        
        path_handler = PathHandler(series, season, episode)
        faces_csv_path = path_handler.get_dialogue_faces_csv_path()
        
        if not os.path.exists(faces_csv_path):
            raise HTTPException(status_code=404, detail="Face data not found. Run speaker identification first.")
        
        # Load face data
        df_faces = pd.read_csv(faces_csv_path)
        
        return [
            FaceDataResponse(
                dialogue_index=int(row['dialogue_index']),
                face_index=int(row['face_index']),
                timestamp_seconds=float(row['timestamp_seconds']),
                speaker=row.get('speaker') if pd.notna(row.get('speaker')) else None,
                speaker_confidence=float(row['speaker_confidence']) if pd.notna(row.get('speaker_confidence')) else None,
                detection_confidence=float(row['detection_confidence']),
                blur_score=float(row['blur_score']),
                image_path=str(row['image_path'])
            )
            for _, row in df_faces.iterrows()
        ]
        
    except Exception as e:
        logger.error(f"❌ Error getting face data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/speaker-identification/stats/{series}/{season}/{episode}", response_model=SpeakerStatsResponse)
async def get_speaker_stats(series: str, season: str, episode: str):
    """Get speaker identification statistics for an episode."""
    try:
        import pandas as pd
        from collections import Counter
        
        path_handler = PathHandler(series, season, episode)
        episode_code = f"{series}{season}{episode}"
        
        # Load dialogue data
        dialogue_json_path = path_handler.get_dialogue_json_path().replace('.json', '_final.json')
        if not os.path.exists(dialogue_json_path):
            dialogue_json_path = path_handler.get_dialogue_json_path()
            
        if not os.path.exists(dialogue_json_path):
            raise HTTPException(status_code=404, detail="Dialogue data not found")
        
        with open(dialogue_json_path, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)
        
        dialogue_lines = dialogue_data.get('dialogue_lines', [])
        
        # Calculate statistics
        total_lines = len(dialogue_lines)
        speakers_found = list(set(
            line.get('speaker') for line in dialogue_lines 
            if line.get('speaker') and line.get('speaker_confidence', 0) >= 50
        ))
        
        # Confidence distribution
        confidence_ranges = {
            "95-100%": 0,
            "90-95%": 0, 
            "80-90%": 0,
            "50-80%": 0,
            "0-50%": 0,
            "No speaker": 0
        }
        
        for line in dialogue_lines:
            conf = line.get('speaker_confidence', 0)
            if conf >= 95:
                confidence_ranges["95-100%"] += 1
            elif conf >= 90:
                confidence_ranges["90-95%"] += 1
            elif conf >= 80:
                confidence_ranges["80-90%"] += 1
            elif conf >= 50:
                confidence_ranges["50-80%"] += 1
            elif conf > 0:
                confidence_ranges["0-50%"] += 1
            else:
                confidence_ranges["No speaker"] += 1
        
        # Face cluster stats
        face_cluster_stats = {}
        faces_csv_path = path_handler.get_dialogue_faces_csv_path()
        if os.path.exists(faces_csv_path):
            df_faces = pd.read_csv(faces_csv_path)
            for speaker in speakers_found:
                speaker_faces = df_faces[
                    (df_faces['speaker'] == speaker) & 
                    (df_faces['is_llm_confident'] == True)
                ]
                face_cluster_stats[speaker] = len(speaker_faces)
        
        total_faces = 0
        faces_csv_path = path_handler.get_dialogue_faces_csv_path()
        if os.path.exists(faces_csv_path):
            df_faces = pd.read_csv(faces_csv_path)
            total_faces = len(df_faces)
        
        return SpeakerStatsResponse(
            episode_code=episode_code,
            total_dialogue_lines=total_lines,
            total_faces_extracted=total_faces,
            speakers_found=speakers_found,
            confidence_distribution=confidence_ranges,
            face_cluster_stats=face_cluster_stats
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting speaker stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/speaker-identification/data/{series}/{season}/{episode}")
async def delete_speaker_data(series: str, season: str, episode: str):
    """Delete speaker identification data for an episode."""
    try:
        from backend.src.narrative_storage_management.vector_store_service import FaceEmbeddingVectorStore
        
        path_handler = PathHandler(series, season, episode)
        
        # List of files to delete
        files_to_delete = [
            path_handler.get_dialogue_json_path(),
            path_handler.get_dialogue_json_path().replace('.json', '_final.json'),
            path_handler.get_speaker_analysis_path(),
            path_handler.get_dialogue_faces_csv_path(),
            path_handler.get_dialogue_embeddings_metadata_csv_path(),
            path_handler.get_speaker_face_associations_path()
        ]
        
        # Directories to delete
        dirs_to_delete = [
            path_handler.get_dialogue_faces_dir(),
            path_handler.get_dialogue_frames_dir(),
            path_handler.get_dialogue_embeddings_dir(),
            path_handler.get_cluster_visualization_dir()
        ]
        
        deleted_files = []
        deleted_dirs = []
        
        # Delete files
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(file_path)
        
        # Delete directories
        import shutil
        for dir_path in dirs_to_delete:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                deleted_dirs.append(dir_path)
        
        # Delete from vector store
        try:
            face_vector_store = FaceEmbeddingVectorStore()
            face_vector_store.delete_episode_faces(series, season, episode)
        except Exception as vs_error:
            logger.warning(f"⚠️ Could not delete from vector store: {vs_error}")
        
        return {
            "message": f"Deleted speaker identification data for {series}{season}{episode}",
            "files_deleted": len(deleted_files),
            "directories_deleted": len(deleted_dirs)
        }
        
    except Exception as e:
        logger.error(f"❌ Error deleting speaker data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/speaker-identification/vector-store/stats")
async def get_face_vector_store_stats():
    """Get statistics about the face embeddings vector store."""
    try:
        from backend.src.narrative_storage_management.vector_store_service import FaceEmbeddingVectorStore
        
        face_vector_store = FaceEmbeddingVectorStore()
        stats = face_vector_store.get_collection_stats()
        
        return {
            "total_faces": stats.get('total_faces', 0),
            "episodes_processed": stats.get('episodes', 0),
            "speakers_identified": stats.get('speakers', 0),
            "high_confidence_faces": stats.get('high_confidence_faces', 0),
            "confidence_rate": round(stats.get('confidence_rate', 0.0), 2)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting vector store stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# CLUSTER VISUALIZATION ENDPOINTS

@app.post("/api/face-processing/visualizations")
async def generate_cluster_visualizations(request: FaceExtractionRequest):
    """Generate 2D PCA cluster visualizations for face embeddings."""
    episode_code = f"{request.series}{request.season}{request.episode}"
    
    logger.info(f"📊 Generating cluster visualizations for {episode_code}")
    
    try:
        # Initialize components
        path_handler = PathHandler(request.series, request.season, request.episode)
        
        # Check if face embeddings exist
        faces_csv_path = path_handler.get_dialogue_faces_csv_path()
        if not os.path.exists(faces_csv_path):
            raise HTTPException(status_code=404, detail="Face data not found. Run face extraction first.")
        
        # Load face data
        df_faces = pd.read_csv(faces_csv_path)
        if df_faces.empty:
            raise HTTPException(status_code=404, detail="No face data available")
        
        # Check if embeddings exist in vector store
        from backend.src.narrative_storage_management.vector_store_service import FaceEmbeddingVectorStore
        face_vector_store = FaceEmbeddingVectorStore()
        
        # Load embeddings from vector store (simplified approach)
        # For visualization, we need the embeddings as numpy arrays
        # This would need to be implemented in the vector store service
        
        # Generate visualizations
        from backend.src.subtitle_speaker_identification.face_cluster_visualizer import FaceClusterVisualizer
        
        visualizer = FaceClusterVisualizer(path_handler)
        viz_paths = visualizer.create_cluster_visualizations(
            df_faces=df_faces,
            output_format=request.detector if hasattr(request, 'output_format') else "both",
            save_plots=True
        )
        
        return {
            "episode_code": episode_code,
            "status": "success",
            "visualizations_generated": len(viz_paths),
            "visualization_paths": viz_paths
        }
        
    except Exception as e:
        logger.error(f"❌ Cluster visualization generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization generation failed: {str(e)}")


@app.get("/api/speaker-identification/visualizations/{series}/{season}/{episode}")
async def get_cluster_visualizations(series: str, season: str, episode: str):
    """Get cluster visualization file paths for an episode."""
    try:
        path_handler = PathHandler(series, season, episode)
        viz_dir = path_handler.get_cluster_visualization_dir()
        
        if not os.path.exists(viz_dir):
            raise HTTPException(status_code=404, detail="Cluster visualizations not found")
        
        # Find visualization files
        viz_files = []
        for file_path in Path(viz_dir).glob("*.png"):
            viz_files.append({
                "name": file_path.name,
                "path": str(file_path),
                "type": "cluster_visualization"
            })
        
        return {
            "episode_code": f"{series}{season}{episode}",
            "visualization_directory": viz_dir,
            "visualizations": viz_files
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting cluster visualizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# FACE PROCESSING ENDPOINTS



@app.post("/api/face-processing/embeddings", response_model=FaceProcessingResponse)
async def generate_face_embeddings(request: FaceEmbeddingRequest):
    """Generate embeddings for extracted faces."""
    try:
        import time
        import pandas as pd
        from backend.src.subtitle_speaker_identification.subtitle_face_embedder import SubtitleFaceEmbedder
        from backend.src.narrative_storage_management.vector_store_service import FaceEmbeddingVectorStore
        from backend.src.path_handler import PathHandler
        
        start_time = time.time()
        episode_code = f"{request.series}{request.season}{request.episode}"
        
        logger.info(f"🧠 Starting embedding generation for {episode_code}")
        
        # Initialize components
        path_handler = PathHandler(request.series, request.season, request.episode)
        face_embedder = SubtitleFaceEmbedder(path_handler)
        face_vector_store = FaceEmbeddingVectorStore()
        
        # Load face data
        faces_csv_path = path_handler.get_dialogue_faces_csv_path()
        if not os.path.exists(faces_csv_path):
            raise HTTPException(status_code=404, detail="Face data not found. Extract faces first.")
        
        df_faces = pd.read_csv(faces_csv_path)
        if df_faces.empty:
            raise HTTPException(status_code=400, detail="No face data found in CSV file")
        
        # Generate embeddings
        df_faces_with_embeddings = face_embedder.generate_embeddings(
            df_faces=df_faces,
            model=request.embedding_model,
            force_regenerate=request.force_regenerate
        )
        
        # Save to vector store
        face_embedder.save_embeddings_to_vector_store(df_faces_with_embeddings, face_vector_store)
        
        processing_time = time.time() - start_time
        embeddings_generated = len(df_faces_with_embeddings) if not df_faces_with_embeddings.empty else 0
        
        logger.info(f"✅ Embedding generation completed: {embeddings_generated} embeddings in {processing_time:.1f}s")
        
        return FaceProcessingResponse(
            episode_code=episode_code,
            status="success",
            faces_extracted=len(df_faces),
            embeddings_generated=embeddings_generated,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        return FaceProcessingResponse(
            episode_code=f"{request.series}{request.season}{request.episode}",
            status="failed",
            faces_extracted=0,
            embeddings_generated=0,
            error_message=str(e)
        )

@app.post("/api/face-processing/complete", response_model=FaceProcessingResponse)
async def extract_faces_and_generate_embeddings(request: FaceExtractionRequest):
    """Complete face processing: extract faces and generate embeddings."""
    try:
        import time
        import pandas as pd
        from backend.src.subtitle_speaker_identification.subtitle_face_extractor import SubtitleFaceExtractor
        from backend.src.subtitle_speaker_identification.subtitle_face_embedder import SubtitleFaceEmbedder
        from backend.src.subtitle_speaker_identification.srt_parser import SRTParser
        from backend.src.narrative_storage_management.vector_store_service import FaceEmbeddingVectorStore
        from backend.src.path_handler import PathHandler
        
        start_time = time.time()
        episode_code = f"{request.series}{request.season}{request.episode}"
        
        logger.info(f"🎬 Starting complete face processing for {episode_code}")
        
        # Initialize components
        path_handler = PathHandler(request.series, request.season, request.episode)
        srt_parser = SRTParser()
        face_extractor = SubtitleFaceExtractor(path_handler)
        face_embedder = SubtitleFaceEmbedder(path_handler)
        face_vector_store = FaceEmbeddingVectorStore()
        
        # Step 1: Parse SRT and extract faces
        srt_path = path_handler.get_srt_file_path()
        if not os.path.exists(srt_path):
            raise HTTPException(status_code=404, detail=f"SRT file not found: {srt_path}")
        
        dialogue_json_path = path_handler.get_dialogue_json_path()
        dialogue_lines = srt_parser.parse(srt_path, dialogue_json_path)
        
        df_faces = face_extractor.extract_faces_from_subtitles(
            dialogue_lines=dialogue_lines,
            detector=request.detector,
            min_confidence=request.min_confidence,
            min_face_area_ratio=request.min_face_area_ratio,
            blur_threshold=request.blur_threshold,
            enable_eye_validation=request.enable_eye_validation,
            eye_alignment_threshold=request.eye_alignment_threshold,
            eye_distance_threshold=request.eye_distance_threshold,
            force_extract=request.force_extract
        )
        
        faces_extracted = len(df_faces) if not df_faces.empty else 0
        
        # Step 2: Generate embeddings if faces were extracted
        embeddings_generated = 0
        if not df_faces.empty:
            df_faces_with_embeddings = face_embedder.generate_embeddings(
                df_faces=df_faces,
                model="Facenet512",  # Default model for complete processing
                force_regenerate=request.force_extract  # Use same force flag
            )
            
            # Save to vector store
            face_embedder.save_embeddings_to_vector_store(df_faces_with_embeddings, face_vector_store)
            embeddings_generated = len(df_faces_with_embeddings) if not df_faces_with_embeddings.empty else 0
            
            # Step 3: Generate cluster visualizations with PCA
            logger.info("📊 Generating face cluster visualizations...")
            try:
                from backend.src.subtitle_speaker_identification.face_cluster_visualizer import FaceClusterVisualizer
                
                visualizer = FaceClusterVisualizer(path_handler)
                viz_paths = visualizer.create_cluster_visualizations(
                    df_faces=df_faces_with_embeddings,
                    output_format="both",  # Both matplotlib and plotly
                    save_plots=True
                )
                
                logger.info(f"✅ Generated {len(viz_paths)} cluster visualizations")
                
            except Exception as viz_error:
                logger.warning(f"⚠️ Visualization generation failed: {viz_error}")
                # Don't fail the entire pipeline for visualization errors
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Complete face processing finished: {faces_extracted} faces, {embeddings_generated} embeddings in {processing_time:.1f}s")
        
        return FaceProcessingResponse(
            episode_code=episode_code,
            status="success",
            faces_extracted=faces_extracted,
            embeddings_generated=embeddings_generated,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Complete face processing failed: {e}")
        return FaceProcessingResponse(
            episode_code=f"{request.series}{request.season}{request.episode}",
            status="failed",
            faces_extracted=0,
            embeddings_generated=0,
            error_message=str(e)
        )
    try:
        from backend.src.subtitle_speaker_identification.face_cluster_visualizer import FaceClusterVisualizer
        
        path_handler = PathHandler(series, season, episode)
        visualizer = FaceClusterVisualizer(path_handler)
        
        # Get existing visualization files
        viz_files = visualizer.get_visualization_urls()
        
        if not viz_files:
            raise HTTPException(status_code=404, detail="No visualizations found. Run speaker identification first.")
        
        # Convert absolute paths to relative paths for API response
        relative_paths = {}
        for viz_type, file_path in viz_files.items():
            # Make path relative to data directory for serving
            if os.path.exists(file_path):
                relative_path = os.path.relpath(file_path, "data")
                relative_paths[viz_type] = f"/data/{relative_path}"
        
        return {
            "episode_code": f"{series}{season}{episode}",
            "visualizations": relative_paths,
            "total_files": len(relative_paths)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting visualizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# MISSING CHARACTER ENDPOINTS
@app.get("/api/characters/{series}", response_model=List[CharacterResponse])
async def get_characters(series: str):
    """Get all characters for a series."""
    try:
        with db_manager.session_scope() as session:
            character_repository = CharacterRepository(session)
            characters = character_repository.get_by_series(series)
            return [
                CharacterResponse(
                    entity_name=char.entity_name,
                    best_appellation=char.best_appellation,
                    series=char.series,
                    appellations=[app.appellation for app in char.appellations]
                )
                for char in characters
            ]
    except Exception as e:
        logger.error(f"Error getting characters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/characters/{series}", response_model=CharacterResponse)
async def create_character(series: str, character_data: CharacterCreateRequest):
    """Create a new character."""
    try:
        with db_manager.session_scope() as session:
            character_service = CharacterService(CharacterRepository(session))
            character = character_service.add_or_update_character(
                EntityLink(
                    entity_name=character_data.entity_name,
                    best_appellation=character_data.best_appellation,
                    appellations=character_data.appellations
                ),
                series=series
            )
            if not character:
                raise HTTPException(status_code=400, detail="Failed to create character")
            return CharacterResponse(
                entity_name=character.entity_name,
                best_appellation=character.best_appellation,
                series=character.series,
                appellations=[app.appellation for app in character.appellations]
            )
    except Exception as e:
        logger.error(f"Error creating character: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/characters/{series}", response_model=CharacterResponse)
async def update_character(series: str, character_data: CharacterCreateRequest):
    """Update an existing character."""
    try:
        with db_manager.session_scope() as session:
            character_service = CharacterService(CharacterRepository(session))
            
            # Create EntityLink with all the data
            entity = EntityLink(
                entity_name=character_data.entity_name,
                best_appellation=character_data.best_appellation,
                appellations=character_data.appellations,
                entity_type="PERSON"  # Default type for characters
            )
            
            # Update character
            character = character_service.add_or_update_character(entity, series)
            if not character:
                raise HTTPException(status_code=404, detail="Character not found")
            
            # Refresh the character to get updated data
            session.refresh(character)
            
            return CharacterResponse(
                entity_name=character.entity_name,
                best_appellation=character.best_appellation,
                series=character.series,
                appellations=[app.appellation for app in character.appellations]
            )
    except Exception as e:
        logger.error(f"Error updating character: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/characters/{series}/{entity_name}")
async def delete_character(series: str, entity_name: str):
    """Delete a character."""
    try:
        with db_manager.session_scope() as session:
            character_repository = CharacterRepository(session)
            character = character_repository.get_by_entity_name(entity_name, series)
            if not character:
                raise HTTPException(status_code=404, detail="Character not found")
            character_repository.delete(character)
            return {"message": "Character deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting character: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/characters/{series}/merge", response_model=dict)
async def merge_characters(series: str, merge_data: CharacterMergeRequest):
    """Merge two characters."""
    try:
        logger.info(f"Attempting to merge characters in series {series}. Request data: {merge_data.dict()}")
        
        with db_manager.session_scope() as session:
            character_service = CharacterService(CharacterRepository(session))
            
            success = character_service.merge_characters(
                merge_data.character1_id,
                merge_data.character2_id,
                series,
                keep_character=merge_data.keep_character
            )
            
            if not success:
                error_msg = f"Failed to merge characters. One or both characters not found: {merge_data.character1_id}, {merge_data.character2_id}"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=400, 
                    detail=error_msg
                )
            
            logger.info(f"Successfully merged characters {merge_data.character1_id} and {merge_data.character2_id}")
            return {"message": "Characters merged successfully"}
            
    except Exception as e:
        error_msg = f"Error merging characters: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# MISSING VECTOR STORE ENDPOINTS
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
                n_results=10
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

@app.post("/api/vector-store/compare", response_model=Dict)
async def compare_arcs(arc_ids: List[str]):
    """Calculate cosine distance between two arcs."""
    try:
        if len(arc_ids) != 2:
            raise HTTPException(
                status_code=400,
                detail="Exactly two arc IDs must be provided"
            )

        vector_store_service = VectorStoreService()
        result = vector_store_service.calculate_arcs_cosine_distances(arc_ids)
        return result

    except Exception as e:
        logger.error(f"Error comparing arcs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vector-store/{series}/clusters", response_model=List[Dict])
async def get_arc_clusters(
    series: str,
    threshold: float = 0.5,  # Similarity threshold (1 - distance)
    min_cluster_size: int = 2,
    max_clusters: int = 5  # New parameter to limit number of clusters
):
    """Get clusters of similar arcs using HDBSCAN."""
    try:
        # Convert similarity threshold to distance threshold
        cluster_selection_epsilon = 1 - threshold
        
        vector_store_service = VectorStoreService()
        clusters = vector_store_service.find_similar_arcs_clusters(
            series=series,
            min_cluster_size=min_cluster_size,
            min_samples=1,  # Keep this low for more flexible clustering
            cluster_selection_epsilon=cluster_selection_epsilon
        )
        
        # Sort clusters by size and probability, then limit to max_clusters
        sorted_clusters = sorted(
            clusters,
            key=lambda x: (x['size'], x['average_probability']),
            reverse=True
        )[:max_clusters]
        
        return sorted_clusters
        
    except Exception as e:
        logger.error(f"Error getting arc clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# MISSING ARC MANAGEMENT ENDPOINTS
@app.patch("/api/progressions/{progression_id}", response_model=ArcProgressionResponse)
async def update_progression(progression_id: str, progression_data: ProgressionUpdateRequest):
    """Update a progression."""
    try:
        with db_manager.session_scope() as session:
            progression_repository = ArcProgressionRepository(session)
            character_service = CharacterService(CharacterRepository(session))
            vector_store_service = VectorStoreService()
            
            # Get the progression
            progression = progression_repository.get_by_id(progression_id)
            if not progression:
                raise HTTPException(status_code=404, detail="Progression not found")
            
            # Update the progression
            progression.content = progression_data.content
            
            # Update interfering characters
            # Split the characters if they're in a string format
            character_names = (
                progression_data.interfering_characters.split(';') 
                if isinstance(progression_data.interfering_characters, str) 
                else progression_data.interfering_characters
            )
            
            # Get characters from the database
            interfering_characters = character_service.get_characters_by_appellations(
                character_names,
                progression.series
            )
            
            progression.interfering_characters = interfering_characters
            
            # Save changes
            session.commit()
            session.refresh(progression)
            
            # Update vector store
            narrative_arc_service = NarrativeArcService(
                arc_repository=NarrativeArcRepository(session),
                progression_repository=progression_repository,
                character_service=character_service,
                llm_service=None,
                vector_store_service=vector_store_service,
                session=session
            )
            narrative_arc_service.update_embeddings(progression.narrative_arc)
            
            return ArcProgressionResponse.from_progression(progression)
            
    except Exception as e:
        logger.error(f"Error updating progression: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/progressions", response_model=ArcProgressionResponse)
async def create_progression(progression: ProgressionCreateRequest):
    """Create a new progression."""
    try:
        with db_manager.session_scope() as session:
            # Format season and episode numbers
            season = f"S{pad_number(progression.season.replace('S', ''))}"
            episode = f"E{pad_number(progression.episode.replace('E', ''))}"

            progression_repository = ArcProgressionRepository(session)
            arc_repository = NarrativeArcRepository(session)
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
            
            # Handle interfering_characters as either string or list
            character_names = (
                progression.interfering_characters.split(';') 
                if isinstance(progression.interfering_characters, str) 
                else progression.interfering_characters
            )

            # Create progression
            new_progression = narrative_arc_service.add_progression(
                arc_id=progression.arc_id,
                content=progression.content,
                series=progression.series,
                season=season,
                episode=episode,
                interfering_characters=character_names  # Pass as list
            )
            
            if not new_progression:
                raise HTTPException(status_code=404, detail="Arc not found")
                
            return ArcProgressionResponse.from_progression(new_progression)
            
    except Exception as e:
        logger.error(f"Error creating progression: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/arcs/merge", response_model=NarrativeArcResponse)
async def merge_arcs(merge_data: ArcMergeRequest):
    """Merge two arcs into a new one."""
    try:
        logger.info(f"Received merge request for arcs: {merge_data.arc_id_1} and {merge_data.arc_id_2}")
        logger.info(f"Merge data: {merge_data.dict()}")
        
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
            
            # Verify both arcs exist before attempting merge
            arc1 = arc_repository.get_by_id(merge_data.arc_id_1)
            arc2 = arc_repository.get_by_id(merge_data.arc_id_2)
            
            if not arc1 or not arc2:
                raise HTTPException(
                    status_code=404,
                    detail="One or both arcs not found"
                )
            
            # Normalize season/episode in progression mappings
            normalized_mappings = []
            for prog in merge_data.progression_mappings:
                if not prog.content.strip():  # Skip empty progressions
                    continue
                    
                season = f"S{pad_number(prog.season.replace('S', ''))}"
                episode = f"E{pad_number(prog.episode.replace('E', ''))}"
                
                normalized_mappings.append({
                    "season": season,
                    "episode": episode,
                    "content": prog.content,
                    "interfering_characters": prog.interfering_characters
                })
            
            merged_arc = narrative_arc_service.merge_arcs(
                arc_id_1=merge_data.arc_id_1,
                arc_id_2=merge_data.arc_id_2,
                merged_title=merge_data.merged_title,
                merged_description=merge_data.merged_description,
                merged_arc_type=merge_data.merged_arc_type,
                main_characters=merge_data.main_characters,
                progression_mappings=normalized_mappings
            )
            
            if not merged_arc:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to merge arcs"
                )
            
            logger.info(f"Successfully merged arcs into '{merged_arc.title}'")
            return NarrativeArcResponse.from_arc(merged_arc)
            
    except Exception as e:
        logger.error(f"Error merging arcs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/progressions/{progression_id}")
async def delete_progression(progression_id: str):
    """Delete a progression."""
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
            
            # Get the progression to find its arc
            progression = progression_repository.get_by_id(progression_id)
            if not progression:
                raise HTTPException(status_code=404, detail="Progression not found")
            
            # Delete the progression
            progression_repository.delete(progression_id)
            
            # Update the vector store for the arc
            narrative_arc_service.update_embeddings(progression.narrative_arc)
            
            return {"message": "Progression deleted successfully"}
            
    except Exception as e:
        logger.error(f"Error deleting progression: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class InitialProgressionRequest(BaseModel):
    content: str
    season: str
    episode: str
    interfering_characters: str  # Comma-separated string

class ArcCreateRequest(BaseModel):
    title: str
    description: str
    arc_type: str
    main_characters: str  # Comma-separated string
    series: str
    initial_progression: InitialProgressionRequest

@app.post("/api/arcs", response_model=NarrativeArcResponse)
async def create_arc(arc_data: ArcCreateRequest):
    """Create a new narrative arc."""
    try:
        # Log the raw request data
        logger.info("=== Raw Request Data ===")
        logger.info(f"Type of arc_data: {type(arc_data)}")
        logger.info(f"Raw arc_data: {arc_data.dict()}")
        
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
            
            # Create arc data dict with initial progression data
            arc_dict = {
                'title': arc_data.title,
                'description': arc_data.description,
                'arc_type': arc_data.arc_type,
                'main_characters': arc_data.main_characters,  # Already semicolon-separated string
                'single_episode_progression_string': arc_data.initial_progression.content if arc_data.initial_progression else None,
                'interfering_episode_characters': arc_data.initial_progression.interfering_characters if arc_data.initial_progression else None
            }

            # Create the arc with its initial progression
            new_arc = narrative_arc_service.add_arc(
                arc_data=arc_dict,
                series=arc_data.series,
                season=arc_data.initial_progression.season if arc_data.initial_progression else "",
                episode=arc_data.initial_progression.episode if arc_data.initial_progression else "",
                initial_progression=None  # We're using the fields in arc_dict instead
            )
            
            logger.info(f"Created new arc with ID: {new_arc.id}")
            if arc_data.initial_progression:
                logger.info(f"Initial progression data: {arc_data.initial_progression}")
            
            return NarrativeArcResponse.from_arc(new_arc)
            
    except Exception as e:
        logger.error(f"Error creating arc: {e}")
        logger.error("Full error details:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/arcs/by-id/{arc_id}", response_model=NarrativeArcResponse)
async def get_arc_by_id_alt(arc_id: str):
    """Get a single arc by ID with all its details."""
    try:
        with db_manager.session_scope() as session:
            arc_repository = NarrativeArcRepository(session)
            arc = arc_repository.get_by_id(arc_id)
            
            if not arc:
                raise HTTPException(status_code=404, detail=f"Arc with ID {arc_id} not found")
            
            # Ensure the arc is fully loaded with its relationships
            session.refresh(arc, ['main_characters', 'progressions'])
            
            # Normalize season/episode format in progressions
            for prog in arc.progressions:
                normalized_season, normalized_episode = normalize_season_episode(prog.season, prog.episode)
                prog.season = normalized_season
                prog.episode = normalized_episode
            
            # Create response using the model
            response = NarrativeArcResponse.from_arc(arc)
            logger.info(f"Retrieved arc '{arc.title}' with {len(arc.progressions)} progressions")
            return response
            
    except Exception as e:
        logger.error(f"Error getting arc by ID {arc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/progressions/generate")
async def generate_progression(
    series: str = Query(...),
    season: str = Query(...),
    episode: str = Query(...),
    data: dict = Body(...)
):
    """Generate progression content for an arc in a specific episode."""
    try:
        logger.info(f"Generating progression for series: {series}, S{season}E{episode}")
        logger.info(f"Request data: {data}")
        
        with db_manager.session_scope() as session:
            arc_repository = NarrativeArcRepository(session)
            progression_repository = ArcProgressionRepository(session)
            character_repository = CharacterRepository(session)
            character_service = CharacterService(character_repository)
            llm_service = LLMService()

            # Get arc details
            arc = None
            arc_title = None
            arc_description = None
            arc_id = None
            
            if data.get('arc_id'):
                arc = arc_repository.get_by_id(data['arc_id'])
                if arc:
                    arc_title = arc.title
                    arc_description = arc.description
                    arc_id = arc.id
                    logger.info(f"Found arc by ID: {arc.title}")
            else:
                arc_title = data.get('arc_title')
                arc_description = data.get('arc_description')
                logger.info(f"Using provided arc title: {arc_title}")

            if not arc_title or not arc_description:
                logger.error("Missing required arc information")
                return {
                    "content": "",
                    "interfering_characters": []
                }

            # Get episode plot path and generate content
            path_handler = PathHandler(series, season, episode)
            episode_plot_path = path_handler.get_entity_normalized_plot_file_path()
            
            logger.info(f"Generating content using plot file: {episode_plot_path}")
            content = llm_service.generate_progression_content(
                arc_title=arc_title,
                arc_description=arc_description,
                episode_plot_path=episode_plot_path
            )
            
            # If content is NO_PROGRESSION and we have an arc, delete any existing progression
            if content == "NO_PROGRESSION" and arc and data.get('delete_existing', False):
                existing_progression = progression_repository.get_single(
                    arc_id=arc.id,
                    series=series,
                    season=season,
                    episode=episode
                )
                if existing_progression:
                    logger.info(f"Deleting existing progression for S{season}E{episode} due to NO_PROGRESSION")
                    progression_repository.delete(existing_progression.id)

                return {
                    "content": "NO_PROGRESSION",
                    "interfering_characters": []
                }

            # If we have valid content, process it
            if content != "NO_PROGRESSION":
                # Get all characters for the series
                all_characters = character_repository.get_by_series(series)
                
                # Find mentioned characters in the content
                mentioned_appellations = []
                for character in all_characters:
                    for appellation in character.appellations:
                        if appellation.appellation in content:
                            mentioned_appellations.append(character.best_appellation)
                            break  # Only add each character once

                logger.info(f"Found interfering characters: {mentioned_appellations}")
                return {
                    "content": content,
                    "interfering_characters": mentioned_appellations
                }

    except Exception as e:
        logger.error(f"Error generating progression: {str(e)}")
        logger.exception("Full error details:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/arcs/{arc_id}", response_model=NarrativeArcResponse)
async def get_arc_by_id(arc_id: str):
    """Get a single arc by ID with all its details."""
    try:
        with db_manager.session_scope() as session:
            arc_repository = NarrativeArcRepository(session)
            arc = arc_repository.get_by_id(arc_id)
            
            if not arc:
                raise HTTPException(status_code=404, detail=f"Arc with ID {arc_id} not found")
            
            # Ensure the arc is fully loaded with its relationships
            session.refresh(arc, ['main_characters', 'progressions'])
            
            # Normalize season/episode format in progressions
            for prog in arc.progressions:
                normalized_season, normalized_episode = normalize_season_episode(prog.season, prog.episode)
                prog.season = normalized_season
                prog.episode = normalized_episode
            
            # Create response using the model
            response = NarrativeArcResponse.from_arc(arc)
            logger.info(f"Retrieved arc '{arc.title}' with {len(arc.progressions)} progressions")
            return response
            
    except Exception as e:
        logger.error(f"Error getting arc by ID {arc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# SCENE TIMESTAMP VALIDATION ENDPOINTS

@app.post("/api/scene-validation/validate", response_model=SceneValidationResponse)
async def validate_scene_timestamps(request: SceneValidationRequest):
    """
    Validate scene timestamp coverage and optionally fix issues.
    """
    try:
        logger.info(f"🔍 Scene validation request: {request.series} {request.season} {request.episode}")
        
        # Build file paths
        from backend.src.path_handler import PathHandler
        path_handler = PathHandler(request.series, request.season, request.episode)
        
        scenes_file_path = path_handler.get_scene_timestamps_path()
        srt_file_path = path_handler.get_srt_file_path()
        
        # Check if required files exist
        if not os.path.exists(scenes_file_path):
            return SceneValidationResponse(
                episode_code=f"{request.series}{request.season}{request.episode}",
                is_valid=False,
                coverage_percentage=0.0,
                total_subtitles=0,
                covered_subtitles=0,
                gaps_found=0,
                gap_details=[],
                error_message=f"Scene timestamps file not found: {scenes_file_path}"
            )
        
        if not os.path.exists(srt_file_path):
            return SceneValidationResponse(
                episode_code=f"{request.series}{request.season}{request.episode}",
                is_valid=False,
                coverage_percentage=0.0,
                total_subtitles=0,
                covered_subtitles=0,
                gaps_found=0,
                gap_details=[],
                error_message=f"SRT file not found: {srt_file_path}"
            )
        
        # Perform validation/correction
        if request.fix_issues:
            # Get LLM for corrections
            llm_cheap = get_llm(LLMType.CHEAP)
            validation_result = validate_and_fix_scene_timestamps(
                scenes_file_path, srt_file_path, llm_cheap, scenes_file_path
            )
            corrections_applied = validation_result.corrected_scenes is not None
        else:
            # Just get coverage report without fixing
            coverage_report = get_scene_coverage_report(scenes_file_path, srt_file_path)
            if "error" in coverage_report:
                return SceneValidationResponse(
                    episode_code=f"{request.series}{request.season}{request.episode}",
                    is_valid=False,
                    coverage_percentage=0.0,
                    total_subtitles=0,
                    covered_subtitles=0,
                    gaps_found=0,
                    gap_details=[],
                    error_message=coverage_report["error"]
                )
            
            # Convert coverage report to validation result format
            validation_result = type('ValidationResult', (), {
                'is_valid': coverage_report['is_valid'],
                'coverage_percentage': coverage_report['coverage_percentage'],
                'total_subtitles': coverage_report['total_subtitles'],
                'covered_subtitles': coverage_report['covered_subtitles'],
                'gaps': [
                    type('Gap', (), {
                        'gap_type': gap['type'],
                        'start_seconds': 0.0,
                        'end_seconds': 0.0,
                        'affected_scenes': gap['affected_scenes'],
                        'missing_subtitle_indices': gap['missing_subtitles'],
                        'description': gap['description']
                    })() for gap in coverage_report['gap_details']
                ]
            })()
            corrections_applied = False
        
        # Convert gaps to response format
        gap_details = [
            SceneCoverageGap(
                gap_type=gap.gap_type,
                start_seconds=gap.start_seconds,
                end_seconds=gap.end_seconds,
                affected_scenes=gap.affected_scenes,
                missing_subtitle_indices=gap.missing_subtitle_indices,
                description=gap.description
            ) for gap in validation_result.gaps
        ]
        
        response = SceneValidationResponse(
            episode_code=f"{request.series}{request.season}{request.episode}",
            is_valid=validation_result.is_valid,
            coverage_percentage=validation_result.coverage_percentage,
            total_subtitles=validation_result.total_subtitles,
            covered_subtitles=validation_result.covered_subtitles,
            gaps_found=len(validation_result.gaps),
            gap_details=gap_details,
            corrections_applied=corrections_applied
        )
        
        logger.info(f"✅ Scene validation completed: {validation_result.coverage_percentage:.1f}% coverage")
        return response
        
    except Exception as e:
        logger.error(f"❌ Scene validation failed: {e}")
        return SceneValidationResponse(
            episode_code=f"{request.series}{request.season}{request.episode}",
            is_valid=False,
            coverage_percentage=0.0,
            total_subtitles=0,
            covered_subtitles=0,
            gaps_found=0,
            gap_details=[],
            error_message=str(e)
        )

@app.get("/api/scene-validation/report/{series}/{season}/{episode}", response_model=Dict)
async def get_scene_coverage_report_endpoint(series: str, season: str, episode: str):
    """
    Get a detailed scene coverage report without making any changes.
    """
    try:
        logger.info(f"📊 Scene coverage report request: {series} {season} {episode}")
        
        # Build file paths
        from backend.src.path_handler import PathHandler
        path_handler = PathHandler(series, season, episode)
        
        scenes_file_path = path_handler.get_scene_timestamps_path()
        srt_file_path = path_handler.get_srt_file_path()
        
        # Check if required files exist
        if not os.path.exists(scenes_file_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Scene timestamps file not found: {scenes_file_path}"
            )
        
        if not os.path.exists(srt_file_path):
            raise HTTPException(
                status_code=404, 
                detail=f"SRT file not found: {srt_file_path}"
            )
        
        # Get coverage report
        report = get_scene_coverage_report(scenes_file_path, srt_file_path)
        
        if "error" in report:
            raise HTTPException(status_code=500, detail=report["error"])
        
        # Add episode information
        report["episode_code"] = f"{series}{season}{episode}"
        report["scenes_file"] = scenes_file_path
        report["srt_file"] = srt_file_path
        
        logger.info(f"📊 Coverage report generated: {report['coverage_percentage']:.1f}% coverage")
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to generate coverage report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scene-validation/fix/{series}/{season}/{episode}")
async def fix_scene_timestamps_endpoint(series: str, season: str, episode: str):
    """
    Force fix scene timestamp issues for a specific episode.
    """
    try:
        logger.info(f"🔧 Scene timestamp fix request: {series} {season} {episode}")
        
        # Build file paths
        from backend.src.path_handler import PathHandler
        path_handler = PathHandler(series, season, episode)
        
        scenes_file_path = path_handler.get_scene_timestamps_path()
        srt_file_path = path_handler.get_srt_file_path()
        
        # Check if required files exist
        if not os.path.exists(scenes_file_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Scene timestamps file not found: {scenes_file_path}"
            )
        
        if not os.path.exists(srt_file_path):
            raise HTTPException(
                status_code=404, 
                detail=f"SRT file not found: {srt_file_path}"
            )
        
        # Get LLM and perform corrections
        llm_cheap = get_llm(LLMType.CHEAP)
        validation_result = validate_and_fix_scene_timestamps(
            scenes_file_path, srt_file_path, llm_cheap, scenes_file_path
        )
        
        return {
            "episode_code": f"{series}{season}{episode}",
            "success": validation_result.is_valid,
            "coverage_percentage": validation_result.coverage_percentage,
            "corrections_applied": validation_result.corrected_scenes is not None,
            "gaps_found": len(validation_result.gaps),
            "message": "Scene timestamps validated and corrected successfully" if validation_result.is_valid else "Issues found but may not be fully resolved"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to fix scene timestamps: {e}")
        raise HTTPException(status_code=500, detail=str(e))
