"""
Background job service for managing episode processing jobs.

This service handles the lifecycle of processing jobs, including
job creation, status tracking, progress updates, and cleanup.
"""

import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid

from backend.src.config import Config
from backend.src.utils.logger_utils import setup_logging
from .episode_processing_service import EpisodeProcessingService, ProcessingResult, ProcessingStatus
from .exceptions import ProcessingError


class JobStatus(str, Enum):
    """Status of background processing job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingJob:
    """Represents a background processing job."""
    id: str
    series: str
    season: str
    episodes: List[str]
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: Dict[str, str] = field(default_factory=dict)
    current_episode: Optional[str] = None
    current_step: Optional[str] = None
    results: Dict[str, ProcessingResult] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert job to dictionary for API responses."""
        return {
            "id": self.id,
            "series": self.series,
            "season": self.season,
            "episodes": self.episodes,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "progress": self.progress,
            "current_episode": self.current_episode,
            "current_step": self.current_step,
            "results_summary": {
                episode: {
                    "status": result.status.value,
                    "files_created": len(result.files_created),
                    "entities_extracted": result.entities_extracted,
                    "narrative_arcs_found": result.narrative_arcs_found
                }
                for episode, result in self.results.items()
            }
        }


@dataclass
class ProcessingRequest:
    """Request to start processing episodes."""
    series: str
    season: str
    episodes: List[str]


class BackgroundJobService:
    """
    Service for managing background processing jobs.
    
    This service handles:
    - Job creation and lifecycle management
    - Progress tracking across multiple episodes
    - Error handling and recovery
    - Job cancellation and cleanup
    """
    
    def __init__(self, episode_service: Optional[EpisodeProcessingService] = None, config: Optional[Config] = None):
        """
        Initialize the background job service.
        
        Args:
            episode_service: Episode processing service instance
            config: Configuration instance
        """
        self.config = config or Config()
        self.logger = setup_logging(self.__class__.__name__)
        self.episode_service = episode_service or EpisodeProcessingService(self.config)
        
        # In-memory job storage (in production, use Redis or database)
        self._jobs: Dict[str, ProcessingJob] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
    async def start_job(self, request: ProcessingRequest) -> ProcessingJob:
        """
        Start a new background processing job.
        
        Args:
            request: Processing request with series, season, and episodes
            
        Returns:
            Created processing job
            
        Raises:
            ProcessingError: If job creation fails
        """
        try:
            # Generate unique job ID
            job_id = f"{request.series}_{request.season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Create job
            job = ProcessingJob(
                id=job_id,
                series=request.series,
                season=request.season,
                episodes=request.episodes.copy(),
                status=JobStatus.PENDING,
                created_at=datetime.now(),
                progress={ep: "pending" for ep in request.episodes}
            )
            
            # Store job
            self._jobs[job_id] = job
            
            # Start background task
            task = asyncio.create_task(self._run_job(job_id))
            self._running_tasks[job_id] = task
            
            self.logger.info(f"🚀 Started processing job {job_id} for {request.series} {request.season} ({len(request.episodes)} episodes)")
            return job
            
        except Exception as e:
            error_msg = f"Failed to start processing job: {str(e)}"
            self.logger.error(error_msg)
            raise ProcessingError(error_msg, step="JOB_CREATION", cause=e)
    
    async def _run_job(self, job_id: str) -> None:
        """
        Run a processing job in the background.
        
        Args:
            job_id: ID of the job to run
        """
        job = self._jobs.get(job_id)
        if not job:
            self.logger.error(f"Job {job_id} not found")
            return
        
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            
            self.logger.info(f"🔄 Running job {job_id}: {job.series} {job.season}")
            
            # Process each episode
            failed_episodes = []
            completed_episodes = []
            
            for episode in job.episodes:
                if job.status == JobStatus.CANCELLED:
                    self.logger.info(f"Job {job_id} was cancelled, stopping processing")
                    break
                
                try:
                    job.current_episode = episode
                    job.progress[episode] = "processing"
                    
                    self.logger.info(f"📺 Processing episode {job.series} {job.season} {episode}")
                    
                    # Create progress callback for this episode
                    def progress_callback(step: str, message: str):
                        job.current_step = step
                        self.logger.info(f"  [{episode}] {step}: {message}")
                    
                    # Process the episode
                    result = await self.episode_service.process_episode(
                        job.series,
                        job.season,
                        episode,
                        progress_callback
                    )
                    
                    # Store result
                    job.results[episode] = result
                    
                    if result.status == ProcessingStatus.COMPLETED:
                        job.progress[episode] = "completed"
                        completed_episodes.append(episode)
                        self.logger.info(f"✅ Successfully processed {job.series} {job.season} {episode}")
                    else:
                        job.progress[episode] = "failed"
                        failed_episodes.append(episode)
                        self.logger.error(f"❌ Failed to process {job.series} {job.season} {episode}: {result.message}")
                        
                except Exception as episode_error:
                    job.progress[episode] = "failed"
                    failed_episodes.append(episode)
                    self.logger.error(f"❌ Error processing {job.series} {job.season} {episode}: {episode_error}")
                    
                    # Store error result
                    job.results[episode] = ProcessingResult(
                        series=job.series,
                        season=job.season,
                        episode=episode,
                        status=ProcessingStatus.FAILED,
                        message=str(episode_error),
                        files_created=[],
                        entities_extracted=0,
                        narrative_arcs_found=0,
                        steps_completed=[]
                    )
            
            # Determine final job status
            if job.status == JobStatus.CANCELLED:
                job.status = JobStatus.CANCELLED
            elif failed_episodes:
                job.status = JobStatus.FAILED
                job.error_message = f"Failed to process episodes: {', '.join(failed_episodes)}"
            else:
                job.status = JobStatus.COMPLETED
            
            job.completed_at = datetime.now()
            job.current_episode = None
            job.current_step = None
            
            self.logger.info(f"🏁 Job {job_id} completed with status: {job.status.value}")
            self.logger.info(f"   Completed: {len(completed_episodes)}, Failed: {len(failed_episodes)}")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = f"Job execution failed: {str(e)}"
            job.completed_at = datetime.now()
            job.current_episode = None
            job.current_step = None
            self.logger.error(f"💥 Job {job_id} failed: {e}")
        
        finally:
            # Cleanup task reference
            if job_id in self._running_tasks:
                del self._running_tasks[job_id]
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: ID of the job to cancel
            
        Returns:
            True if job was cancelled, False if not found or not cancellable
        """
        job = self._jobs.get(job_id)
        if not job:
            self.logger.warning(f"Cannot cancel job {job_id}: not found")
            return False
        
        if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
            self.logger.warning(f"Cannot cancel job {job_id}: status is {job.status.value}")
            return False
        
        # Mark job as cancelled
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        
        # Cancel the background task
        task = self._running_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.logger.info(f"🛑 Cancelled job {job_id}")
        return True
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """
        Get the status of a specific job.
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            Job instance if found, None otherwise
        """
        return self._jobs.get(job_id)
    
    def list_jobs(self, series_filter: Optional[str] = None, limit: Optional[int] = None) -> List[ProcessingJob]:
        """
        List processing jobs, optionally filtered by series.
        
        Args:
            series_filter: Only return jobs for this series (if provided)
            limit: Maximum number of jobs to return
            
        Returns:
            List of processing jobs, sorted by creation time (newest first)
        """
        jobs = list(self._jobs.values())
        
        # Filter by series if specified
        if series_filter:
            jobs = [job for job in jobs if job.series == series_filter]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply limit if specified
        if limit:
            jobs = jobs[:limit]
        
        return jobs
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job (only allowed for completed/failed/cancelled jobs).
        
        Args:
            job_id: ID of the job to delete
            
        Returns:
            True if job was deleted, False if not found or not deletable
        """
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        if job.status == JobStatus.RUNNING:
            self.logger.warning(f"Cannot delete running job {job_id}")
            return False
        
        # Clean up task reference if it exists
        if job_id in self._running_tasks:
            del self._running_tasks[job_id]
        
        # Delete job
        del self._jobs[job_id]
        self.logger.info(f"🗑️ Deleted job {job_id}")
        return True
    
    def get_job_statistics(self) -> Dict[str, any]:
        """
        Get statistics about all jobs.
        
        Returns:
            Dictionary with job statistics
        """
        jobs = list(self._jobs.values())
        
        total_jobs = len(jobs)
        status_counts = {}
        for status in JobStatus:
            status_counts[status.value] = len([job for job in jobs if job.status == status])
        
        # Calculate average processing time for completed jobs
        completed_jobs = [job for job in jobs if job.status == JobStatus.COMPLETED and job.started_at and job.completed_at]
        avg_processing_time = None
        if completed_jobs:
            total_time = sum(
                (job.completed_at - job.started_at).total_seconds() 
                for job in completed_jobs
            )
            avg_processing_time = total_time / len(completed_jobs)
        
        return {
            "total_jobs": total_jobs,
            "status_counts": status_counts,
            "running_jobs": len(self._running_tasks),
            "average_processing_time_seconds": avg_processing_time
        }
