"""
Enhanced SRT Generation Logger

This module provides comprehensive logging for the enhanced SRT generation process,
tracking face analysis, candidate detection, assignment decisions, and final SRT formatting.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class FaceAnalysisResult:
    """Result of analyzing a single face in a dialogue."""
    dialogue_index: int
    face_index: int
    timestamp_seconds: float
    detection_confidence: float
    blur_score: float
    image_path: str
    cluster_id: Optional[int]
    cluster_character: Optional[str]
    cluster_confidence: Optional[float]
    best_character_match: Optional[str]
    best_similarity: float
    qualification_threshold: float
    qualified: bool
    assignment_threshold: float
    assignable: bool
    method: str  # 'cluster_assigned', 'character_median_direct', 'llm_original'

@dataclass
class DialogueAnalysisResult:
    """Result of analyzing all faces in a dialogue."""
    dialogue_index: int
    original_speaker: str
    original_confidence: Optional[float]
    total_faces_detected: int
    qualified_candidates: List[str]
    qualified_similarities: List[float]
    all_candidates: List[str]
    all_similarities: List[float]
    best_assignment: Optional[Dict[str, Any]]
    final_speaker: str
    final_confidence: Optional[float]
    resolution_method: str
    srt_prefix: str
    processing_time_ms: float

class EnhancedSRTLogger:
    """
    Comprehensive logger for enhanced SRT generation process.
    
    Tracks:
    - Face detection and analysis
    - Character median comparisons
    - Candidate qualification and assignment
    - SRT prefix generation
    - Performance metrics
    """
    
    def __init__(self, series: str, season: str, episode: str, log_dir: Optional[str] = None):
        """
        Initialize the enhanced SRT logger.
        
        Args:
            series: Series name
            season: Season number
            episode: Episode number
            log_dir: Directory for log files (defaults to data/{series}/{season}/{episode}/logs/)
        """
        self.series = series
        self.season = season
        self.episode = episode
        self.episode_code = f"{series}_S{season}_E{episode}"
        
        # Setup log directory
        if log_dir is None:
            log_dir = f"data/{series}/{season}/{episode}/logs"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Statistics tracking
        self.stats = {
            'total_dialogues': 0,
            'dialogues_with_faces': 0,
            'total_faces_analyzed': 0,
            'faces_qualified': 0,
            'faces_assigned': 0,
            'multiple_candidate_dialogues': 0,
            'direct_assignments': 0,
            'cluster_assignments': 0,
            'llm_fallback': 0,
            'processing_times': []
        }
        
        # Store detailed results
        self.face_analyses: List[FaceAnalysisResult] = []
        self.dialogue_analyses: List[DialogueAnalysisResult] = []
        
        # Start session
        self.session_start = datetime.now()
        self.logger.info(f"🎬 Enhanced SRT Logger started for {self.episode_code}")
        self.logger.info(f"📁 Log directory: {self.log_dir}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create logger
        self.logger = logging.getLogger(f"enhanced_srt_{self.episode_code}")
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # File handler for detailed logs
        log_file = self.log_dir / f"enhanced_srt_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_face_analysis(self, face_data: Dict[str, Any], analysis_result: Dict[str, Any]) -> FaceAnalysisResult:
        """
        Log analysis of a single face.
        
        Args:
            face_data: Original face data from DataFrame
            analysis_result: Result from _analyze_dialogue_faces_direct
            
        Returns:
            FaceAnalysisResult with analysis details
        """
        dialogue_index = face_data.get('dialogue_index', -1)
        face_index = face_data.get('face_index', -1)
        
        # Extract analysis details
        best_character = analysis_result.get('best_character')
        best_similarity = analysis_result.get('best_similarity', 0.0)
        qualified = analysis_result.get('qualified', False)
        assignable = analysis_result.get('assignable', False)
        method = analysis_result.get('method', 'unknown')
        
        # Create result object
        result = FaceAnalysisResult(
            dialogue_index=dialogue_index,
            face_index=face_index,
            timestamp_seconds=face_data.get('timestamp_seconds', 0.0),
            detection_confidence=face_data.get('detection_confidence', 0.0),
            blur_score=face_data.get('blur_score', 0.0),
            image_path=face_data.get('image_path', ''),
            cluster_id=face_data.get('face_id', face_data.get('cluster_id', -1)),
            cluster_character=analysis_result.get('cluster_character'),
            cluster_confidence=analysis_result.get('cluster_confidence'),
            best_character_match=best_character,
            best_similarity=best_similarity,
            qualification_threshold=analysis_result.get('qualification_threshold', 0.5),
            qualified=qualified,
            assignment_threshold=analysis_result.get('assignment_threshold', 0.7),
            assignable=assignable,
            method=method
        )
        
        # Log the analysis
        self.logger.debug(f"👤 Face Analysis - Dialogue {dialogue_index}, Face {face_index}:")
        self.logger.debug(f"   📊 Detection: {result.detection_confidence:.1f}%, Blur: {result.blur_score:.3f}")
        self.logger.debug(f"   🎭 Best Match: '{result.best_character_match}' ({result.best_similarity:.3f})")
        self.logger.debug(f"   ✅ Qualified: {result.qualified}, Assignable: {result.assignable}")
        self.logger.debug(f"   🔧 Method: {result.method}")
        
        # Update statistics
        self.stats['total_faces_analyzed'] += 1
        if result.qualified:
            self.stats['faces_qualified'] += 1
        if result.assignable:
            self.stats['faces_assigned'] += 1
        
        # Store result
        self.face_analyses.append(result)
        
        return result
    
    def log_dialogue_analysis(self, dialogue_index: int, dialogue_faces: List[Dict], 
                            face_candidates: Dict[str, Any], original_dialogue: Any,
                            final_dialogue: Any, processing_time_ms: float) -> DialogueAnalysisResult:
        """
        Log analysis of all faces in a dialogue.
        
        Args:
            dialogue_index: Index of the dialogue
            dialogue_faces: List of face data for this dialogue
            face_candidates: Result from _analyze_dialogue_faces_direct
            original_dialogue: Original dialogue line
            final_dialogue: Final dialogue line after assignment
            processing_time_ms: Time taken to process this dialogue
            
        Returns:
            DialogueAnalysisResult with analysis details
        """
        # Extract candidate information
        qualified_candidates = face_candidates.get('qualified_candidates', [])
        qualified_similarities = face_candidates.get('qualified_similarities', [])
        all_candidates = face_candidates.get('all_candidates', [])
        all_similarities = face_candidates.get('all_similarities', [])
        best_assignment = face_candidates.get('best_assignment')
        
        # Determine resolution method and final speaker
        resolution_method = getattr(final_dialogue, 'resolution_method', 'unknown')
        final_speaker = getattr(final_dialogue, 'speaker', 'Unknown')
        final_confidence = getattr(final_dialogue, 'speaker_confidence', None)
        
        # Generate SRT prefix for logging
        srt_prefix = self._generate_srt_prefix_for_logging(final_dialogue)
        
        # Create result object
        result = DialogueAnalysisResult(
            dialogue_index=dialogue_index,
            original_speaker=getattr(original_dialogue, 'speaker', 'Unknown'),
            original_confidence=getattr(original_dialogue, 'speaker_confidence', None),
            total_faces_detected=len(dialogue_faces),
            qualified_candidates=qualified_candidates,
            qualified_similarities=qualified_similarities,
            all_candidates=all_candidates,
            all_similarities=all_similarities,
            best_assignment=best_assignment,
            final_speaker=final_speaker,
            final_confidence=final_confidence,
            resolution_method=resolution_method,
            srt_prefix=srt_prefix,
            processing_time_ms=processing_time_ms
        )
        
        # Log the dialogue analysis
        self.logger.info(f"🎬 Dialogue {dialogue_index} Analysis:")
        original_conf_str = f"{result.original_confidence:.1f}" if result.original_confidence is not None else "None"
        self.logger.info(f"   👥 Original: '{result.original_speaker}' ({original_conf_str}%)")
        self.logger.info(f"   📷 Faces Detected: {result.total_faces_detected}")
        self.logger.info(f"   ✅ Qualified Candidates: {len(result.qualified_candidates)}")
        final_conf_str = f"{result.final_confidence:.1f}" if result.final_confidence is not None else "None"
        self.logger.info(f"   🎯 Final: '{result.final_speaker}' ({final_conf_str}%)")
        self.logger.info(f"   🔧 Method: {result.resolution_method}")
        self.logger.info(f"   📝 SRT: {result.srt_prefix}")
        self.logger.info(f"   ⏱️  Processing Time: {result.processing_time_ms:.1f}ms")
        
        # Log multiple candidate scenarios
        if len(result.qualified_candidates) > 1:
            self.logger.info(f"   🎭 MULTIPLE CANDIDATES: {result.qualified_candidates}")
            self.stats['multiple_candidate_dialogues'] += 1
        
        # Update statistics
        self.stats['total_dialogues'] += 1
        if result.total_faces_detected > 0:
            self.stats['dialogues_with_faces'] += 1
        
        if result.resolution_method == "character_median_direct":
            self.stats['direct_assignments'] += 1
        elif result.resolution_method == "cluster_assigned":
            self.stats['cluster_assignments'] += 1
        elif result.resolution_method in ["llm_original", "face_clustering_multi_unresolved"]:
            self.stats['llm_fallback'] += 1
        
        self.stats['processing_times'].append(processing_time_ms)
        
        # Store result
        self.dialogue_analyses.append(result)
        
        return result
    
    def _generate_srt_prefix_for_logging(self, dialogue: Any) -> str:
        """Generate SRT prefix for logging purposes."""
        try:
            # Import the speaker prefix generation function
            from .speaker_identification_pipeline import SpeakerIdentificationPipeline
            from ..config import config
            
            # Create a temporary pipeline instance to access the method
            temp_pipeline = SpeakerIdentificationPipeline("", "", "")
            return temp_pipeline._generate_speaker_prefix(dialogue)  # Boolean confidence doesn't need threshold
        except Exception as e:
            # Fallback if we can't generate the actual prefix
            speaker = getattr(dialogue, 'speaker', 'Unknown')
            confidence = getattr(dialogue, 'speaker_confidence', 0)
            confidence_str = f"{confidence:.0f}" if confidence is not None else "0"
            return f"{speaker.upper()}({confidence_str}%): "
    
    def log_enhanced_srt_generation_start(self, total_dialogues: int):
        """Log the start of enhanced SRT generation."""
        self.logger.info(f"🚀 Starting Enhanced SRT Generation")
        self.logger.info(f"   📊 Total Dialogues: {total_dialogues}")
        self.logger.info(f"   ⚙️  Multiface Processing: {'ENABLED' if self._is_multiface_enabled() else 'DISABLED'}")
        self.logger.info(f"   🎯 Qualification Threshold: {self._get_qualification_threshold():.2f}")
        self.logger.info(f"   🏆 Assignment Threshold: {self._get_assignment_threshold():.2f}")
    
    def log_enhanced_srt_generation_complete(self, enhanced_srt_path: str):
        """Log the completion of enhanced SRT generation."""
        session_duration = datetime.now() - self.session_start
        
        self.logger.info(f"✅ Enhanced SRT Generation Complete")
        self.logger.info(f"   📁 Output File: {enhanced_srt_path}")
        self.logger.info(f"   ⏱️  Total Duration: {session_duration.total_seconds():.1f}s")
        
        # Log final statistics
        self._log_final_statistics()
        
        # Save detailed results to JSON
        self._save_detailed_results()
    
    def _is_multiface_enabled(self) -> bool:
        """Check if multiface processing is enabled."""
        try:
            from ..config import config
            return config.enable_multiface_processing
        except:
            return False
    
    def _get_qualification_threshold(self) -> float:
        """Get the current qualification threshold."""
        try:
            from ..config import config
            return config.character_median_similarity_threshold
        except:
            return 0.5
    
    def _get_assignment_threshold(self) -> float:
        """Get the current assignment threshold."""
        try:
            from ..config import config
            return config.character_median_assignment_threshold
        except:
            return 0.7
    
    def _log_final_statistics(self):
        """Log final statistics."""
        stats = self.stats
        
        self.logger.info(f"📊 Final Statistics:")
        self.logger.info(f"   🎬 Total Dialogues: {stats['total_dialogues']}")
        self.logger.info(f"   📷 Dialogues with Faces: {stats['dialogues_with_faces']}")
        self.logger.info(f"   👤 Total Faces Analyzed: {stats['total_faces_analyzed']}")
        self.logger.info(f"   ✅ Faces Qualified: {stats['faces_qualified']}")
        self.logger.info(f"   🎯 Faces Assigned: {stats['faces_assigned']}")
        self.logger.info(f"   🎭 Multiple Candidate Dialogues: {stats['multiple_candidate_dialogues']}")
        self.logger.info(f"   🔧 Direct Assignments: {stats['direct_assignments']}")
        self.logger.info(f"   🔗 Cluster Assignments: {stats['cluster_assignments']}")
        self.logger.info(f"   🔄 LLM Fallbacks: {stats['llm_fallback']}")
        
        if stats['processing_times']:
            avg_time = sum(stats['processing_times']) / len(stats['processing_times'])
            self.logger.info(f"   ⏱️  Average Processing Time: {avg_time:.1f}ms per dialogue")
    
    def _save_detailed_results(self):
        """Save detailed results to JSON files."""
        # Save face analyses
        face_analyses_file = self.log_dir / "face_analyses.json"
        with open(face_analyses_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(result) for result in self.face_analyses], f, indent=2, default=str)
        
        # Save dialogue analyses
        dialogue_analyses_file = self.log_dir / "dialogue_analyses.json"
        with open(dialogue_analyses_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(result) for result in self.dialogue_analyses], f, indent=2, default=str)
        
        # Save statistics
        stats_file = self.log_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        self.logger.info(f"💾 Detailed results saved to:")
        self.logger.info(f"   📄 Face Analyses: {face_analyses_file}")
        self.logger.info(f"   📄 Dialogue Analyses: {dialogue_analyses_file}")
        self.logger.info(f"   📄 Statistics: {stats_file}")
    
    def get_multiple_character_dialogues(self) -> List[DialogueAnalysisResult]:
        """Get all dialogues that had multiple character candidates."""
        return [d for d in self.dialogue_analyses if len(d.qualified_candidates) > 1]
    
    def get_direct_assignments(self) -> List[DialogueAnalysisResult]:
        """Get all dialogues that were directly assigned via character median matching."""
        return [d for d in self.dialogue_analyses if d.resolution_method == "character_median_direct"]
    
    def get_llm_fallbacks(self) -> List[DialogueAnalysisResult]:
        """Get all dialogues that fell back to LLM assignment."""
        return [d for d in self.dialogue_analyses if d.resolution_method in ["llm_original", "face_clustering_multi_unresolved"]] 