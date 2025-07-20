"""
Scene timestamp validation and correction functionality.
Provides comprehensive scene boundary validation and automated fixing.
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..utils.logger_utils import setup_logging
from ..ai_models.ai_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from ..utils.llm_utils import clean_llm_json_response
from .subtitle_processing import SubtitleEntry, PlotScene, parse_srt_file

logger = setup_logging(__name__)

@dataclass
class ValidationResult:
    """Result of scene timestamp validation."""
    total_subtitles: int
    covered_subtitles: int
    coverage_percentage: float
    issues_found: List[Dict]
    scenes_processed: int
    corrections_applied: int
    success: bool
    error_message: Optional[str] = None

@dataclass
class SceneCoverageGap:
    """Represents a gap in scene coverage."""
    gap_type: str  # 'missing_coverage', 'overlap', 'out_of_order'
    start_subtitle_index: int
    end_subtitle_index: int
    affected_scenes: List[int]
    description: str

def analyze_scene_coverage(scenes: List[PlotScene], subtitles: List[SubtitleEntry]) -> ValidationResult:
    """
    Analyze scene coverage of subtitles and detect issues.
    
    Args:
        scenes: List of scenes with timestamps
        subtitles: List of subtitle entries
        
    Returns:
        ValidationResult with coverage analysis
    """
    logger.info("🔍 Analyzing scene coverage of subtitles")
    
    total_subtitles = len(subtitles)
    covered_indices = set()
    issues = []
    
    # Check each scene's coverage
    for scene in scenes:
        if not scene.start_seconds or not scene.end_seconds:
            issues.append({
                "type": "missing_timestamps",
                "scene_number": scene.scene_number,
                "description": f"Scene {scene.scene_number} missing start/end timestamps"
            })
            continue
            
        # Find subtitles covered by this scene
        scene_covered = []
        for subtitle in subtitles:
            # Check if subtitle overlaps with scene time range
            if (subtitle.start_seconds >= scene.start_seconds and 
                subtitle.end_seconds <= scene.end_seconds):
                scene_covered.append(subtitle.index)
                covered_indices.add(subtitle.index)
        
        logger.debug(f"Scene {scene.scene_number} covers {len(scene_covered)} subtitles")
    
    # Find uncovered subtitles
    uncovered_indices = []
    for subtitle in subtitles:
        if subtitle.index not in covered_indices:
            uncovered_indices.append(subtitle.index)
    
    if uncovered_indices:
        issues.append({
            "type": "missing_coverage",
            "subtitle_indices": uncovered_indices,
            "count": len(uncovered_indices),
            "description": f"{len(uncovered_indices)} subtitles not covered by any scene"
        })
    
    # Check for scene overlaps
    sorted_scenes = sorted([s for s in scenes if s.start_seconds and s.end_seconds], 
                          key=lambda x: x.start_seconds)
    
    for i in range(len(sorted_scenes) - 1):
        current = sorted_scenes[i]
        next_scene = sorted_scenes[i + 1]
        
        if current.end_seconds > next_scene.start_seconds:
            overlap_duration = current.end_seconds - next_scene.start_seconds
            issues.append({
                "type": "scene_overlap",
                "scene_before": current.scene_number,
                "scene_after": next_scene.scene_number,
                "overlap_duration": overlap_duration,
                "description": f"Scenes {current.scene_number} and {next_scene.scene_number} overlap by {overlap_duration:.2f}s"
            })
    
    covered_count = len(covered_indices)
    coverage_percentage = (covered_count / total_subtitles * 100) if total_subtitles > 0 else 0
    
    logger.info(f"Coverage analysis: {covered_count}/{total_subtitles} subtitles covered ({coverage_percentage:.1f}%)")
    logger.info(f"Found {len(issues)} issues")
    
    return ValidationResult(
        total_subtitles=total_subtitles,
        covered_subtitles=covered_count,
        coverage_percentage=coverage_percentage,
        issues_found=issues,
        scenes_processed=len(scenes),
        corrections_applied=0,
        success=coverage_percentage >= 95.0 and len([i for i in issues if i['type'] in ['scene_overlap', 'missing_timestamps']]) == 0
    )

def fix_scene_boundaries_with_llm(
    issues: List[Dict], 
    scenes: List[PlotScene], 
    subtitles: List[SubtitleEntry], 
    llm: AzureChatOpenAI
) -> Tuple[List[PlotScene], int]:
    """
    Fix scene boundary issues using LLM analysis.
    
    Args:
        issues: List of detected issues
        scenes: List of scenes to fix
        subtitles: List of subtitle entries
        llm: LLM for boundary correction
        
    Returns:
        Tuple of (corrected_scenes, corrections_applied)
    """
    logger.info("🔧 Fixing scene boundaries with LLM")
    
    corrected_scenes = scenes.copy()
    corrections_applied = 0
    
    # Handle missing coverage issues
    missing_coverage_issues = [i for i in issues if i['type'] == 'missing_coverage']
    for issue in missing_coverage_issues:
        logger.info(f"🔧 Fixing missing coverage for {issue['count']} subtitles")
        
        uncovered_indices = issue['subtitle_indices']
        uncovered_subtitles = [sub for sub in subtitles if sub.index in uncovered_indices]
        
        if not uncovered_subtitles:
            continue
            
        # Group consecutive uncovered subtitles
        gaps = []
        current_gap = [uncovered_subtitles[0]]
        
        for i in range(1, len(uncovered_subtitles)):
            if uncovered_subtitles[i].index == current_gap[-1].index + 1:
                current_gap.append(uncovered_subtitles[i])
            else:
                gaps.append(current_gap)
                current_gap = [uncovered_subtitles[i]]
        gaps.append(current_gap)
        
        # Fix each gap
        for gap in gaps:
            corrected_scenes, corrections = _fix_coverage_gap(gap, corrected_scenes, subtitles, llm)
            corrections_applied += corrections
    
    # Handle scene overlap issues
    overlap_issues = [i for i in issues if i['type'] == 'scene_overlap']
    for issue in overlap_issues:
        logger.info(f"🔧 Fixing overlap between scenes {issue['scene_before']}-{issue['scene_after']}")
        
        corrected_scenes, corrections = _fix_scene_overlap(
            issue['scene_before'], issue['scene_after'], corrected_scenes, subtitles, llm
        )
        corrections_applied += corrections
    
    logger.info(f"✅ Applied {corrections_applied} boundary corrections")
    return corrected_scenes, corrections_applied

def _fix_coverage_gap(
    gap_subtitles: List[SubtitleEntry], 
    scenes: List[PlotScene], 
    all_subtitles: List[SubtitleEntry], 
    llm: AzureChatOpenAI
) -> Tuple[List[PlotScene], int]:
    """Fix a specific coverage gap by extending scene boundaries."""
    
    if not gap_subtitles:
        return scenes, 0
    
    gap_start = gap_subtitles[0].start_seconds
    gap_end = gap_subtitles[-1].end_seconds
    
    # Find adjacent scenes
    scenes_with_times = [s for s in scenes if s.start_seconds is not None and s.end_seconds is not None]
    before_scene = None
    after_scene = None
    
    for scene in scenes_with_times:
        if scene.end_seconds <= gap_start:
            if not before_scene or scene.end_seconds > before_scene.end_seconds:
                before_scene = scene
        if scene.start_seconds >= gap_end:
            if not after_scene or scene.start_seconds < after_scene.start_seconds:
                after_scene = scene
    
    # Create context for LLM
    gap_text = "\n".join([f"[{sub.start_time} --> {sub.end_time}] {sub.text}" for sub in gap_subtitles])
    
    before_context = ""
    after_context = ""
    
    if before_scene:
        before_context = f"Before: {before_scene.plot_segment[:100]}..."
    if after_scene:
        after_context = f"After: {after_scene.plot_segment[:100]}..."
    
    prompt = f"""You need to assign uncovered subtitles to the most appropriate adjacent scene. Analyze the content and determine which scene should be extended to cover this gap.

UNCOVERED SUBTITLES:
{gap_text}

ADJACENT SCENES:
{before_context}
{after_context}

TASK: Determine which scene (before or after) should be extended to cover these subtitles. Consider:
1. Content relevance - which scene's narrative best matches the subtitle content
2. Natural flow - which extension creates the most coherent scene boundary

Return ONLY a JSON object:
{{
    "extend_scene": "before" or "after",
    "reasoning": "Brief explanation of the decision"
}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        decision = clean_llm_json_response(response.content)
        
        if isinstance(decision, list) and len(decision) > 0:
            decision = decision[0]
        
        if not isinstance(decision, dict):
            logger.error(f"Invalid LLM response format: {decision}")
            return scenes, 0
        
        extend_direction = decision.get("extend_scene")
        reasoning = decision.get("reasoning", "No reasoning provided")
        
        if extend_direction == "before" and before_scene:
            # Extend the before scene to cover the gap
            scene_idx = next(i for i, s in enumerate(scenes) if s.scene_number == before_scene.scene_number)
            scenes[scene_idx].end_time = gap_subtitles[-1].end_time
            scenes[scene_idx].end_seconds = gap_subtitles[-1].end_seconds
            logger.info(f"✅ Extended scene {before_scene.scene_number} to cover gap. Reasoning: {reasoning}")
            return scenes, 1
            
        elif extend_direction == "after" and after_scene:
            # Extend the after scene to cover the gap
            scene_idx = next(i for i, s in enumerate(scenes) if s.scene_number == after_scene.scene_number)
            scenes[scene_idx].start_time = gap_subtitles[0].start_time
            scenes[scene_idx].start_seconds = gap_subtitles[0].start_seconds
            logger.info(f"✅ Extended scene {after_scene.scene_number} to cover gap. Reasoning: {reasoning}")
            return scenes, 1
        
    except Exception as e:
        logger.error(f"❌ Failed to fix coverage gap: {e}")
    
    return scenes, 0

def _fix_scene_overlap(
    scene_before_num: int, 
    scene_after_num: int, 
    scenes: List[PlotScene], 
    subtitles: List[SubtitleEntry], 
    llm: AzureChatOpenAI
) -> Tuple[List[PlotScene], int]:
    """Fix overlap between two specific scenes."""
    
    before_scene = next((s for s in scenes if s.scene_number == scene_before_num), None)
    after_scene = next((s for s in scenes if s.scene_number == scene_after_num), None)
    
    if not before_scene or not after_scene:
        return scenes, 0
    
    # Get overlapping subtitles
    overlap_start = after_scene.start_seconds
    overlap_end = before_scene.end_seconds
    
    overlap_subtitles = [
        sub for sub in subtitles 
        if overlap_start <= sub.start_seconds <= overlap_end
    ]
    
    if not overlap_subtitles:
        return scenes, 0
    
    # Create context for LLM decision
    overlap_text = "\n".join([f"[{sub.start_time} --> {sub.end_time}] {sub.text}" for sub in overlap_subtitles])
    
    prompt = f"""Two scenes overlap in time. You need to determine the optimal boundary between them by analyzing the content.

SCENE {scene_before_num}: {before_scene.plot_segment}

SCENE {scene_after_num}: {after_scene.plot_segment}

OVERLAPPING SUBTITLES:
{overlap_text}

TASK: Determine where to place the boundary between these scenes. Choose a subtitle timestamp that creates the most natural division.

Return ONLY a JSON object:
{{
    "boundary_subtitle_index": <index_of_subtitle_that_ends_first_scene>,
    "reasoning": "Brief explanation of the boundary choice"
}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        decision = clean_llm_json_response(response.content)
        
        if isinstance(decision, list) and len(decision) > 0:
            decision = decision[0]
        
        if not isinstance(decision, dict):
            logger.error(f"Invalid LLM response format: {decision}")
            return scenes, 0
        
        boundary_index = decision.get("boundary_subtitle_index")
        reasoning = decision.get("reasoning", "No reasoning provided")
        
        if boundary_index:
            boundary_subtitle = next((sub for sub in subtitles if sub.index == boundary_index), None)
            if boundary_subtitle:
                # Update scene boundaries
                before_idx = next(i for i, s in enumerate(scenes) if s.scene_number == scene_before_num)
                after_idx = next(i for i, s in enumerate(scenes) if s.scene_number == scene_after_num)
                
                scenes[before_idx].end_time = boundary_subtitle.end_time
                scenes[before_idx].end_seconds = boundary_subtitle.end_seconds
                
                # Find next subtitle for after scene start
                next_subtitle = next((sub for sub in subtitles if sub.index == boundary_index + 1), None)
                if next_subtitle:
                    scenes[after_idx].start_time = next_subtitle.start_time
                    scenes[after_idx].start_seconds = next_subtitle.start_seconds
                
                logger.info(f"✅ Fixed overlap between scenes {scene_before_num}-{scene_after_num} at subtitle {boundary_index}. Reasoning: {reasoning}")
                return scenes, 1
        
    except Exception as e:
        logger.error(f"❌ Failed to fix scene overlap: {e}")
    
    return scenes, 0

def validate_and_fix_scene_timestamps(
    scenes_file_path: str,
    srt_file_path: str,
    llm: AzureChatOpenAI,
    save_corrected: bool = True
) -> ValidationResult:
    """
    Main function to validate and fix scene timestamps.
    
    Args:
        scenes_file_path: Path to scene timestamps JSON file
        srt_file_path: Path to SRT subtitle file
        llm: LLM for boundary correction
        save_corrected: Whether to save corrected scenes back to file
        
    Returns:
        ValidationResult with analysis and correction results
    """
    logger.info("🔍 Starting scene timestamp validation and correction")
    
    try:
        # Load scenes
        with open(scenes_file_path, 'r', encoding='utf-8') as f:
            scenes_data = json.load(f)
        
        scenes = []
        for scene_data in scenes_data.get("scenes", []):
            scene = PlotScene(
                scene_number=scene_data.get("scene_number", 0),
                plot_segment=scene_data.get("plot_segment", ""),
                start_time=scene_data.get("start_time"),
                end_time=scene_data.get("end_time"),
                start_seconds=scene_data.get("start_seconds"),
                end_seconds=scene_data.get("end_seconds")
            )
            scenes.append(scene)
        
        # Load subtitles
        subtitles = parse_srt_file(srt_file_path)
        
        # Analyze coverage
        validation_result = analyze_scene_coverage(scenes, subtitles)
        
        # Fix issues if found
        if validation_result.issues_found:
            logger.info(f"Found {len(validation_result.issues_found)} issues, applying corrections")
            
            corrected_scenes, corrections_applied = fix_scene_boundaries_with_llm(
                validation_result.issues_found, scenes, subtitles, llm
            )
            
            validation_result.corrections_applied = corrections_applied
            
            # Re-analyze after corrections
            final_result = analyze_scene_coverage(corrected_scenes, subtitles)
            validation_result.covered_subtitles = final_result.covered_subtitles
            validation_result.coverage_percentage = final_result.coverage_percentage
            validation_result.success = final_result.success
            
            # Save corrected scenes if requested
            if save_corrected and corrections_applied > 0:
                corrected_data = {
                    "scenes": [
                        {
                            "scene_number": scene.scene_number,
                            "plot_segment": scene.plot_segment,
                            "start_time": scene.start_time,
                            "end_time": scene.end_time,
                            "start_seconds": scene.start_seconds,
                            "end_seconds": scene.end_seconds
                        }
                        for scene in corrected_scenes
                    ]
                }
                
                with open(scenes_file_path, 'w', encoding='utf-8') as f:
                    json.dump(corrected_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"💾 Saved corrected scenes to {scenes_file_path}")
        
        logger.info(f"✅ Validation complete: {validation_result.coverage_percentage:.1f}% coverage, {validation_result.corrections_applied} corrections applied")
        return validation_result
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return ValidationResult(
            total_subtitles=0,
            covered_subtitles=0,
            coverage_percentage=0.0,
            issues_found=[],
            scenes_processed=0,
            corrections_applied=0,
            success=False,
            error_message=str(e)
        )

def get_scene_coverage_report(
    scenes_file_path: str,
    srt_file_path: str
) -> Dict:
    """
    Generate a detailed coverage report without making corrections.
    
    Args:
        scenes_file_path: Path to scene timestamps JSON file
        srt_file_path: Path to SRT subtitle file
        
    Returns:
        Dictionary with detailed coverage analysis
    """
    logger.info("📊 Generating scene coverage report")
    
    try:
        # Load scenes
        with open(scenes_file_path, 'r', encoding='utf-8') as f:
            scenes_data = json.load(f)
        
        scenes = []
        for scene_data in scenes_data.get("scenes", []):
            scene = PlotScene(
                scene_number=scene_data.get("scene_number", 0),
                plot_segment=scene_data.get("plot_segment", ""),
                start_time=scene_data.get("start_time"),
                end_time=scene_data.get("end_time"),
                start_seconds=scene_data.get("start_seconds"),
                end_seconds=scene_data.get("end_seconds")
            )
            scenes.append(scene)
        
        # Load subtitles
        subtitles = parse_srt_file(srt_file_path)
        
        # Analyze coverage
        validation_result = analyze_scene_coverage(scenes, subtitles)
        
        return {
            "total_subtitles": validation_result.total_subtitles,
            "covered_subtitles": validation_result.covered_subtitles,
            "coverage_percentage": validation_result.coverage_percentage,
            "issues": validation_result.issues_found,
            "scenes_count": validation_result.scenes_processed,
            "success": validation_result.success
        }
        
    except Exception as e:
        logger.error(f"❌ Coverage report generation failed: {e}")
        return {
            "error": str(e),
            "success": False
        }
