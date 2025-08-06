"""
Validation utilities for recap detection functionality.
"""

import json
import os
from typing import Dict, List, Tuple
from ..utils.logger_utils import setup_logging

logger = setup_logging(__name__)

def validate_recap_detection(series: str, season: str, episode: str, path_handler) -> Dict[str, any]:
    """
    Validate that recap detection is working properly for an episode.
    
    Args:
        series: Series name
        season: Season name  
        episode: Episode name
        path_handler: PathHandler instance
        
    Returns:
        Dict with validation results
    """
    results = {
        "has_scenes_json": False,
        "has_plot_txt": False,
        "total_scenes": 0,
        "recap_scenes": 0,
        "non_recap_scenes": 0,
        "plot_txt_scenes": 0,
        "recap_filtered_correctly": False,
        "validation_passed": False,
        "issues": []
    }
    
    try:
        # Check scenes JSON file
        scenes_json_path = path_handler.get_plot_scenes_json_path()
        if os.path.exists(scenes_json_path):
            results["has_scenes_json"] = True
            
            with open(scenes_json_path, 'r') as f:
                data = json.load(f)
            
            scenes = data.get("scenes", [])
            results["total_scenes"] = len(scenes)
            
            recap_scenes = [s for s in scenes if s.get("is_recap", False)]
            non_recap_scenes = [s for s in scenes if not s.get("is_recap", False)]
            
            results["recap_scenes"] = len(recap_scenes)
            results["non_recap_scenes"] = len(non_recap_scenes)
            
            if recap_scenes:
                logger.info(f"🔄 Found {len(recap_scenes)} recap scene(s) in {series} {season} {episode}")
                for scene in recap_scenes:
                    logger.info(f"   Recap Scene {scene.get('scene_number', '?')}: {scene.get('plot_segment', '')[:50]}...")
        else:
            results["issues"].append("Scenes JSON file not found")
        
        # Check plot TXT file
        plot_txt_path = path_handler.get_raw_plot_file_path()
        if os.path.exists(plot_txt_path):
            results["has_plot_txt"] = True
            
            with open(plot_txt_path, 'r') as f:
                content = f.read()
            
            # Count scenes in plot txt (rough estimate)
            scene_lines = [line for line in content.split('\n') if line.strip().startswith('Scene ')]
            results["plot_txt_scenes"] = len(scene_lines)
            
            # Check if filtering worked correctly
            if results["has_scenes_json"]:
                expected_scenes = results["non_recap_scenes"]
                actual_scenes = results["plot_txt_scenes"]
                
                if expected_scenes == actual_scenes:
                    results["recap_filtered_correctly"] = True
                    logger.info(f"✅ Recap filtering working correctly: {actual_scenes} scenes in plot.txt (expected {expected_scenes})")
                else:
                    results["issues"].append(f"Scene count mismatch: {actual_scenes} in plot.txt vs {expected_scenes} non-recap scenes expected")
        else:
            results["issues"].append("Plot TXT file not found")
        
        # Overall validation
        results["validation_passed"] = (
            results["has_scenes_json"] and 
            results["has_plot_txt"] and 
            results["recap_filtered_correctly"] and
            len(results["issues"]) == 0
        )
        
        # Log summary
        if results["validation_passed"]:
            logger.info(f"✅ Recap detection validation PASSED for {series} {season} {episode}")
        else:
            logger.warning(f"⚠️ Recap detection validation FAILED for {series} {season} {episode}")
            for issue in results["issues"]:
                logger.warning(f"   Issue: {issue}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error during recap validation: {e}")
        results["issues"].append(f"Validation error: {str(e)}")
        return results

def get_recap_summary_for_series(series: str, base_dir: str = "data") -> Dict[str, any]:
    """
    Get a summary of recap detection across all episodes in a series.
    
    Args:
        series: Series name
        base_dir: Base data directory
        
    Returns:
        Dict with series-wide recap summary
    """
    from ..path_handler import PathHandler
    
    summary = {
        "series": series,
        "episodes_processed": 0,
        "episodes_with_recaps": 0,
        "total_recap_scenes": 0,
        "validation_issues": []
    }
    
    try:
        series_dir = os.path.join(base_dir, series)
        if not os.path.exists(series_dir):
            logger.warning(f"Series directory not found: {series_dir}")
            return summary
        
        for season_dir in os.listdir(series_dir):
            if not season_dir.startswith('S'):
                continue
                
            season_path = os.path.join(series_dir, season_dir)
            if not os.path.isdir(season_path):
                continue
            
            for episode_dir in os.listdir(season_path):
                if not episode_dir.startswith('E'):
                    continue
                
                try:
                    path_handler = PathHandler(series, season_dir, episode_dir)
                    validation = validate_recap_detection(series, season_dir, episode_dir, path_handler)
                    
                    summary["episodes_processed"] += 1
                    
                    if validation["recap_scenes"] > 0:
                        summary["episodes_with_recaps"] += 1
                        summary["total_recap_scenes"] += validation["recap_scenes"]
                    
                    if not validation["validation_passed"]:
                        summary["validation_issues"].extend([
                            f"{season_dir}{episode_dir}: {issue}" for issue in validation["issues"]
                        ])
                
                except Exception as e:
                    logger.error(f"Error processing {season_dir}{episode_dir}: {e}")
                    summary["validation_issues"].append(f"{season_dir}{episode_dir}: Processing error")
        
        # Log summary
        logger.info(f"📊 Recap Summary for {series}:")
        logger.info(f"   Episodes processed: {summary['episodes_processed']}")
        logger.info(f"   Episodes with recaps: {summary['episodes_with_recaps']}")
        logger.info(f"   Total recap scenes: {summary['total_recap_scenes']}")
        logger.info(f"   Validation issues: {len(summary['validation_issues'])}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating recap summary: {e}")
        return summary
