"""
Validation utilities for recap generation.

This module provides functions for validating input files, JSON schemas,
and output quality checks throughout the recap generation process.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import jsonschema

from ...utils.logger_utils import setup_logging
from ..exceptions.recap_exceptions import (
    MissingInputFilesError, 
    ConfigurationError,
    SubtitleProcessingError
)
from ..models.recap_models import RecapConfiguration, RecapMetadata
from ...path_handler import PathHandler

logger = setup_logging(__name__)


class ValidationUtils:
    """Utility class for validation operations in recap generation."""
    
    @staticmethod
    def is_valid_timestamp(timestamp: str) -> bool:
        """Check if timestamp is in valid HH:MM:SS or HH:MM:SS,mmm format."""
        if not timestamp:
            return False
        try:
            # Check for milliseconds format first
            if ',' in timestamp:
                time_part, ms_part = timestamp.split(',')
                if len(ms_part) != 3 or not ms_part.isdigit():
                    return False
            else:
                time_part = timestamp
            
            # Validate time part
            parts = time_part.split(':')
            if len(parts) != 3:
                return False
            
            hours, minutes, seconds = parts
            
            # Check if all parts are numeric
            if not (hours.isdigit() and minutes.isdigit() and seconds.isdigit()):
                return False
            
            # Check ranges
            if int(hours) > 23 or int(minutes) > 59 or int(seconds) > 59:
                return False
            
            return True
        except:
            return False
    
    @staticmethod 
    def calculate_duration(start_timestamp: str, end_timestamp: str) -> float:
        """Calculate duration in seconds between two timestamps."""
        try:
            start_seconds = ValidationUtils.timestamp_to_seconds(start_timestamp)
            end_seconds = ValidationUtils.timestamp_to_seconds(end_timestamp)
            return end_seconds - start_seconds
        except:
            return 0.0
    
    @staticmethod
    def timestamp_to_seconds(timestamp: str) -> float:
        """Convert timestamp to total seconds."""
        try:
            # Handle milliseconds
            if ',' in timestamp:
                time_part, ms_part = timestamp.split(',')
                milliseconds = float(ms_part) / 1000
            else:
                time_part = timestamp
                milliseconds = 0.0
            
            # Parse time part
            parts = time_part.split(':')
            hours, minutes, seconds = map(int, parts)
            
            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds
            return total_seconds
        except:
            return 0.0
    
    @staticmethod
    def is_valid_video_file(filename: str) -> bool:
        """Check if filename has a valid video extension."""
        if not filename:
            return False
        valid_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
        return Path(filename).suffix.lower() in valid_extensions
    
    @staticmethod
    def is_valid_subtitle_file(filepath: str) -> bool:
        """Check if subtitle file exists and has valid format."""
        try:
            if not os.path.exists(filepath):
                return False
            if not filepath.lower().endswith('.srt'):
                return False
            # Basic content check
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read(100)  # Read first 100 chars
                return bool(content.strip())
        except:
            return False
    
    @staticmethod
    def validate_episode_prerequisites(path_handler: PathHandler) -> Dict[str, Any]:
        """
        Validate that an episode has all prerequisites for recap generation.
        
        Args:
            path_handler: PathHandler instance for the episode
            
        Returns:
            Dictionary with validation results
        """
        try:
            series = path_handler.get_series()
            season = path_handler.get_season()
            episode = path_handler.get_episode()
            
            # Define required files for recap generation
            required_files = {
                'plot_possible_speakers': path_handler.get_plot_possible_speakers_path(),
                'present_running_plotlines': path_handler.get_present_running_plotlines_path(),
                'possible_speakers_srt': path_handler.get_possible_speakers_srt_path(),
                'video_file': path_handler.get_video_file_path()
            }
            
            # Optional but recommended files
            optional_files = {
                'refined_entities': path_handler.get_episode_refined_entities_path()
            }
            
            # Check required files
            missing_required = []
            existing_required = []
            
            for file_type, file_path in required_files.items():
                if os.path.exists(file_path):
                    # Additional file-specific validation
                    file_info = ValidationUtils._validate_individual_file(file_path, file_type)
                    existing_required.append({
                        'type': file_type,
                        'path': file_path,
                        'size_mb': file_info['size_mb'],
                        'valid': file_info['valid'],
                        'issues': file_info.get('issues', [])
                    })
                else:
                    missing_required.append({
                        'type': file_type,
                        'path': file_path
                    })
            
            # Check optional files
            missing_optional = []
            existing_optional = []
            
            for file_type, file_path in optional_files.items():
                if os.path.exists(file_path):
                    file_info = ValidationUtils._validate_individual_file(file_path, file_type)
                    existing_optional.append({
                        'type': file_type,
                        'path': file_path,
                        'size_mb': file_info['size_mb'],
                        'valid': file_info['valid']
                    })
                else:
                    missing_optional.append({
                        'type': file_type,
                        'path': file_path
                    })
            
            # Check if episode processing is complete
            processing_complete = len(missing_required) == 0
            
            # Assess readiness level
            readiness_score = len(existing_required) / len(required_files)
            if len(optional_files) > 0:
                optional_score = len(existing_optional) / len(optional_files)
                readiness_score = (readiness_score * 0.8) + (optional_score * 0.2)
            
            return {
                'ready_for_recap': processing_complete,
                'readiness_score': readiness_score,
                'series': series,
                'season': season,
                'episode': episode,
                'required_files': {
                    'existing': existing_required,
                    'missing': missing_required,
                    'total': len(required_files)
                },
                'optional_files': {
                    'existing': existing_optional,
                    'missing': missing_optional,
                    'total': len(optional_files)
                },
                'issues_found': any(
                    not file_info['valid'] for file_info in existing_required
                ) or len(missing_required) > 0
            }
            
        except Exception as e:
            logger.error(f"❌ Validation failed for {series}{season}{episode}: {e}")
            return {
                'ready_for_recap': False,
                'readiness_score': 0,
                'error': str(e),
                'issues_found': True
            }
    
    @staticmethod
    def _validate_individual_file(file_path: str, file_type: str) -> Dict[str, Any]:
        """Validate an individual file based on its type."""
        try:
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)
            
            issues = []
            
            # Basic size checks
            if size_mb < 0.001:  # Less than 1KB
                issues.append(f"File too small: {size_mb:.3f} MB")
            
            # File-type specific validation
            if file_type == 'video_file':
                if size_mb < 10:  # Video files should be at least 10MB
                    issues.append(f"Video file suspiciously small: {size_mb:.1f} MB")
                
                # Check file extension
                if not file_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    issues.append("Video file has unexpected extension")
            
            elif file_type in ['plot_possible_speakers', 'season_plot']:
                if size_mb < 0.01:  # Plot files should be at least 10KB
                    issues.append(f"Plot file too small: {size_mb:.3f} MB")
                
                # Basic text file validation
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read(1000)  # Read first 1KB
                        if not content.strip():
                            issues.append("Plot file appears to be empty")
                except UnicodeDecodeError:
                    issues.append("Plot file encoding issues")
            
            elif file_type.endswith('_srt'):
                from .subtitle_utils import validate_subtitle_file
                srt_validation = validate_subtitle_file(file_path)
                if not srt_validation['valid']:
                    issues.append(f"SRT validation failed: {srt_validation.get('error', 'Unknown error')}")
            
            elif file_type.endswith('.json') or 'json' in file_type:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    issues.append(f"Invalid JSON format: {e}")
            
            return {
                'valid': len(issues) == 0,
                'size_mb': size_mb,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'valid': False,
                'size_mb': 0,
                'issues': [f"Validation error: {e}"]
            }
    
    @staticmethod
    def validate_recap_configuration(config: RecapConfiguration) -> Dict[str, Any]:
        """
        Validate recap configuration settings.
        
        Args:
            config: RecapConfiguration object
            
        Returns:
            Dictionary with validation results
        """
        try:
            issues = []
            warnings = []
            
            # Duration validation
            if config.target_duration_seconds < 30:
                issues.append("Target duration too short (minimum 30 seconds)")
            elif config.target_duration_seconds > 120:
                warnings.append("Target duration quite long (>2 minutes)")
            
            # Event count validation
            if config.max_events < 3:
                issues.append("Max events too low (minimum 3)")
            elif config.max_events > 15:
                warnings.append("Max events quite high (>15)")
            
            # Duration constraints
            if config.min_event_duration >= config.max_event_duration:
                issues.append("Min event duration must be less than max event duration")
            
            # Check if target duration is achievable
            min_total_duration = config.max_events * config.min_event_duration
            max_total_duration = config.max_events * config.max_event_duration
            
            if config.target_duration_seconds < min_total_duration:
                issues.append(
                    f"Target duration ({config.target_duration_seconds}s) too short for "
                    f"{config.max_events} events of {config.min_event_duration}s each"
                )
            
            if config.target_duration_seconds > max_total_duration:
                warnings.append(
                    f"Target duration ({config.target_duration_seconds}s) may be too long"
                )
            
            # Threshold validation
            if config.quality_threshold < 0.3:
                warnings.append("Quality threshold quite low (<0.3)")
            elif config.quality_threshold > 0.9:
                warnings.append("Quality threshold very high (>0.9) - may exclude too many clips")
            
            if config.relevance_threshold < 0.5:
                warnings.append("Relevance threshold quite low (<0.5)")
            elif config.relevance_threshold > 0.9:
                warnings.append("Relevance threshold very high (>0.9) - may exclude too many events")
            
            # Codec validation
            valid_video_codecs = ['libx264', 'libx265', 'h264', 'h265']
            if config.video_codec not in valid_video_codecs:
                warnings.append(f"Unusual video codec: {config.video_codec}")
            
            valid_audio_codecs = ['aac', 'mp3', 'ac3']
            if config.audio_codec not in valid_audio_codecs:
                warnings.append(f"Unusual audio codec: {config.audio_codec}")
            
            # FFmpeg preset validation
            valid_presets = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']
            if config.ffmpeg_preset not in valid_presets:
                warnings.append(f"Unknown FFmpeg preset: {config.ffmpeg_preset}")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings,
                'recommendation': ValidationUtils._generate_config_recommendations(config, issues, warnings)
            }
            
        except Exception as e:
            return {
                'valid': False,
                'issues': [f"Configuration validation error: {e}"],
                'warnings': [],
                'recommendation': "Fix validation errors before proceeding"
            }
    
    @staticmethod
    def _generate_config_recommendations(config: RecapConfiguration, 
                                       issues: List[str], 
                                       warnings: List[str]) -> str:
        """Generate configuration recommendations based on validation results."""
        if issues:
            return "Fix critical issues before proceeding with recap generation"
        
        if not warnings:
            return "Configuration looks good for recap generation"
        
        recommendations = []
        
        if config.target_duration_seconds > 90:
            recommendations.append("Consider reducing target duration for better viewer engagement")
        
        if config.max_events > 12:
            recommendations.append("Consider reducing max events to avoid information overload")
        
        if config.quality_threshold < 0.6:
            recommendations.append("Consider raising quality threshold for better output quality")
        
        return "; ".join(recommendations) if recommendations else "Configuration acceptable with minor warnings"
    
    @staticmethod
    def validate_json_schema(json_data: Dict[str, Any], schema_type: str) -> Dict[str, Any]:
        """
        Validate JSON data against predefined schemas.
        
        Args:
            json_data: JSON data to validate
            schema_type: Type of schema to validate against
            
        Returns:
            Dictionary with validation results
        """
        try:
            schema = ValidationUtils._get_json_schema(schema_type)
            
            # Validate against schema
            jsonschema.validate(json_data, schema)
            
            return {
                'valid': True,
                'schema_type': schema_type,
                'message': f"JSON data valid for schema: {schema_type}"
            }
            
        except jsonschema.ValidationError as e:
            return {
                'valid': False,
                'schema_type': schema_type,
                'error': str(e),
                'path': list(e.path) if e.path else [],
                'message': f"JSON schema validation failed: {e.message}"
            }
        except Exception as e:
            return {
                'valid': False,
                'schema_type': schema_type,
                'error': str(e),
                'message': f"Schema validation error: {e}"
            }
    
    @staticmethod
    def _get_json_schema(schema_type: str) -> Dict[str, Any]:
        """Get JSON schema definition for a given type."""
        schemas = {
            'present_running_plotlines': {
                "type": "object",
                "required": ["series", "season", "episode", "running_plotlines"],
                "properties": {
                    "series": {"type": "string"},
                    "season": {"type": "string"},
                    "episode": {"type": "string"},
                    "running_plotlines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "title", "description"],
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "main_characters": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            },
            'recap_metadata': {
                "type": "object",
                "required": ["id", "series", "season", "episode", "configuration", "events", "clips"],
                "properties": {
                    "id": {"type": "string"},
                    "series": {"type": "string"},
                    "season": {"type": "string"},
                    "episode": {"type": "string"},
                    "generated_at": {"type": "string"},
                    "configuration": {"type": "object"},
                    "events": {"type": "array"},
                    "clips": {"type": "array"},
                    "total_duration": {"type": "number"},
                    "success": {"type": "boolean"}
                }
            }
        }
        
        if schema_type not in schemas:
            raise ConfigurationError(f"Unknown schema type: {schema_type}")
        
        return schemas[schema_type]
    
    @staticmethod
    def validate_output_quality(recap_metadata: RecapMetadata) -> Dict[str, Any]:
        """
        Validate the quality of generated recap output.
        
        Args:
            recap_metadata: RecapMetadata object with generation results
            
        Returns:
            Dictionary with quality assessment
        """
        try:
            quality_checks = []
            passed_checks = []
            failed_checks = []
            
            # Duration compliance check
            target_duration = recap_metadata.configuration.target_duration_seconds
            actual_duration = recap_metadata.total_duration
            duration_diff = abs(actual_duration - target_duration)
            duration_tolerance = target_duration * 0.2  # 20% tolerance
            
            check_name = "duration_compliance"
            if duration_diff <= duration_tolerance:
                passed_checks.append(check_name)
                quality_checks.append({
                    'check': check_name,
                    'passed': True,
                    'message': f"Duration within tolerance: {actual_duration:.1f}s (target: {target_duration}s)"
                })
            else:
                failed_checks.append(check_name)
                quality_checks.append({
                    'check': check_name,
                    'passed': False,
                    'message': f"Duration outside tolerance: {actual_duration:.1f}s (target: {target_duration}s, diff: {duration_diff:.1f}s)"
                })
            
            # Event count check
            target_events = recap_metadata.configuration.max_events
            actual_events = len(recap_metadata.events)
            min_events = max(3, recap_metadata.configuration.max_events // 2)
            
            check_name = "event_count"
            if min_events <= actual_events <= target_events:
                passed_checks.append(check_name)
                quality_checks.append({
                    'check': check_name,
                    'passed': True,
                    'message': f"Event count appropriate: {actual_events} events"
                })
            else:
                failed_checks.append(check_name)
                quality_checks.append({
                    'check': check_name,
                    'passed': False,
                    'message': f"Event count suboptimal: {actual_events} events (target: {target_events}, min: {min_events})"
                })
            
            # Clip quality check
            check_name = "clip_quality"
            quality_threshold = recap_metadata.configuration.quality_threshold
            low_quality_clips = [
                clip for clip in recap_metadata.clips
                if clip.video_quality_score and clip.video_quality_score < quality_threshold
            ]
            
            if len(low_quality_clips) == 0:
                passed_checks.append(check_name)
                quality_checks.append({
                    'check': check_name,
                    'passed': True,
                    'message': f"All clips meet quality threshold ({quality_threshold})"
                })
            else:
                failed_checks.append(check_name)
                quality_checks.append({
                    'check': check_name,
                    'passed': False,
                    'message': f"{len(low_quality_clips)} clips below quality threshold"
                })
            
            # Processing efficiency check
            check_name = "processing_efficiency"
            if recap_metadata.processing_time_seconds and recap_metadata.processing_time_seconds > 0:
                processing_minutes = recap_metadata.processing_time_seconds / 60
                
                if processing_minutes < 10:  # Less than 10 minutes is good
                    passed_checks.append(check_name)
                    quality_checks.append({
                        'check': check_name,
                        'passed': True,
                        'message': f"Processing completed efficiently: {processing_minutes:.1f} minutes"
                    })
                else:
                    failed_checks.append(check_name)
                    quality_checks.append({
                        'check': check_name,
                        'passed': False,
                        'message': f"Processing took too long: {processing_minutes:.1f} minutes"
                    })
            
            # Overall quality score
            total_checks = len(quality_checks)
            passed_count = len(passed_checks)
            quality_score = passed_count / total_checks if total_checks > 0 else 0
            
            # Quality tier
            if quality_score >= 0.9:
                quality_tier = "excellent"
            elif quality_score >= 0.7:
                quality_tier = "good"
            elif quality_score >= 0.5:
                quality_tier = "acceptable"
            else:
                quality_tier = "poor"
            
            return {
                'overall_quality_score': quality_score,
                'quality_tier': quality_tier,
                'passed_checks': passed_checks,
                'failed_checks': failed_checks,
                'quality_checks': quality_checks,
                'total_checks': total_checks,
                'recommendation': ValidationUtils._generate_quality_recommendations(quality_tier, failed_checks)
            }
            
        except Exception as e:
            return {
                'overall_quality_score': 0,
                'quality_tier': 'error',
                'error': str(e),
                'recommendation': 'Fix validation errors and regenerate recap'
            }
    
    @staticmethod
    def _generate_quality_recommendations(quality_tier: str, failed_checks: List[str]) -> str:
        """Generate recommendations based on quality assessment."""
        if quality_tier == "excellent":
            return "Recap meets all quality standards - ready for use"
        
        if quality_tier == "good":
            return "Recap quality is good with minor issues - acceptable for use"
        
        recommendations = []
        
        if "duration_compliance" in failed_checks:
            recommendations.append("Adjust configuration or event selection to better match target duration")
        
        if "event_count" in failed_checks:
            recommendations.append("Review event selection criteria to achieve target event count")
        
        if "clip_quality" in failed_checks:
            recommendations.append("Increase quality threshold or improve source video quality")
        
        if "processing_efficiency" in failed_checks:
            recommendations.append("Optimize processing pipeline or reduce complexity")
        
        if not recommendations:
            recommendations.append("Review failed quality checks and adjust configuration accordingly")
        
        return "; ".join(recommendations)
