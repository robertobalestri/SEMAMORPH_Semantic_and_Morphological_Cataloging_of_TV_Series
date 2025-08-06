"""
Configuration validation for SEMAMORPH speaker identification system.
Validates threshold relationships and prevents configuration conflicts.
"""
import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from .config import config

logger = logging.getLogger(__name__)

@dataclass
class ValidationError:
    """Represents a configuration validation error."""
    field: str
    message: str
    severity: str  # "error", "warning", "info"
    suggested_fix: str

class ConfigValidator:
    """Validates configuration parameters for logical consistency and conflicts."""
    
    def __init__(self, config_obj=None):
        """Initialize validator with configuration object."""
        self.config = config_obj or config
    
    def validate_all(self) -> Tuple[List[ValidationError], List[ValidationError]]:
        """
        Validate all configuration parameters.
        
        Returns:
            Tuple of (errors, warnings) - errors must be fixed, warnings are recommended fixes
        """
        errors = []
        warnings = []
        
        # Validate threshold relationships
        thresh_errors, thresh_warnings = self.validate_threshold_relationships()
        errors.extend(thresh_errors)
        warnings.extend(thresh_warnings)
        
        # Validate face processing parameters
        face_errors, face_warnings = self.validate_face_processing()
        errors.extend(face_errors)
        warnings.extend(face_warnings)
        
        # Validate clustering parameters
        cluster_errors, cluster_warnings = self.validate_clustering_parameters()
        errors.extend(cluster_errors)
        warnings.extend(cluster_warnings)
        
        # Validate multi-face processing
        multi_errors, multi_warnings = self.validate_multiface_processing()
        errors.extend(multi_errors)
        warnings.extend(multi_warnings)
        
        # Validate sex validation parameters
        sex_errors, sex_warnings = self.validate_sex_validation()
        errors.extend(sex_errors)
        warnings.extend(sex_warnings)
        
        return errors, warnings
    
    def validate_threshold_relationships(self) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate that similarity thresholds have logical relationships."""
        errors = []
        warnings = []
        
        # Critical: Cluster merging should be <= clustering threshold
        if self.config.centroid_merge_threshold > self.config.cosine_similarity_threshold:
            errors.append(ValidationError(
                field="centroid_merge_threshold",
                message=f"centroid_merge_threshold ({self.config.centroid_merge_threshold}) cannot be higher than "
                       f"cosine_similarity_threshold ({self.config.cosine_similarity_threshold}). "
                       "Clusters that couldn't be created can't be merged.",
                severity="error",
                suggested_fix=f"Set centroid_merge_threshold <= {self.config.cosine_similarity_threshold}"
            ))
        
        # Warning: Ambiguity resolution typically higher than clustering
        if self.config.ambiguous_resolution_threshold <= self.config.cosine_similarity_threshold:
            warnings.append(ValidationError(
                field="ambiguous_resolution_threshold",
                message=f"ambiguous_resolution_threshold ({self.config.ambiguous_resolution_threshold}) is not higher than "
                       f"cosine_similarity_threshold ({self.config.cosine_similarity_threshold}). "
                       "This may resolve ambiguities too aggressively.",
                severity="warning",
                suggested_fix=f"Consider setting ambiguous_resolution_threshold > {self.config.cosine_similarity_threshold}"
            ))
        
        # Warning: LLM disambiguation should be higher than ambiguity resolution  
        if self.config.multiface_llm_disambiguation_threshold <= self.config.ambiguous_resolution_threshold:
            warnings.append(ValidationError(
                field="multiface_llm_disambiguation_threshold",
                message=f"multiface_llm_disambiguation_threshold ({self.config.multiface_llm_disambiguation_threshold}) should typically be higher than "
                       f"ambiguous_resolution_threshold ({self.config.ambiguous_resolution_threshold}). "
                       "LLM should only be triggered for the most difficult cases.",
                severity="warning",
                suggested_fix=f"Consider setting multiface_llm_disambiguation_threshold > {self.config.ambiguous_resolution_threshold}"
            ))
        
        # Outlier detection should be much lower than clustering
        if self.config.outlier_distance_threshold > self.config.cosine_similarity_threshold * 0.5:
            warnings.append(ValidationError(
                field="outlier_distance_threshold",
                message=f"outlier_distance_threshold ({self.config.outlier_distance_threshold}) seems high "
                       f"compared to cosine_similarity_threshold ({self.config.cosine_similarity_threshold}). "
                       "May not detect outliers effectively.",
                severity="warning",
                suggested_fix=f"Consider setting outlier_distance_threshold <= {self.config.cosine_similarity_threshold * 0.5:.2f}"
            ))
        
        # Cross-episode similarity should be reasonable
        if self.config.cross_episode_character_similarity_threshold < 0.5 or self.config.cross_episode_character_similarity_threshold > 0.95:
            warnings.append(ValidationError(
                field="cross_episode_character_similarity_threshold", 
                message=f"cross_episode_character_similarity_threshold ({self.config.cross_episode_character_similarity_threshold}) is outside "
                       "typical range (0.5-0.95). Very low values may cause false matches, very high values may miss matches.",
                severity="warning",
                suggested_fix="Consider setting cross_episode_character_similarity_threshold between 0.5 and 0.95"
            ))
        
        return errors, warnings
    
    def validate_face_processing(self) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate face processing parameters."""
        errors = []
        warnings = []
        
        # Face area ratio validation
        if self.config.face_min_area_ratio > 0.5:
            errors.append(ValidationError(
                field="face_min_area_ratio",
                message=f"face_min_area_ratio ({self.config.face_min_area_ratio}) > 0.5 would require faces "
                       "to cover 50%+ of frame. This is unrealistic for most TV content.",
                severity="error",
                suggested_fix="Set face_min_area_ratio <= 0.5, typically 0.01-0.1"
            ))
        
        if self.config.face_min_area_ratio < 0.005:
            warnings.append(ValidationError(
                field="face_min_area_ratio",
                message=f"face_min_area_ratio ({self.config.face_min_area_ratio}) is very low and may include "
                       "many false positive tiny faces.",
                severity="warning", 
                suggested_fix="Consider setting face_min_area_ratio >= 0.01 for better quality"
            ))
        
        # Confidence validation (note: 99% is intentionally conservative and correct)
        if self.config.face_min_confidence < 0.5:
            warnings.append(ValidationError(
                field="face_min_confidence",
                message=f"face_min_confidence ({self.config.face_min_confidence}) is low and may include many false positives.",
                severity="warning",
                suggested_fix="Consider higher confidence threshold for better precision"
            ))
        
        # Blur threshold validation
        if self.config.face_blur_threshold <= 0:
            errors.append(ValidationError(
                field="face_blur_threshold", 
                message=f"face_blur_threshold ({self.config.face_blur_threshold}) must be positive.",
                severity="error",
                suggested_fix="Set face_blur_threshold > 0, typically 3-10"
            ))
        
        # Eye validation parameters
        if self.config.face_enable_eye_validation:
            if self.config.face_eye_alignment_threshold <= 0 or self.config.face_eye_alignment_threshold > 180:
                errors.append(ValidationError(
                    field="face_eye_alignment_threshold",
                    message=f"face_eye_alignment_threshold ({self.config.face_eye_alignment_threshold}) must be between 0 and 180 degrees.",
                    severity="error",
                    suggested_fix="Set face_eye_alignment_threshold between 10-90 degrees"
                ))
            
            if self.config.face_eye_distance_threshold <= 0:
                errors.append(ValidationError(
                    field="face_eye_distance_threshold",
                    message=f"face_eye_distance_threshold ({self.config.face_eye_distance_threshold}) must be positive.",
                    severity="error",
                    suggested_fix="Set face_eye_distance_threshold > 0, typically 20-50"
                ))
        
        return errors, warnings
    
    def validate_clustering_parameters(self) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate clustering parameters."""
        errors = []
        warnings = []
        
        # Similarity threshold range
        if not (0.0 <= self.config.cosine_similarity_threshold <= 1.0):
            errors.append(ValidationError(
                field="cosine_similarity_threshold",
                message=f"cosine_similarity_threshold ({self.config.cosine_similarity_threshold}) must be between 0.0 and 1.0.",
                severity="error",
                suggested_fix="Set cosine_similarity_threshold between 0.0 and 1.0"
            ))
        
        # Minimum cluster size validation
        if self.config.min_cluster_size_final < 1:
            errors.append(ValidationError(
                field="min_cluster_size_final",
                message=f"min_cluster_size_final ({self.config.min_cluster_size_final}) must be at least 1.",
                severity="error",
                suggested_fix="Set min_cluster_size_final >= 1"
            ))
        
        if self.config.min_cluster_size_final > 10:
            warnings.append(ValidationError(
                field="min_cluster_size_final",
                message=f"min_cluster_size_final ({self.config.min_cluster_size_final}) is very high and may "
                       "exclude characters with limited screen time.",
                severity="warning",
                suggested_fix="Consider min_cluster_size_final <= 5 for better character coverage"
            ))
        
        # Spatial outlier parameters
        if self.config.enable_spatial_outlier_removal:
            if not (0.0 <= self.config.spatial_outlier_threshold <= 1.0):
                errors.append(ValidationError(
                    field="spatial_outlier_threshold",
                    message=f"spatial_outlier_threshold ({self.config.spatial_outlier_threshold}) must be between 0.0 and 1.0.",
                    severity="error",
                    suggested_fix="Set spatial_outlier_threshold between 0.0 and 1.0"
                ))
            
            if self.config.min_clusters_for_outlier_detection < 2:
                errors.append(ValidationError(
                    field="min_clusters_for_outlier_detection",
                    message=f"min_clusters_for_outlier_detection ({self.config.min_clusters_for_outlier_detection}) "
                           "must be at least 2 (need multiple clusters to detect outliers).",
                    severity="error",
                    suggested_fix="Set min_clusters_for_outlier_detection >= 2"
                ))
        
        return errors, warnings
    
    def validate_multiface_processing(self) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate multi-face processing parameters."""
        errors = []
        warnings = []
        
        if self.config.enable_multiface_processing:
            # Max faces per dialogue validation
            if self.config.multiface_max_faces_per_dialogue < 1:
                errors.append(ValidationError(
                    field="multiface_max_faces_per_dialogue",
                    message=f"multiface_max_faces_per_dialogue ({self.config.multiface_max_faces_per_dialogue}) must be at least 1.",
                    severity="error", 
                    suggested_fix="Set multiface_max_faces_per_dialogue >= 1"
                ))
            
            if self.config.multiface_max_faces_per_dialogue > 20:
                warnings.append(ValidationError(
                    field="multiface_max_faces_per_dialogue",
                    message=f"multiface_max_faces_per_dialogue ({self.config.multiface_max_faces_per_dialogue}) is very high and may "
                           "slow down processing significantly.",
                    severity="warning",
                    suggested_fix="Consider multiface_max_faces_per_dialogue <= 10 for better performance"
                ))
            
            # Cluster minimum occurrences validation
            if self.config.cluster_minimum_occurrences <= 0:
                errors.append(ValidationError(
                    field="cluster_minimum_occurrences",
                    message=f"cluster_minimum_occurrences ({self.config.cluster_minimum_occurrences}) must be positive.",
                    severity="error",
                    suggested_fix="Set cluster_minimum_occurrences > 0, typically 1-5"
                ))
        
        return errors, warnings
    
    def validate_sex_validation(self) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate sex validation parameters."""
        errors = []
        warnings = []
        
        if self.config.enable_sex_validation:
            # Validate sex confidence threshold (0-100%)
            if not (0.0 <= self.config.sex_confidence_threshold <= 100.0):
                errors.append(ValidationError(
                    field="sex_confidence_threshold",
                    message=f"sex_confidence_threshold ({self.config.sex_confidence_threshold}) must be between 0.0 and 100.0.",
                    severity="error",
                    suggested_fix="Set sex_confidence_threshold between 0.0 and 100.0"
                ))
            
            # Validate max faces for analysis
            if self.config.max_faces_for_sex_analysis < 1:
                errors.append(ValidationError(
                    field="max_faces_for_sex_analysis",
                    message=f"max_faces_for_sex_analysis ({self.config.max_faces_for_sex_analysis}) must be at least 1.",
                    severity="error",
                    suggested_fix="Set max_faces_for_sex_analysis >= 1"
                ))
            
            if self.config.max_faces_for_sex_analysis > 20:
                warnings.append(ValidationError(
                    field="max_faces_for_sex_analysis",
                    message=f"max_faces_for_sex_analysis ({self.config.max_faces_for_sex_analysis}) is very high and may impact performance.",
                    severity="warning",
                    suggested_fix="Consider max_faces_for_sex_analysis <= 10 for better performance"
                ))
            
            # Validate reassignment similarity threshold
            if not (0.0 <= self.config.sex_reassignment_similarity_threshold <= 1.0):
                errors.append(ValidationError(
                    field="sex_reassignment_similarity_threshold",
                    message=f"sex_reassignment_similarity_threshold ({self.config.sex_reassignment_similarity_threshold}) must be between 0.0 and 1.0.",
                    severity="error",
                    suggested_fix="Set sex_reassignment_similarity_threshold between 0.0 and 1.0"
                ))
        
        return errors, warnings
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get a comprehensive validation report."""
        errors, warnings = self.validate_all()
        
        # Group by severity
        error_fields = [e.field for e in errors]
        warning_fields = [w.field for w in warnings]
        
        # Calculate validation score (percentage of parameters that pass validation)
        total_parameters = 20  # Approximate number of key parameters
        failed_parameters = len(set(error_fields + warning_fields))
        validation_score = max(0, (total_parameters - failed_parameters) / total_parameters * 100)
        
        return {
            'validation_score': validation_score,
            'status': 'PASS' if not errors else 'FAIL',
            'errors': [{'field': e.field, 'message': e.message, 'fix': e.suggested_fix} for e in errors],
            'warnings': [{'field': w.field, 'message': w.message, 'fix': w.suggested_fix} for w in warnings],
            'error_count': len(errors),
            'warning_count': len(warnings),
            'recommendations': self._get_recommendations(errors, warnings)
        }
    
    def _get_recommendations(self, errors: List[ValidationError], warnings: List[ValidationError]) -> List[str]:
        """Generate configuration recommendations based on validation results."""
        recommendations = []
        
        if errors:
            recommendations.append(f"🚨 CRITICAL: Fix {len(errors)} configuration errors before running the pipeline.")
        
        if warnings:
            recommendations.append(f"⚠️ RECOMMENDED: Address {len(warnings)} configuration warnings for optimal performance.")
        
        # Specific recommendations based on common issues
        error_fields = [e.field for e in errors]
        warning_fields = [w.field for w in warnings]
        
        if 'centroid_merge_threshold' in error_fields:
            recommendations.append("🔧 Consider using the 'balanced' preset configuration for better threshold coordination.")
        
        if len(warning_fields) > 5:
            recommendations.append("🎛️ Consider using a preset configuration (conservative/balanced/aggressive) instead of manual tuning.")
        
        if 'min_cluster_size_final' in warning_fields:
            recommendations.append("👥 For series with many minor characters, consider reducing min_cluster_size_final to 1-2.")
        
        return recommendations

def validate_config(config_obj=None) -> Dict[str, Any]:
    """
    Convenience function to validate configuration.
    
    Args:
        config_obj: Configuration object to validate (uses global config if None)
        
    Returns:
        Validation report dictionary
    """
    validator = ConfigValidator(config_obj)
    return validator.get_validation_report()

# Command-line validation
if __name__ == "__main__":
    import sys
    
    print("🔍 SEMAMORPH Configuration Validation")
    print("=" * 50)
    
    try:
        report = validate_config()
        
        print(f"Validation Score: {report['validation_score']:.1f}%")
        print(f"Status: {report['status']}")
        print()
        
        if report['errors']:
            print("🚨 ERRORS (must fix):")
            for error in report['errors']:
                print(f"  • {error['field']}: {error['message']}")
                print(f"    Fix: {error['fix']}")
            print()
        
        if report['warnings']:
            print("⚠️ WARNINGS (recommended fixes):")
            for warning in report['warnings']:
                print(f"  • {warning['field']}: {warning['message']}")
                print(f"    Fix: {warning['fix']}")
            print()
        
        if report['recommendations']:
            print("💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  {rec}")
            print()
        
        if report['status'] == 'FAIL':
            print("❌ Configuration validation FAILED. Please fix errors before running the pipeline.")
            sys.exit(1)
        else:
            print("✅ Configuration validation PASSED!")
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        sys.exit(1) 