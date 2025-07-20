"""
Sex-based validation system for speaker clusters.
Uses DeepFace to analyze facial sex and validate cluster assignments against character biological sex.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter
import logging

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    
from ..config import config
from ..utils.logger_utils import setup_logging

logger = setup_logging(__name__)

class SexValidator:
    """
    Sex-based validation system for speaker clusters.
    
    This class implements the sex validation pipeline:
    1. Analyze facial sex using DeepFace
    2. Compare with character biological sex
    3. Invalidate mismatched clusters
    4. Attempt reassignment to same-sex characters
    """
    
    def __init__(self, path_handler):
        self.path_handler = path_handler
        self.character_sex_map = {}  # character_name -> biological_sex
        
        if not DEEPFACE_AVAILABLE:
            logger.warning("⚠️ DeepFace not available - sex validation will be disabled")
    
    def is_available(self) -> bool:
        """Check if sex validation is available (DeepFace installed and enabled)."""
        return DEEPFACE_AVAILABLE and config.enable_sex_validation
    
    def load_character_sex_data(self, characters_data: List[Dict]) -> None:
        """
        Load character biological sex data for validation.
        
        Args:
            characters_data: List of character data dictionaries with biological_sex field
        """
        self.character_sex_map.clear()
        
        for char_data in characters_data:
            character_name = char_data.get('best_appellation')
            biological_sex = char_data.get('biological_sex')
            
            if character_name and biological_sex in ['M', 'F']:
                self.character_sex_map[character_name] = biological_sex
                if config.enable_sex_validation_logging:
                    logger.debug(f"📊 [SEX-VALIDATION] Character sex loaded: {character_name} = {biological_sex}")
        
        logger.info(f"🧬 [SEX-VALIDATION] Loaded sex data for {len(self.character_sex_map)} characters")
    
    def analyze_cluster_sex(self, df_faces: pd.DataFrame, cluster_id: int) -> Optional[Dict]:
        """
        Analyze the biological sex of faces in a cluster using DeepFace.
        
        Args:
            df_faces: DataFrame containing face data
            cluster_id: ID of cluster to analyze
            
        Returns:
            Dictionary with sex analysis results or None if analysis fails
        """
        if not self.is_available():
            return None
        
        # Get faces for this cluster
        cluster_faces = df_faces[df_faces['cluster_id'] == cluster_id]
        
        if cluster_faces.empty:
            logger.warning(f"⚠️ [SEX-VALIDATION] No faces found for cluster {cluster_id}")
            return None
        
        # Limit to max_faces_for_sex_analysis for performance
        max_faces = config.max_faces_for_sex_analysis
        if len(cluster_faces) > max_faces:
            # Select best faces for sex analysis (prefer larger faces)
            if 'face_width' in cluster_faces.columns and 'face_height' in cluster_faces.columns:
                # Calculate face size and select largest faces
                cluster_faces['face_size'] = cluster_faces['face_width'] * cluster_faces['face_height']
                
                if config.enable_sex_validation_logging:
                    logger.info(f"🔧 [SEX-VALIDATION] Cluster {cluster_id}: Selecting top {max_faces} faces from {len(cluster_faces)} total faces by largest face size")
                    
                cluster_faces = cluster_faces.nlargest(max_faces, 'face_size')
                
                if config.enable_sex_validation_logging:
                    selected_paths = cluster_faces['image_path'].tolist()
                    selected_sizes = cluster_faces['face_size'].tolist()
                    selected_widths = cluster_faces['face_width'].tolist()
                    selected_heights = cluster_faces['face_height'].tolist()
                    logger.info(f"📋 [SEX-VALIDATION] Selected faces for cluster {cluster_id} (by face size):")
                    for i, (path, size, width, height) in enumerate(zip(selected_paths, selected_sizes, selected_widths, selected_heights)):
                        logger.info(f"   {i+1}. {path} (size: {int(size)} pixels, {int(width)}×{int(height)})")
            else:
                # Fallback to detection confidence if face dimensions not available
                if config.enable_sex_validation_logging:
                    logger.info(f"🔧 [SEX-VALIDATION] Cluster {cluster_id}: Selecting top {max_faces} faces from {len(cluster_faces)} total faces by highest detection confidence")
                    
                cluster_faces = cluster_faces.nlargest(max_faces, 'detection_confidence')
                
                if config.enable_sex_validation_logging:
                    selected_paths = cluster_faces['image_path'].tolist()
                    selected_confidences = cluster_faces['detection_confidence'].tolist()
                    logger.info(f"📋 [SEX-VALIDATION] Selected faces for cluster {cluster_id} (by detection confidence):")
                    for i, (path, conf) in enumerate(zip(selected_paths, selected_confidences)):
                        logger.info(f"   {i+1}. {path} (detection_conf: {conf:.1f}%)")
        else:
            if config.enable_sex_validation_logging:
                logger.info(f"🔧 [SEX-VALIDATION] Cluster {cluster_id}: Using all {len(cluster_faces)} faces (within limit)")
        
        sex_results = []
        analyzed_faces = 0
        skipped_faces = []
        
        for _, face_row in cluster_faces.iterrows():
            image_path = face_row.get('image_path')
            
            if config.enable_sex_validation_logging:
                logger.info(f"🔍 [SEX-VALIDATION] Processing face: {image_path}")
            
            if not image_path:
                skip_reason = "No image path provided"
                if config.enable_sex_validation_logging:
                    logger.warning(f"⚠️ [SEX-VALIDATION] Face skipped: {skip_reason}")
                skipped_faces.append({
                    "image_path": "Unknown",
                    "skip_reason": skip_reason
                })
                continue
                
            if not os.path.exists(image_path):
                skip_reason = f"File does not exist - {image_path}"
                if config.enable_sex_validation_logging:
                    logger.warning(f"⚠️ [SEX-VALIDATION] Face skipped: {skip_reason}")
                skipped_faces.append({
                    "image_path": image_path,
                    "skip_reason": skip_reason
                })
                continue
            
            try:
                # Analyze sex using DeepFace
                result = DeepFace.analyze(
                    img_path=image_path,
                    actions=['gender'],
                    enforce_detection=False  # Don't fail if face not detected perfectly
                )
                
                # Handle both list and dict returns from DeepFace
                if isinstance(result, list):
                    result_dict = result[0] if result else {}
                else:
                    result_dict = result
                
                gender_data = result_dict.get('gender', {}) if isinstance(result_dict, dict) else {}
                if config.enable_sex_validation_logging:
                    logger.info(f"🧠 [SEX-VALIDATION] DeepFace result for {image_path}: {gender_data}")
                
                if gender_data:
                    male_confidence = gender_data.get('Man', 0)
                    female_confidence = gender_data.get('Woman', 0)
                    
                    # Calculate individual face confidence (highest of the two)
                    individual_confidence = max(male_confidence, female_confidence)
                    predicted_individual = 'M' if male_confidence > female_confidence else 'F'
                    confidence_diff = abs(male_confidence - female_confidence)
                    
                    # Check if this face meets the minimum confidence threshold from config
                    min_individual_confidence = config.sex_confidence_threshold
                    
                    if individual_confidence >= min_individual_confidence:
                        sex_results.append({
                            'image_path': image_path,
                            'male_confidence': male_confidence,
                            'female_confidence': female_confidence,
                            'predicted_sex': predicted_individual,
                            'confidence_difference': confidence_diff,
                            'individual_confidence': individual_confidence
                        })
                        analyzed_faces += 1
                        
                        if config.enable_sex_validation_logging:
                            logger.info(f"🔬 [SEX-VALIDATION] Face {image_path}: Male={male_confidence:.1f}%, Female={female_confidence:.1f}% → Predicted: {predicted_individual} (conf: {individual_confidence:.1f}%) ✅ ACCEPTED")
                    else:
                        skip_reason = f"Confidence {individual_confidence:.1f}% below threshold {min_individual_confidence:.0f}% (Male={male_confidence:.1f}%, Female={female_confidence:.1f}%)"
                        if config.enable_sex_validation_logging:
                            logger.info(f"🔬 [SEX-VALIDATION] Face {image_path}: Male={male_confidence:.1f}%, Female={female_confidence:.1f}% → Predicted: {predicted_individual} (conf: {individual_confidence:.1f}%) ❌ SKIPPED (< {min_individual_confidence:.0f}%)")
                        skipped_faces.append({
                            "image_path": image_path,
                            "skip_reason": skip_reason,
                            "male_confidence": male_confidence,
                            "female_confidence": female_confidence,
                            "predicted_sex": predicted_individual,
                            "individual_confidence": individual_confidence
                        })
                else:
                    skip_reason = f"No gender data returned by DeepFace for {image_path}"
                    if config.enable_sex_validation_logging:
                        logger.warning(f"⚠️ [SEX-VALIDATION] Face skipped: {skip_reason}")
                    skipped_faces.append({
                        "image_path": image_path,
                        "skip_reason": skip_reason
                    })
                
            except Exception as e:
                skip_reason = f"DeepFace analysis failed for {image_path}: {str(e)}"
                if config.enable_sex_validation_logging:
                    logger.warning(f"⚠️ [SEX-VALIDATION] Face skipped: {skip_reason}")
                skipped_faces.append({
                    "image_path": image_path,
                    "skip_reason": skip_reason
                })
                continue
        
        if not sex_results:
            logger.warning(f"⚠️ [SEX-VALIDATION] No faces could be analyzed for cluster {cluster_id}")
            return None
        
        # Aggregate results
        total_male_confidence = sum(r['male_confidence'] for r in sex_results)
        total_female_confidence = sum(r['female_confidence'] for r in sex_results)
        avg_male_confidence = total_male_confidence / len(sex_results)
        avg_female_confidence = total_female_confidence / len(sex_results)
        
        confidence_difference = abs(avg_male_confidence - avg_female_confidence)
        predicted_sex = 'M' if avg_male_confidence > avg_female_confidence else 'F'
        
        # Check if confidence difference meets threshold
        reliable_prediction = confidence_difference >= config.sex_confidence_threshold
        
        result = {
            'cluster_id': cluster_id,
            'faces_analyzed': analyzed_faces,
            'faces_skipped': len(skipped_faces),
            'avg_male_confidence': avg_male_confidence,
            'avg_female_confidence': avg_female_confidence,
            'confidence_difference': confidence_difference,
            'predicted_sex': predicted_sex,
            'reliable_prediction': reliable_prediction,
            'individual_results': sex_results,
            'skipped_faces': skipped_faces
        }
        
        if config.enable_sex_validation_logging:
            logger.info(f"🧬 [SEX-VALIDATION] Cluster {cluster_id}: {predicted_sex} ({confidence_difference:.1f}% confidence, {analyzed_faces} faces)")
        
        return result
    
    def validate_cluster_sex(self, cluster_id: int, character_name: str, sex_analysis: Dict) -> Dict:
        """
        Validate a cluster's sex analysis against the assigned character's biological sex.
        
        Args:
            cluster_id: ID of cluster being validated
            character_name: Name of character assigned to cluster
            sex_analysis: Result from analyze_cluster_sex()
            
        Returns:
            Dictionary with validation results
        """
        if not sex_analysis or not sex_analysis.get('reliable_prediction'):
            return {
                'cluster_id': cluster_id,
                'character_name': character_name,
                'validation_status': 'UNRELIABLE_SEX_DETECTION',
                'action': 'INVALIDATE',
                'reason': 'Sex detection confidence below threshold',
                'character_sex': self.character_sex_map.get(character_name),
                'detected_sex': sex_analysis.get('predicted_sex') if sex_analysis else None,
                'confidence_difference': sex_analysis.get('confidence_difference', 0) if sex_analysis else 0
            }
        
        character_sex = self.character_sex_map.get(character_name)
        detected_sex = sex_analysis.get('predicted_sex')
        
        if not character_sex:
            return {
                'cluster_id': cluster_id,
                'character_name': character_name,
                'validation_status': 'NO_CHARACTER_SEX_DATA',
                'action': 'KEEP',
                'reason': 'No biological sex data available for character',
                'character_sex': None,
                'detected_sex': detected_sex,
                'confidence_difference': sex_analysis.get('confidence_difference', 0)
            }
        
        sex_matches = character_sex == detected_sex
        
        if sex_matches:
            result = {
                'cluster_id': cluster_id,
                'character_name': character_name,
                'validation_status': 'SEX_MATCH',
                'action': 'KEEP',
                'reason': f'Detected sex ({detected_sex}) matches character sex ({character_sex})',
                'character_sex': character_sex,
                'detected_sex': detected_sex,
                'confidence_difference': sex_analysis.get('confidence_difference', 0)
            }
        else:
            result = {
                'cluster_id': cluster_id,
                'character_name': character_name,
                'validation_status': 'SEX_MISMATCH',
                'action': 'INVALIDATE',
                'reason': f'Detected sex ({detected_sex}) does not match character sex ({character_sex})',
                'character_sex': character_sex,
                'detected_sex': detected_sex,
                'confidence_difference': sex_analysis.get('confidence_difference', 0)
            }
        
        if config.enable_sex_validation_logging:
            status_emoji = "✅" if sex_matches else "❌"
            logger.info(f"{status_emoji} [SEX-VALIDATION] Cluster {cluster_id} → {character_name}: {result['validation_status']}")
            logger.info(f"📊 [SEX-VALIDATION] Comparison details:")
            logger.info(f"   Character '{character_name}' biological sex: {character_sex}")
            logger.info(f"   Cluster {cluster_id} detected sex: {detected_sex}")
            logger.info(f"   Confidence difference: {sex_analysis.get('confidence_difference', 0):.1f}%")
            logger.info(f"   Action: {result['action']}")
        
        return result
    
    def find_same_sex_reassignment_candidates(
        self, 
        cluster_id: int, 
        detected_sex: str, 
        current_character: str,
        clusters_dict: Dict,
        character_medians: Dict
    ) -> List[Dict]:
        """
        Find potential reassignment candidates with matching biological sex.
        
        Args:
            cluster_id: ID of cluster to reassign
            detected_sex: Sex detected by DeepFace ('M' or 'F')
            current_character: Currently assigned character (to exclude)
            clusters_dict: Dictionary of cluster information
            character_medians: Dictionary of character median embeddings
            
        Returns:
            List of potential reassignment candidates sorted by similarity
        """
        candidates = []
        
        # Find all characters with matching biological sex
        matching_sex_characters = [
            char_name for char_name, char_sex in self.character_sex_map.items()
            if char_sex == detected_sex and char_name != current_character
        ]
        
        if not matching_sex_characters:
            if config.enable_sex_validation_logging:
                logger.debug(f"🔍 [SEX-VALIDATION] No characters with matching sex ({detected_sex}) found for reassignment")
            return candidates
        
        # Get cluster median embedding for similarity calculation
        cluster_info = clusters_dict.get(cluster_id, {})
        cluster_median = cluster_info.get('median_embedding')
        
        if cluster_median is None:
            if config.enable_sex_validation_logging:
                logger.debug(f"⚠️ [SEX-VALIDATION] No cluster median available for cluster {cluster_id}")
            return candidates
        
        # Calculate similarity to each same-sex character
        for char_name in matching_sex_characters:
            char_median = character_medians.get(char_name)
            
            if char_median is not None:
                # Calculate cosine similarity
                similarity = np.dot(cluster_median, char_median) / (
                    np.linalg.norm(cluster_median) * np.linalg.norm(char_median)
                )
                
                if similarity >= config.sex_reassignment_similarity_threshold:
                    candidates.append({
                        'character_name': char_name,
                        'similarity': float(similarity),
                        'biological_sex': detected_sex
                    })
        
        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        if config.enable_sex_validation_logging and candidates:
            logger.info(f"🔄 [SEX-VALIDATION] Found {len(candidates)} reassignment candidates for cluster {cluster_id}")
            for candidate in candidates[:3]:  # Log top 3
                logger.info(f"   → {candidate['character_name']} (similarity: {candidate['similarity']:.3f})")
        
        return candidates
    
    def validate_all_clusters(
        self, 
        df_faces: pd.DataFrame, 
        clusters_dict: Dict,
        character_medians: Dict
    ) -> Dict:
        """
        Validate all clusters using sex-based analysis.
        
        Args:
            df_faces: DataFrame containing face data
            clusters_dict: Dictionary of cluster information
            character_medians: Dictionary of character median embeddings
            
        Returns:
            Dictionary containing validation results and recommended actions
        """
        if not self.is_available():
            logger.info("🧬 [SEX-VALIDATION] Skipped - DeepFace not available or sex validation disabled")
            return {
                'validation_enabled': False,
                'clusters_validated': 0,
                'clusters_kept': 0,
                'clusters_invalidated': 0,
                'clusters_reassigned': 0,
                'validation_details': {}
            }
        
        logger.info(f"🧬 [SEX-VALIDATION] Starting validation of {len(clusters_dict)} clusters")
        
        validation_results = {
            'validation_enabled': True,
            'clusters_validated': 0,
            'clusters_kept': 0,
            'clusters_invalidated': 0,
            'clusters_reassigned': 0,
            'validation_details': {},
            'reassignment_actions': {}
        }
        
        for cluster_id, cluster_info in clusters_dict.items():
            character_name = cluster_info.get('character_name')
            
            # Skip clusters without character assignments or with invalid statuses
            if not character_name or cluster_info.get('cluster_status') in ['SPATIAL_OUTLIER', 'SPATIAL_OUTLIER_FROM_CHARACTER_MEDIAN', 'AMBIGUOUS', 'INSUFFICIENT_EVIDENCE']:
                continue
            
            # Analyze cluster sex
            sex_analysis = self.analyze_cluster_sex(df_faces, cluster_id)
            
            # Validate against character sex - skip if no sex analysis available
            if sex_analysis is None:
                logger.warning(f"⚠️ [SEX-VALIDATION] No sex analysis available for cluster {cluster_id}")
                continue
                
            validation_result = self.validate_cluster_sex(cluster_id, character_name, sex_analysis)
            
            validation_results['validation_details'][cluster_id] = validation_result
            validation_results['clusters_validated'] += 1
            
            action = validation_result['action']
            
            if action == 'KEEP':
                validation_results['clusters_kept'] += 1
            
            elif action == 'INVALIDATE':
                validation_results['clusters_invalidated'] += 1
                
                # Try to find reassignment candidates if we have reliable sex detection
                if sex_analysis and sex_analysis.get('reliable_prediction'):
                    detected_sex = sex_analysis.get('predicted_sex')
                    if detected_sex:  # Ensure detected_sex is not None
                        candidates = self.find_same_sex_reassignment_candidates(
                            cluster_id, detected_sex, character_name, clusters_dict, character_medians
                        )
                        
                    if candidates:
                        # Recommend reassignment to best candidate
                        best_candidate = candidates[0]
                        validation_results['reassignment_actions'][cluster_id] = {
                            'from_character': character_name,
                            'to_character': best_candidate['character_name'],
                            'similarity': best_candidate['similarity'],
                            'detected_sex': detected_sex,
                            'reason': 'Sex-based reassignment to matching character'
                        }
                        validation_results['clusters_reassigned'] += 1
        
        # Log summary
        logger.info(f"🧬 [SEX-VALIDATION] Validation complete:")
        logger.info(f"   📊 Validated: {validation_results['clusters_validated']} clusters")
        logger.info(f"   ✅ Kept: {validation_results['clusters_kept']} clusters")
        logger.info(f"   ❌ Invalidated: {validation_results['clusters_invalidated']} clusters")
        logger.info(f"   🔄 Reassignment candidates: {validation_results['clusters_reassigned']} clusters")
        
        # Generate debug file if enabled
        if config.enable_sex_validation_logging:
            self._save_sex_validation_debug_file(validation_results, df_faces, clusters_dict)
        
        return validation_results
    
    def _save_sex_validation_debug_file(self, validation_results: Dict, df_faces: pd.DataFrame, clusters_dict: Dict) -> None:
        """
        Generate a comprehensive debug file for sex validation analysis.
        
        Args:
            validation_results: Results from validate_all_clusters()
            df_faces: DataFrame containing face data
            clusters_dict: Dictionary of cluster information
        """
        try:
            import json
            from datetime import datetime
            
            debug_data = {
                "timestamp": datetime.now().isoformat(),
                "episode_code": self.path_handler.get_episode_code(),
                "validation_summary": {
                    "validation_enabled": validation_results.get('validation_enabled', False),
                    "clusters_validated": validation_results.get('clusters_validated', 0),
                    "clusters_kept": validation_results.get('clusters_kept', 0),
                    "clusters_invalidated": validation_results.get('clusters_invalidated', 0),
                    "clusters_reassigned": validation_results.get('clusters_reassigned', 0)
                },
                "configuration": {
                    "max_faces_for_analysis": config.max_faces_for_sex_analysis,
                    "sex_confidence_threshold": config.sex_confidence_threshold,
                    "sex_reassignment_similarity_threshold": config.sex_reassignment_similarity_threshold,
                    "min_individual_confidence": 95.0  # Hardcoded in the code
                },
                "character_sex_mapping": dict(self.character_sex_map),
                "clusters_analyzed": {},
                "validation_details": validation_results.get('validation_details', {}),
                "reassignment_actions": validation_results.get('reassignment_actions', {})
            }
            
            # Analyze each cluster in detail
            for cluster_id, cluster_info in clusters_dict.items():
                character_name = cluster_info.get('character_name')
                
                if not character_name or cluster_info.get('cluster_status') in ['SPATIAL_OUTLIER', 'SPATIAL_OUTLIER_FROM_CHARACTER_MEDIAN', 'AMBIGUOUS', 'INSUFFICIENT_EVIDENCE']:
                    continue
                
                # Get cluster faces
                cluster_faces = df_faces[df_faces['cluster_id'] == cluster_id] if 'cluster_id' in df_faces.columns else df_faces[df_faces['face_id'] == cluster_id]
                
                cluster_debug = {
                    "cluster_info": {
                        "cluster_id": cluster_id,
                        "character_name": character_name,
                        "character_biological_sex": self.character_sex_map.get(character_name, "Unknown"),
                        "cluster_status": cluster_info.get('cluster_status', 'Unknown'),
                        "face_count": cluster_info.get('face_count', 0),
                        "cluster_confidence": cluster_info.get('cluster_confidence', 0.0)
                    },
                    "face_analysis": {
                        "total_faces_in_cluster": len(cluster_faces),
                        "faces_selected_for_analysis": min(len(cluster_faces), config.max_faces_for_sex_analysis),
                        "faces_analyzed": 0,
                        "faces_skipped": 0,
                        "face_details": [],
                        "skipped_faces": []
                    },
                    "sex_analysis_results": None,
                    "validation_result": validation_results.get('validation_details', {}).get(cluster_id, None),
                    "reassignment_candidate": validation_results.get('reassignment_actions', {}).get(cluster_id, None)
                }
                
                # Analyze individual faces if we have them
                if not cluster_faces.empty:
                    # Simulate the face selection logic
                    max_faces = config.max_faces_for_sex_analysis
                    selected_faces = cluster_faces
                    
                    if len(cluster_faces) > max_faces:
                        if 'face_width' in cluster_faces.columns and 'face_height' in cluster_faces.columns:
                            cluster_faces_copy = cluster_faces.copy()
                            cluster_faces_copy['face_size'] = cluster_faces_copy['face_width'] * cluster_faces_copy['face_height']
                            selected_faces = cluster_faces_copy.nlargest(max_faces, 'face_size')
                            cluster_debug["face_analysis"]["selection_method"] = "largest_face_size"
                        else:
                            selected_faces = cluster_faces.nlargest(max_faces, 'detection_confidence')
                            cluster_debug["face_analysis"]["selection_method"] = "highest_detection_confidence"
                    else:
                        cluster_debug["face_analysis"]["selection_method"] = "all_faces_used"
                    
                    # Record face details
                    for idx, (_, face_row) in enumerate(selected_faces.iterrows()):
                        image_path = face_row.get('image_path', 'Unknown')
                        face_detail = {
                            "face_index": idx + 1,
                            "image_path": image_path,
                            "detection_confidence": face_row.get('detection_confidence', 0.0),
                            "file_exists": os.path.exists(image_path) if image_path != 'Unknown' else False,
                            "sex_analysis": "Would be analyzed with DeepFace here"
                        }
                        
                        if 'face_width' in face_row and 'face_height' in face_row:
                            face_detail["face_dimensions"] = {
                                "width": face_row.get('face_width', 0),
                                "height": face_row.get('face_height', 0),
                                "area": face_row.get('face_width', 0) * face_row.get('face_height', 0)
                            }
                        
                        cluster_debug["face_analysis"]["face_details"].append(face_detail)
                    
                    # Get actual sex analysis results if available
                    sex_analysis = self.analyze_cluster_sex(df_faces, cluster_id)
                    if sex_analysis:
                        # Update face analysis stats with actual results
                        cluster_debug["face_analysis"]["faces_analyzed"] = sex_analysis.get('faces_analyzed', 0)
                        cluster_debug["face_analysis"]["faces_skipped"] = sex_analysis.get('faces_skipped', 0)
                        cluster_debug["face_analysis"]["skipped_faces"] = sex_analysis.get('skipped_faces', [])
                        cluster_debug["sex_analysis_results"] = {
                            "faces_analyzed": sex_analysis.get('faces_analyzed', 0),
                            "avg_male_confidence": sex_analysis.get('avg_male_confidence', 0.0),
                            "avg_female_confidence": sex_analysis.get('avg_female_confidence', 0.0),
                            "confidence_difference": sex_analysis.get('confidence_difference', 0.0),
                            "predicted_sex": sex_analysis.get('predicted_sex', 'Unknown'),
                            "reliable_prediction": sex_analysis.get('reliable_prediction', False),
                            "individual_face_results": []
                        }
                        
                        # Add individual face results if available
                        for result in sex_analysis.get('individual_results', []):
                            cluster_debug["sex_analysis_results"]["individual_face_results"].append({
                                "image_path": result.get('image_path', ''),
                                "male_confidence": result.get('male_confidence', 0.0),
                                "female_confidence": result.get('female_confidence', 0.0),
                                "predicted_sex": result.get('predicted_sex', ''),
                                "individual_confidence": result.get('individual_confidence', 0.0),
                                "confidence_difference": result.get('confidence_difference', 0.0)
                            })
                
                debug_data["clusters_analyzed"][str(cluster_id)] = cluster_debug
            
            # Save debug file
            debug_file_path = os.path.join(
                self.path_handler.base_dir,
                self.path_handler.series,
                self.path_handler.season,
                self.path_handler.episode,
                f"{self.path_handler.get_episode_code()}_sex_validation_debug.json"
            )
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            debug_data_clean = convert_numpy_types(debug_data)
            
            with open(debug_file_path, 'w', encoding='utf-8') as f:
                json.dump(debug_data_clean, f, indent=2, ensure_ascii=False)
            
            logger.info(f"🧬 [SEX-VALIDATION-DEBUG] Debug file saved: {debug_file_path}")
            logger.info(f"🧬 [SEX-VALIDATION-DEBUG] File contains analysis of {len(debug_data['clusters_analyzed'])} clusters")
            
        except Exception as e:
            logger.error(f"❌ [SEX-VALIDATION-DEBUG] Failed to save debug file: {e}")
            import traceback
            logger.error(f"❌ [SEX-VALIDATION-DEBUG] Traceback: {traceback.format_exc()}") 