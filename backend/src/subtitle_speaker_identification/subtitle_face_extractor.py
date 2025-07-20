"""
Face extraction system adapted for subtitle-based face detection.
Extracts faces from original video at dialogue midpoints.
"""
import os
import re
import cv2
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from PIL import Image

# Configure TensorFlow before importing DeepFace
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

# Configure GPU memory growth to prevent CUDA out-of-memory errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")

from deepface  import DeepFace 
from tqdm import tqdm

from ..narrative_storage_management.narrative_models import DialogueLine
from ..utils.logger_utils import setup_logging
from ..config import config

logger = setup_logging(__name__)

class SubtitleFaceExtractor:
    """Extracts faces from video at subtitle dialogue midpoints."""
    
    def __init__(self, path_handler):
        self.path_handler = path_handler
        
    def extract_faces_from_subtitles(
        self,
        dialogue_lines: List[DialogueLine],
        detector: str = config.face_detector,
        min_confidence: float = config.face_min_confidence,
        min_face_area_ratio: float = config.face_min_area_ratio,
        blur_threshold: float = config.face_blur_threshold,
        enable_eye_validation: bool = config.face_enable_eye_validation,
        eye_alignment_threshold: float = config.face_eye_alignment_threshold,
        eye_distance_threshold: float = config.face_eye_distance_threshold,
        force_extract: bool = False
    ) -> pd.DataFrame:
        """
        Extract faces from video at dialogue midpoints.
        
        Args:
            dialogue_lines: List of dialogue lines with timestamps
            detector: DeepFace detector backend
            min_confidence: Minimum face detection confidence
            min_face_area_ratio: Minimum face area relative to frame
            blur_threshold: Laplacian variance threshold for blur detection
            enable_eye_validation: Enable facial landmark validation
            eye_alignment_threshold: Maximum Y-axis difference between eyes
            eye_distance_threshold: Minimum X-axis distance between eyes
            force_extract: If True, re-extract even if files exist
            
        Returns:
            DataFrame with face metadata
        """
        logger.info(f"🎬 Extracting faces from {len(dialogue_lines)} dialogue lines")
        
        # Setup paths
        video_path = self.path_handler.get_video_file_path()
        faces_dir = self.path_handler.get_dialogue_faces_dir()
        frames_dir = self.path_handler.get_dialogue_frames_dir()
        faces_csv_path = self.path_handler.get_dialogue_faces_csv_path()
        
        # Create directories
        os.makedirs(faces_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        
        # Initialize debug file
        debug_file_path = self.path_handler.get_dialogue_faces_debug_path()
        debug_data = []
        
        # Check if we should skip extraction
        if not force_extract and os.path.exists(faces_csv_path):
            logger.info(f"📂 Loading existing face data from: {faces_csv_path}")
            try:
                df_faces = pd.read_csv(faces_csv_path)
                logger.info(f"✅ Loaded {len(df_faces)} existing face records")
                return df_faces
            except Exception as e:
                logger.warning(f"⚠️ Error loading existing CSV: {e}")
        
        # Load existing face data for resume functionality
        existing_face_records = []
        processed_dialogues = set()
        
        if not force_extract and os.path.exists(faces_csv_path):
            try:
                existing_df = pd.read_csv(faces_csv_path)
                existing_face_records = existing_df.to_dict('records')
                processed_dialogues = set(existing_df['dialogue_index'].unique())
                logger.info(f"🔄 Found {len(existing_face_records)} existing face records for {len(processed_dialogues)} dialogues")
            except Exception as e:
                logger.warning(f"⚠️ Error loading existing CSV for resume: {e}")
        
        # Also check for processed frame files (backup method)
        if not force_extract and os.path.exists(frames_dir):
            existing_frames = list(Path(frames_dir).glob("dialogue_*_frame.png"))
            for frame_file in existing_frames:
                # Extract dialogue index from filename like "dialogue_0042_frame.png"
                match = re.search(r'dialogue_(\d+)_frame\.png', frame_file.name)
                if match:
                    processed_dialogues.add(int(match.group(1)))
            
            if processed_dialogues:
                logger.info(f"🔄 Total processed dialogues from CSV and frames: {len(processed_dialogues)}")
        
        # Validate video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"❌ Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"❌ Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"📹 Video info: {fps:.1f} FPS, {total_frames} frames, {duration:.1f}s duration")
        logger.info(f"🎬 Starting face extraction from {len(dialogue_lines)} dialogue lines")
        
        # Show current blur detection configuration
        method = config.blur_detection_method
        if method == 'gradient':
            threshold = config.blur_threshold_gradient
        elif method == 'tenengrad':
            threshold = config.blur_threshold_tenengrad
        elif method == 'composite':
            threshold = config.blur_threshold_composite
        else:
            threshold = blur_threshold
        
        logger.info(f"⚙️ Settings: detector={detector}, min_conf={min_confidence}, blur_method={method}, blur_threshold={threshold}")
        
        # Extract faces
        face_records = []
        processed_count = 0
        skipped_count = 0
        total_faces_found = 0
        
        # Progress logging every N dialogues
        progress_interval = max(1, len(dialogue_lines) // 20)  # Log progress 20 times during processing
        
        for dialogue in tqdm(dialogue_lines, desc="Extracting faces"):
            try:
                logger.debug(f"🔍 Processing dialogue {dialogue.index}: '{dialogue.text[:50]}...' (speaker: {dialogue.speaker})")
                
                # Skip if already processed (unless force_extract)
                if not force_extract and dialogue.index in processed_dialogues:
                    logger.debug(f"⏭️ Skipping dialogue {dialogue.index} - already processed")
                    skipped_count += 1
                    continue
                
                # Calculate midpoint timestamp
                midpoint_seconds = (dialogue.start_time + dialogue.end_time) / 2.0
                
                # Skip if beyond video duration
                if midpoint_seconds >= duration:
                    logger.debug(f"⚠️ Dialogue {dialogue.index} midpoint ({midpoint_seconds:.1f}s) beyond video duration")
                    skipped_count += 1
                    continue
                
                # Extract frame at midpoint
                frame = self._extract_frame_at_timestamp(cap, midpoint_seconds, fps)
                if frame is None:
                    logger.debug(f"⚠️ Could not extract frame for dialogue {dialogue.index}")
                    skipped_count += 1
                    continue
                
                # Save frame
                frame_filename = f"dialogue_{dialogue.index:04d}_frame.png"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                # Detect faces
                faces = self._detect_faces_in_frame(
                    frame, 
                    detector, 
                    min_confidence, 
                    min_face_area_ratio,
                    blur_threshold,
                    enable_eye_validation,
                    eye_alignment_threshold,
                    eye_distance_threshold
                )
                
                logger.debug(f"👥 Dialogue {dialogue.index}: Found {len(faces)} valid faces")
                total_faces_found += len(faces)
                
                # Create debug entry for this dialogue
                debug_entry = {
                    'dialogue_index': dialogue.index,
                    'timestamp_seconds': midpoint_seconds,
                    'dialogue_text': dialogue.text,
                    'speaker': dialogue.speaker,
                    'is_llm_confident': dialogue.is_llm_confident,
                    'scene_number': dialogue.scene_number,
                    'frame_filename': frame_filename,
                    'total_faces_detected': len(faces),
                    'faces': []
                }
                
                # Save face crops and create records
                for face_idx, face_data in enumerate(faces):
                    face_filename = f"dialogue_{dialogue.index:04d}_face_{face_idx:02d}.png"
                    face_path = os.path.join(faces_dir, face_filename)
                    
                    # Save face crop as RGB PNG
                    face_crop_rgb = cv2.cvtColor(face_data['crop'], cv2.COLOR_BGR2RGB)
                    face_image = Image.fromarray(face_crop_rgb)
                    face_image.save(face_path)
                    
                    # Add detailed face info to debug entry
                    face_debug_info = {
                        'face_index': face_idx,
                        'face_filename': face_filename,
                        'detection_confidence': face_data['confidence'],
                        'blur_score': face_data['blur_score'],
                        'bounding_box': {
                            'x': face_data['region']['x'],
                            'y': face_data['region']['y'],
                            'width': face_data['region']['w'],
                            'height': face_data['region']['h']
                        },
                        'face_area_pixels': face_data['region']['w'] * face_data['region']['h'],
                        'face_area_ratio': (face_data['region']['w'] * face_data['region']['h']) / (frame.shape[0] * frame.shape[1])
                    }
                    
                    # Add eye information if available
                    if 'left_eye' in face_data['region'] and 'right_eye' in face_data['region']:
                        left_eye = face_data['region']['left_eye']
                        right_eye = face_data['region']['right_eye']
                        face_debug_info['eyes'] = {
                            'left_eye': {'x': left_eye[0], 'y': left_eye[1]} if isinstance(left_eye, (list, tuple)) and len(left_eye) >= 2 else None,
                            'right_eye': {'x': right_eye[0], 'y': right_eye[1]} if isinstance(right_eye, (list, tuple)) and len(right_eye) >= 2 else None,
                            'eye_alignment_y_diff': abs(float(left_eye[1]) - float(right_eye[1])) if isinstance(left_eye, (list, tuple)) and isinstance(right_eye, (list, tuple)) and len(left_eye) >= 2 and len(right_eye) >= 2 else None,
                            'eye_distance_x': abs(float(left_eye[0]) - float(right_eye[0])) if isinstance(left_eye, (list, tuple)) and isinstance(right_eye, (list, tuple)) and len(left_eye) >= 2 and len(right_eye) >= 2 else None
                        }
                    else:
                        face_debug_info['eyes'] = {
                            'left_eye': None,
                            'right_eye': None,
                            'eye_alignment_y_diff': None,
                            'eye_distance_x': None,
                            'note': 'Eye landmarks not available'
                        }
                    
                    debug_entry['faces'].append(face_debug_info)
                    face_filename = f"dialogue_{dialogue.index:04d}_face_{face_idx:02d}.png"
                    face_path = os.path.join(faces_dir, face_filename)
                    
                    # Save face crop as RGB PNG
                    face_crop_rgb = cv2.cvtColor(face_data['crop'], cv2.COLOR_BGR2RGB)
                    face_image = Image.fromarray(face_crop_rgb)
                    face_image.save(face_path)
                    
                    # Create record
                    record = {
                        'dialogue_index': dialogue.index,
                        'face_index': face_idx,
                        'image_path': face_path,
                        'frame_path': frame_path,
                        'timestamp_seconds': midpoint_seconds,
                        'dialogue_text': dialogue.text,
                        'speaker': dialogue.speaker,
                        'is_llm_confident': dialogue.is_llm_confident,
                        'scene_number': dialogue.scene_number,
                        'detection_confidence': face_data['confidence'],
                        'face_x': face_data['region']['x'],
                        'face_y': face_data['region']['y'],
                        'face_width': face_data['region']['w'],
                        'face_height': face_data['region']['h'],
                        'blur_score': face_data['blur_score'],
                        'series': self.path_handler.get_series(),
                        'season': self.path_handler.get_season(),
                        'episode': self.path_handler.get_episode(),
                        'episode_code': self.path_handler.get_episode_code()
                    }
                    
                    face_records.append(record)
                
                # Add debug entry to list (regardless of whether faces were found)
                debug_data.append(debug_entry)
                
                processed_count += 1
                
                # Periodic progress logging
                if processed_count % progress_interval == 0:
                    progress_pct = (processed_count / len(dialogue_lines)) * 100
                    avg_faces_per_dialogue = total_faces_found / processed_count if processed_count > 0 else 0
                    logger.info(f"📊 Progress: {processed_count}/{len(dialogue_lines)} dialogues ({progress_pct:.1f}%) - {total_faces_found} faces found (avg: {avg_faces_per_dialogue:.1f}/dialogue)")
                
                # Log completion summary for this dialogue
                if len(faces) > 0:
                    logger.debug(f"✅ Dialogue {dialogue.index}: Processed {len(faces)} faces successfully")
                else:
                    logger.debug(f"⚠️ Dialogue {dialogue.index}: No valid faces found")
                
            except Exception as e:
                logger.error(f"❌ Error processing dialogue {dialogue.index}: {e}")
                logger.debug(f"   Dialogue text: '{dialogue.text[:100]}...'")
                logger.debug(f"   Speaker: {dialogue.speaker}, Confident: {dialogue.is_llm_confident}")
                
                # Add debug entry for failed dialogue
                failed_debug_entry = {
                    'dialogue_index': dialogue.index,
                    'timestamp_seconds': (dialogue.start_time + dialogue.end_time) / 2.0,
                    'dialogue_text': dialogue.text,
                    'speaker': dialogue.speaker,
                    'is_llm_confident': dialogue.is_llm_confident,
                    'scene_number': dialogue.scene_number,
                    'frame_filename': None,
                    'total_faces_detected': 0,
                    'faces': [],
                    'error': str(e)
                }
                debug_data.append(failed_debug_entry)
                
                skipped_count += 1
                continue
        
        cap.release()
        
        # Combine existing and new face records
        all_face_records = existing_face_records + face_records
        
        # Create DataFrame and save
        if all_face_records:
            df_faces = pd.DataFrame(all_face_records)
            
            # Remove duplicates based on image_path (keep first occurrence)
            df_faces = df_faces.drop_duplicates(subset=['image_path'], keep='first')
            
            df_faces.to_csv(faces_csv_path, index=False)
            logger.info(f"💾 Saved {len(df_faces)} face records to: {faces_csv_path} (merged {len(existing_face_records)} existing + {len(face_records)} new)")
        else:
            df_faces = pd.DataFrame()
            logger.warning("⚠️ No faces extracted")
        
        logger.info(f"✅ Face extraction complete:")
        logger.info(f"   📊 Dialogues: {processed_count} processed, {skipped_count} skipped ({processed_count + skipped_count} total)")
        logger.info(f"   👥 Faces: {total_faces_found} total found, {len(face_records)} new faces extracted")
        if processed_count > 0:
            logger.info(f"   📈 Average: {total_faces_found / processed_count:.1f} faces per dialogue")
            dialogue_success_rate = (processed_count / (processed_count + skipped_count)) * 100
            logger.info(f"   🎯 Success rate: {dialogue_success_rate:.1f}% dialogues processed successfully")
        
        # Save debug information
        if debug_data:
            try:
                debug_summary = {
                    'extraction_info': {
                        'total_dialogues_processed': processed_count,
                        'total_dialogues_skipped': skipped_count,
                        'total_dialogues_attempted': processed_count + skipped_count,
                        'total_faces_found': total_faces_found,
                        'new_faces_extracted': len(face_records),
                        'avg_faces_per_dialogue': total_faces_found / processed_count if processed_count > 0 else 0,
                        'dialogue_success_rate': (processed_count / (processed_count + skipped_count)) * 100 if (processed_count + skipped_count) > 0 else 0,
                        'detector_used': detector,
                        'min_confidence': min_confidence,
                        'min_face_area_ratio': min_face_area_ratio,
                        'blur_threshold': blur_threshold,
                        'blur_detection_method': config.blur_detection_method,
                        'eye_validation_enabled': enable_eye_validation,
                        'eye_alignment_threshold': eye_alignment_threshold,
                        'eye_distance_threshold': eye_distance_threshold
                    },
                    'dialogue_face_details': debug_data
                }
                
                # Save debug file only if enabled in config
                if config.enable_debug_output:
                    with open(debug_file_path, 'w', encoding='utf-8') as f:
                        json.dump(debug_summary, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"📋 Saved face detection debug information to: {debug_file_path}")
                else:
                    logger.debug(f"📋 Debug file generation disabled (enable_debug_files=false)")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to save debug file: {e}")
        
        return df_faces
    
    def _extract_frame_at_timestamp(
        self, 
        cap: cv2.VideoCapture, 
        timestamp_seconds: float, 
        fps: float
    ) -> Optional[np.ndarray]:
        """Extract frame at specific timestamp."""
        frame_number = int(timestamp_seconds * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            return frame
        else:
            logger.debug(f"⚠️ Could not read frame at {timestamp_seconds:.1f}s (frame {frame_number})")
            return None
    
    def _detect_faces_in_frame(
        self,
        frame: np.ndarray,
        detector: str,
        min_confidence: float,
        min_face_area_ratio: float,
        blur_threshold: float,
        enable_eye_validation: bool = True,
        eye_alignment_threshold: float = 50.0,
        eye_distance_threshold: float = 10.0
    ) -> List[Dict]:
        """Detect and validate faces in frame using DeepFace.extract_faces."""
        try:
            # Convert BGR to RGB for DeepFace
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame_rgb.shape[:2]
            frame_area = frame_height * frame_width
            
            # Extract faces using DeepFace.extract_faces - this returns actual face crops
            try:
                face_objs = DeepFace.extract_faces(
                    img_path=frame_rgb,
                    detector_backend=detector,
                    enforce_detection=False,
                    align=False  # Use align=False as in the working example
                )
                
                if not face_objs:
                    logger.debug("No faces detected by DeepFace.extract_faces")
                    return []
                    
            except Exception as e:
                logger.debug(f"DeepFace.extract_faces failed: {e}")
                return []
            
            valid_faces = []
            
            for i, face_obj in enumerate(face_objs):
                try:
                    # Get face crop and metadata
                    face_crop = face_obj.get('face', None)
                    facial_area = face_obj.get('facial_area', {})
                    confidence = face_obj.get('confidence', 0.0)
                    
                    if face_crop is None:
                        logger.debug(f"Face {i} has no crop data")
                        continue
                    
                    # Validate confidence
                    if confidence < min_confidence:
                        logger.debug(f"Face {i} rejected: low confidence ({confidence:.3f} < {min_confidence})")
                        continue
                    
                    # Convert face crop to proper format and scale (DeepFace returns 0-1 range)
                    if face_crop.max() <= 1.0:
                        face_crop = (face_crop * 255).astype(np.uint8)
                    
                    # Keep face_crop in RGB format for saving (DeepFace.extract_faces returns RGB)
                    face_crop_rgb = face_crop
                    
                    # Validate face area if facial_area is available
                    if facial_area:
                        face_w = facial_area.get('w', 0)
                        face_h = facial_area.get('h', 0)
                        face_area = face_w * face_h
                        area_ratio = face_area / frame_area
                        
                        # Reject faces that are too small
                        if area_ratio < min_face_area_ratio:
                            logger.debug(f"Face {i} rejected: small area ratio ({area_ratio:.4f} < {min_face_area_ratio})")
                            continue
                        
                        # Reject faces that are suspiciously large (likely full frame detection failure)
                        max_face_area_ratio = 0.8  # No face should be more than 80% of the frame
                        if area_ratio > max_face_area_ratio:
                            logger.debug(f"Face {i} rejected: suspiciously large area ratio ({area_ratio:.4f} > {max_face_area_ratio}) - likely full frame")
                            continue
                        
                        # Reject faces with bad aspect ratio (height >= 2 * width)
                        if face_w > 0 and face_h >= 2 * face_w:
                            aspect_ratio = face_h / face_w
                            logger.debug(f"Face {i} rejected: bad aspect ratio ({aspect_ratio:.2f}, height >= 2*width)")
                            continue
                            
                    else:
                        # If no facial_area, use the crop dimensions
                        crop_h, crop_w = face_crop_rgb.shape[:2]
                        crop_area = crop_w * crop_h
                        area_ratio = crop_area / frame_area
                        
                        # Reject faces that are too small
                        if area_ratio < min_face_area_ratio:
                            logger.debug(f"Face {i} rejected: small crop area ratio ({area_ratio:.4f} < {min_face_area_ratio})")
                            continue
                        
                        # Reject faces that are suspiciously large (likely full frame detection failure)
                        max_face_area_ratio = 0.8  # No face should be more than 80% of the frame
                        if area_ratio > max_face_area_ratio:
                            logger.debug(f"Face {i} rejected: suspiciously large crop area ratio ({area_ratio:.4f} > {max_face_area_ratio}) - likely full frame")
                            continue
                        
                        # Reject faces with bad aspect ratio using crop dimensions
                        if crop_w > 0 and crop_h >= 2 * crop_w:
                            aspect_ratio = crop_h / crop_w
                            logger.debug(f"Face {i} rejected: bad crop aspect ratio ({aspect_ratio:.2f}, height >= 2*width)")
                            continue
                    
                    # Validate blur using the face crop with improved method
                    blur_score = self._calculate_comprehensive_blur_score(face_crop_rgb)
                    
                    # Get method-specific threshold from config
                    method = config.blur_detection_method
                    if method == 'gradient':
                        threshold = config.blur_threshold_gradient
                    elif method == 'tenengrad':
                        threshold = config.blur_threshold_tenengrad
                    elif method == 'composite':
                        threshold = config.blur_threshold_composite
                    else:
                        threshold = blur_threshold  # Use original threshold for laplacian
                    
                    logger.debug(f"🎯 Blur check: method={method}, score={blur_score:.2f}, threshold={threshold}")
                    
                    if blur_score < threshold:
                        logger.debug(f"Face {i} rejected: blurry ({blur_score:.1f} < {threshold}, method={method})")
                        continue
                    
                    # Enhanced eye validation if enabled and landmarks available
                    if enable_eye_validation:
                        if not facial_area or not self._is_valid_face_by_landmarks(
                            facial_area, eye_alignment_threshold, eye_distance_threshold
                        ):
                            logger.debug(f"Face {i} rejected: failed eye validation")
                            continue
                    
                    # Create region dict for compatibility
                    region = facial_area if facial_area else {
                        'x': 0, 'y': 0, 
                        'w': face_crop_rgb.shape[1], 
                        'h': face_crop_rgb.shape[0]
                    }
                    
                    valid_faces.append({
                        'crop': face_crop_rgb,  # Keep RGB format for saving
                        'region': region,
                        'confidence': confidence,
                        'blur_score': blur_score
                    })
                    
                except Exception as e:
                    logger.debug(f"Error validating face {i}: {e}")
                    continue
            
            return valid_faces
            
        except Exception as e:
            logger.debug(f"Error detecting faces: {e}")
            return []
    
    def _calculate_blur_score(self, image: np.ndarray) -> float:
        """Calculate Laplacian variance for blur detection (legacy method)."""
        try:
            # Handle both RGB and BGR images
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            return 0.0
    
    def _calculate_gradient_blur_score(self, image: np.ndarray) -> float:
        """
        Gradient magnitude method - more robust than Laplacian across skin tones.
        Higher scores indicate sharper images.
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Calculate gradients using Sobel operators
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Return mean gradient magnitude
            return float(np.mean(gradient_magnitude))
            
        except Exception as e:
            logger.warning(f"Gradient blur calculation failed: {e}")
            return 0.0
    
    def _calculate_tenengrad_score(self, image: np.ndarray) -> float:
        """
        Tenengrad focus measure - excellent for faces, less biased by skin tone.
        Higher scores indicate sharper images.
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Apply Sobel operators
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate Tenengrad (sum of squared gradients)
            tenengrad = sobel_x**2 + sobel_y**2
            
            # Return mean of squared gradients
            return float(np.mean(tenengrad))
            
        except Exception as e:
            logger.warning(f"Tenengrad calculation failed: {e}")
            return 0.0
    
    def _calculate_comprehensive_blur_score(self, image: np.ndarray) -> float:
        """
        Calculate comprehensive blur score using the configured method.
        Supports multiple methods with reduced skin tone bias.
        """
        try:
            method = config.blur_detection_method
            
            if method == 'laplacian':
                return self._calculate_blur_score(image)
            elif method == 'gradient':
                return self._calculate_gradient_blur_score(image)
            elif method == 'tenengrad':
                return self._calculate_tenengrad_score(image)
            elif method == 'composite':
                # Weighted combination of methods for best results
                gradient_score = self._calculate_gradient_blur_score(image)
                tenengrad_score = self._calculate_tenengrad_score(image)
                laplacian_score = self._calculate_blur_score(image)
                
                # Normalize scores to similar ranges and combine
                normalized_gradient = gradient_score / 20.0  # Typical range: 0-40
                normalized_tenengrad = tenengrad_score / 1000.0  # Typical range: 0-2000
                normalized_laplacian = laplacian_score / 10.0  # Typical range: 0-20
                
                # Weighted average (favor gradient and tenengrad)
                composite = (normalized_gradient * 0.5 + 
                           normalized_tenengrad * 0.3 + 
                           normalized_laplacian * 0.2)
                
                return float(composite * 20.0)  # Scale back to usable range
            else:
                # Default to gradient method
                return self._calculate_gradient_blur_score(image)
                
        except Exception as e:
            logger.warning(f"Comprehensive blur calculation failed: {e}")
            # Fallback to simple Laplacian
            return self._calculate_blur_score(image)
    
    def _is_valid_face_by_landmarks(
        self, 
        facial_area: Dict, 
        eye_alignment_threshold: float = 50.0, 
        eye_distance_threshold: float = 10.0
    ) -> bool:
        """
        Enhanced face validation using facial landmarks from DeepFace.extract_faces.
        Validates face quality based on eye position and alignment.
        
        Args:
            facial_area: Facial area dict from DeepFace.extract_faces result
            eye_alignment_threshold: Maximum Y-axis difference between eyes
            eye_distance_threshold: Minimum X-axis distance between eyes
            
        Returns:
            True if face passes landmark validation, False otherwise
        """
        try:
            # Get eye landmarks from facial_area
            left_eye_tuple = facial_area.get("left_eye", None)
            right_eye_tuple = facial_area.get("right_eye", None)

            # STRICT: Reject faces without eye landmarks - they are not reliable
            if not (isinstance(left_eye_tuple, (list, tuple)) and len(left_eye_tuple) >= 2 and
                    isinstance(right_eye_tuple, (list, tuple)) and len(right_eye_tuple) >= 2):
                logger.debug("Face rejected: eye landmarks missing or invalid - not reliable face detection")
                return False  # Reject if landmarks unavailable
            
            # Check for negative coordinates
            if (left_eye_tuple[0] < 0 or left_eye_tuple[1] < 0 or 
                right_eye_tuple[0] < 0 or right_eye_tuple[1] < 0):
                logger.debug("Face rejected: negative eye coordinates")
                return False

            # Calculate vertical alignment (Y-axis difference)
            y_diff = abs(float(left_eye_tuple[1]) - float(right_eye_tuple[1]))
            if y_diff >= eye_alignment_threshold:
                logger.debug(f"Face rejected: eye alignment diff {y_diff:.1f} >= {eye_alignment_threshold}")
                return False

            # Calculate horizontal distance (X-axis difference)
            x_diff = abs(float(left_eye_tuple[0]) - float(right_eye_tuple[0]))
            if x_diff < eye_distance_threshold:
                logger.debug(f"Face rejected: eye horizontal distance {x_diff:.1f} < {eye_distance_threshold}")
                return False

            return True

        except (ValueError, TypeError) as e:
            logger.debug(f"Error in eye coordinate processing: {e}. Rejecting face.")
            return False  # Reject faces that cause processing errors
