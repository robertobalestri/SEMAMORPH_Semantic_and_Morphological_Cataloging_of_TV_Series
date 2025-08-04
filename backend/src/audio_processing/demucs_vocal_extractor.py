"""
Demucs vocal extraction module for SEMAMORPH.
Extracts vocals from audio files to improve speaker diarization quality.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch

import sys
import os

# Add the backend/src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger_utils import setup_logging
from config import Config

logger = setup_logging(__name__)

class DemucsVocalExtractor:
    """
    Extracts vocals from audio files using Demucs for improved speaker diarization.
    
    Demucs separates audio into 4 stems: drums, bass, other, vocals
    We extract only the vocals for cleaner speaker identification.
    """
    
    def __init__(self, config):
        """Initialize the Demucs vocal extractor."""
        self.config = config
        self.demucs_model = getattr(config, 'demucs_model', 'htdemucs')
        self.demucs_device = getattr(config, 'demucs_device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.demucs_segment_length = getattr(config, 'demucs_segment_length', 7)  # Max for htdemucs is 7.8
        self.demucs_overlap = getattr(config, 'demucs_overlap', 0.25)
        self.demucs_shifts = getattr(config, 'demucs_shifts', 1)
        
        # Check if demucs is available
        self._check_demucs_availability()
    
    def _check_demucs_availability(self) -> bool:
        """Check if Demucs is installed and available."""
        try:
            result = subprocess.run(['demucs', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ Demucs is available")
                return True
            else:
                logger.warning("⚠️ Demucs command failed, trying python module")
                return self._check_demucs_python()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("⚠️ Demucs command not found, trying python module")
            return self._check_demucs_python()
    
    def _check_demucs_python(self) -> bool:
        """Check if Demucs is available as a Python module."""
        try:
            import demucs
            logger.info("✅ Demucs Python module is available")
            return True
        except ImportError:
            logger.error("❌ Demucs is not installed. Please install it with: pip install demucs")
            return False
    
    def extract_vocals(self, audio_path: str, output_dir: Optional[str] = None, path_handler=None) -> Optional[str]:
        """
        Extract vocals from an audio file using Demucs.
        
        Args:
            audio_path: Path to the input audio file
            output_dir: Directory to save extracted vocals (optional)
            path_handler: Optional PathHandler instance for consistent path generation
            
        Returns:
            Path to the extracted vocals file, or None if extraction failed
        """
        if not os.path.exists(audio_path):
            logger.error(f"❌ Audio file not found: {audio_path}")
            return None
        
        # Determine output directory and vocals path
        audio_name = Path(audio_path).stem  # Always get audio_name
        
        if path_handler:
            vocals_path = path_handler.get_vocals_file_path(audio_path)
            # For Demucs, we need to use the parent directory of vocals_path as output_dir
            # because Demucs will create MODEL_NAME/AUDIO_NAME/ structure inside it
            output_dir = os.path.dirname(vocals_path)
        else:
            # Fallback to simple path generation
            if output_dir is None:
                output_dir = os.path.dirname(audio_path)
            vocals_dir = os.path.join(output_dir, "vocals")
            vocals_path = os.path.join(vocals_dir, f"{audio_name}_vocals.wav")
            output_dir = vocals_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if vocals already exist in the expected location
        if os.path.exists(vocals_path):
            logger.info(f"📂 Vocals already exist: {vocals_path}")
            return vocals_path
        
        # Check if vocals exist in Demucs output structure
        demucs_vocals_file = self._find_vocals_file(audio_name, output_dir)
        if demucs_vocals_file and os.path.exists(demucs_vocals_file):
            logger.info(f"📂 Found existing vocals in Demucs output: {demucs_vocals_file}")
            # Copy to our desired location
            import shutil
            shutil.copy2(demucs_vocals_file, vocals_path)
            logger.info(f"✅ Vocals copied to: {vocals_path}")
            
            # Clean up Demucs output directory
            demucs_output_dir = os.path.join(output_dir, self.demucs_model)
            if os.path.exists(demucs_output_dir):
                shutil.rmtree(demucs_output_dir)
                logger.info(f"🗑️ Cleaned up Demucs output directory: {demucs_output_dir}")
            
            return vocals_path
        
        logger.info(f"🎵 Extracting vocals from: {audio_path}")
        logger.info(f"📁 Output will be saved to: {vocals_path}")
        
        try:
            # Build Demucs command
            cmd = self._build_demucs_command(audio_path, output_dir)
            
            logger.info(f"🔧 Running Demucs command: {' '.join(cmd)}")
            
            # Run Demucs
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Demucs vocal extraction completed successfully")
                
                # Find the vocals file in the separated directory
                vocals_file = self._find_vocals_file(audio_name, output_dir)
                
                if vocals_file and os.path.exists(vocals_file):
                    # Copy to our desired location
                    import shutil
                    shutil.copy2(vocals_file, vocals_path)
                    logger.info(f"✅ Vocals extracted to: {vocals_path}")
                    
                    # Clean up Demucs output directory
                    demucs_output_dir = os.path.join(output_dir, self.demucs_model)
                    if os.path.exists(demucs_output_dir):
                        shutil.rmtree(demucs_output_dir)
                        logger.info(f"🗑️ Cleaned up Demucs output directory: {demucs_output_dir}")
                    
                    return vocals_path
                else:
                    logger.error("❌ Vocals file not found in Demucs output")
                    return None
            else:
                logger.error(f"❌ Demucs failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Demucs extraction timed out (1 hour)")
            return None
        except Exception as e:
            logger.error(f"❌ Error during Demucs extraction: {e}")
            return None
    
    def _build_demucs_command(self, audio_path: str, output_dir: str) -> list:
        """Build the Demucs command with appropriate parameters."""
        cmd = [
            'demucs',
            '--two-stems=vocals',  # Only extract vocals
            '-n', self.demucs_model,  # Model name
            '-d', self.demucs_device,  # Device (cuda/cpu)
            '--segment', str(self.demucs_segment_length),  # Segment length
            '--overlap', str(self.demucs_overlap),  # Overlap percentage
            '--shifts', str(self.demucs_shifts),  # Number of shifts
            '--out', output_dir,  # Output directory
            audio_path  # Input file
        ]
        
        # Add MP3 output if configured
        if getattr(self.config, 'demucs_output_mp3', False):
            cmd.extend(['--mp3', '--mp3-bitrate', '320'])
        
        return cmd
    
    def _find_vocals_file(self, audio_name: str, output_dir: str) -> Optional[str]:
        """Find the vocals file in the Demucs output directory."""
        # Demucs creates a directory structure: MODEL_NAME/AUDIO_NAME/
        model_dir = os.path.join(output_dir, self.demucs_model, audio_name)
        
        if not os.path.exists(model_dir):
            logger.warning(f"⚠️ Demucs output directory not found: {model_dir}")
            return None
        
        # Look for vocals file
        vocals_file = os.path.join(model_dir, 'vocals.wav')
        if os.path.exists(vocals_file):
            return vocals_file
        
        # Try MP3 if configured
        vocals_file_mp3 = os.path.join(model_dir, 'vocals.mp3')
        if os.path.exists(vocals_file_mp3):
            return vocals_file_mp3
        
        logger.warning(f"⚠️ Vocals file not found in: {model_dir}")
        return None
    
    def get_vocals_path(self, audio_path: str, path_handler=None) -> str:
        """
        Get the expected path for vocals file.
        
        Args:
            audio_path: Path to the original audio file
            path_handler: Optional PathHandler instance for consistent path generation
            
        Returns:
            Path to the vocals file
        """
        if path_handler:
            return path_handler.get_vocals_file_path(audio_path)
        else:
            # Fallback to simple path generation
            audio_name = Path(audio_path).stem
            output_dir = os.path.dirname(audio_path)
            vocals_dir = os.path.join(output_dir, "vocals")
            return os.path.join(vocals_dir, f"{audio_name}_vocals.wav")

def extract_vocals_for_diarization(audio_path: str, config, path_handler=None) -> Optional[str]:
    """
    Convenience function to extract vocals for diarization.
    
    Args:
        audio_path: Path to the input audio file
        config: Configuration object
        path_handler: Optional PathHandler instance for consistent path generation
        
    Returns:
        Path to the extracted vocals file, or None if extraction failed
    """
    extractor = DemucsVocalExtractor(config)
    return extractor.extract_vocals(audio_path, path_handler=path_handler) 