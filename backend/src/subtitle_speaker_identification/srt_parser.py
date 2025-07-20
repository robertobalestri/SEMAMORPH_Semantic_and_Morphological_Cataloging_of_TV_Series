"""
SRT Parser for SEMAMORPH project.
Parses SRT subtitle files and converts them to DialogueLine objects.
"""
import re
import json
import os
from typing import List
from ..narrative_storage_management.narrative_models import DialogueLine
from ..utils.logger_utils import setup_logging

logger = setup_logging(__name__)

class SRTParser:
    """Parser for SRT subtitle files."""
    
    SRT_PATTERN = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?:\n\n|\Z)'

    def parse(self, file_path: str, output_path: str = None) -> List[DialogueLine]:
        """
        Parse SRT file and optionally save as JSON.
        
        Args:
            file_path: Path to SRT file
            output_path: Optional path to save JSON output
            
        Returns:
            List of DialogueLine objects
        """
        logger.info(f"📄 Parsing SRT file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"❌ SRT file not found: {file_path}")
            raise FileNotFoundError(f"SRT file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file is empty
            if not content.strip():
                logger.warning(f"⚠️ SRT file is empty: {file_path}")
                return []
                
            matches = re.findall(self.SRT_PATTERN, content)
            
            # Check if no subtitles were found
            if not matches:
                logger.warning(f"⚠️ No valid subtitle entries found in SRT file: {file_path}")
                return []
                
            dialogue_lines = self._process_matches(matches)
            
            # Replace newlines with spaces in text
            for line in dialogue_lines:
                line.text = line.text.replace('\n', ' ')
            
            logger.info(f"✅ Parsed {len(dialogue_lines)} dialogue lines")
            
            # Save as JSON if output path provided
            if output_path is not None:
                dialogue_data = [line.to_dict() for line in dialogue_lines]
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(dialogue_data, f, indent=4, ensure_ascii=False)
                logger.info(f"💾 Saved dialogue data to: {output_path}")

            return dialogue_lines
            
        except Exception as e:
            logger.error(f"❌ Error parsing SRT file {file_path}: {e}")
            raise
            
    def _process_matches(self, matches) -> List[DialogueLine]:
        """Process regex matches into DialogueLine objects."""
        return [
            DialogueLine(
                int(num), 
                self._convert_time(start), 
                self._convert_time(end),
                self._clean_text(text)
            ) 
            for num, start, end, text in matches
        ]

    def _convert_time(self, time_str: str) -> float:
        """Convert SRT time format to seconds."""
        h, m, s = time_str.replace(',', '.').split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    def _clean_text(self, text: str) -> str:
        """Clean subtitle text."""
        return re.sub(r' +', ' ', text.strip())

    @staticmethod
    def load_dialogue_json(json_path: str) -> List[DialogueLine]:
        """Load dialogue lines from JSON file."""
        logger.info(f"📂 Loading dialogue from JSON: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            dialogue_lines = [DialogueLine.from_dict(item) for item in data]
            logger.info(f"✅ Loaded {len(dialogue_lines)} dialogue lines from JSON")
            return dialogue_lines
            
        except Exception as e:
            logger.error(f"❌ Error loading dialogue JSON {json_path}: {e}")
            raise
