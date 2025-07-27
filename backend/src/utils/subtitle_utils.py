
import re
from typing import List
from dataclasses import dataclass

import logging
logger = logging.getLogger(__name__)

@dataclass
class SubtitleEntry:
    """Represents a single subtitle entry."""
    index: int
    start_time: str
    end_time: str
    text: str
    start_seconds: float
    end_seconds: float

def parse_srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT time format (HH:MM:SS,mmm) to seconds."""
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def parse_srt_file(srt_path: str) -> List[SubtitleEntry]:
    """Parse an SRT subtitle file and return list of subtitle entries."""
    logger.info(f"Parsing SRT file: {srt_path}")
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(srt_path, 'r', encoding='latin-1') as f:
            content = f.read()
    
    # Split into subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    subtitles = []
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
            
        try:
            # Parse index
            index = int(lines[0])
            
            # Parse time range
            time_line = lines[1]
            time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', time_line)
            if not time_match:
                continue
                
            start_time = time_match.group(1)
            end_time = time_match.group(2)
            
            # Parse subtitle text (may span multiple lines)
            text = '\n'.join(lines[2:]).strip()
            
            # Convert times to seconds
            start_seconds = parse_srt_time_to_seconds(start_time)
            end_seconds = parse_srt_time_to_seconds(end_time)
            
            subtitles.append(SubtitleEntry(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text,
                start_seconds=start_seconds,
                end_seconds=end_seconds
            ))
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse subtitle block: {block[:50]}... Error: {e}")
            continue
    
    logger.info(f"Parsed {len(subtitles)} subtitle entries")
    return subtitles

