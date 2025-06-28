# character_entities_utils.py

from typing import List, Set
import re

# Comprehensive list of titles used for character recognition
TITLES = [
    "Dr", "Dr.", "Mr", "Mr.", "Ms", "Ms.", "Miss", "Miss.", "Mrs", "Mrs.",
    "Mme", "Mme.", "Prof", "Prof.", "Doc", "Doctor", "Sir", "Lady", "Dame",
    "Duke", "Duchess", "Count", "Countess", "Baron", "Baroness", "King",
    "Queen", "Prince", "Princess", "Earl", "Don", "Sister", "Brother",
    "Colonel", "Major", "Captain", "Lieutenant", "Sergeant", "Private",
    "Doctorate", "PhD", "MD", "MA", "BA", "BS", "BSc", "Bachelor", "Master",
    "Masters", "M.A.", "M.B.A.", "M.D.", "M.S.", "M.Sc.", "Chief", "Judge",
    "Father", "Mother", "Pastor", "Reverend", "Rev", "Rev.", "Senator",
    "Representative", "President", "Vice President", "Governor", "Mayor",
    "General", "Admiral", "Commander", "Sheriff", "Detective", "Officer",
    "Agent", "Inspector", "Constable", "Nurse", "Paramedic", "EMT",
    "Intern", "Resident", "Attending", "Fellow", "Chaplain", "Coach"
]

def normalize_entity_name(name: str) -> str:
    """
    Normalize entity name to lowercase with underscores, removing titles.
    
    Args:
        name: The entity name to normalize
        
    Returns:
        Normalized entity name
    """
    if not name:
        return ""
    
    # Remove common titles from the beginning
    normalized = name.strip()
    
    # Split by spaces and filter out titles
    words = normalized.split()
    filtered_words = []
    
    for word in words:
        # Remove punctuation from word for comparison
        clean_word = word.rstrip('.,!?;:')
        if clean_word not in TITLES:
            filtered_words.append(word)
    
    # Join filtered words and convert to lowercase with underscores
    if filtered_words:
        result = "_".join(filtered_words).lower()
        # Clean up any extra punctuation
        result = re.sub(r'[^\w_]', '', result)
        # Remove multiple underscores
        result = re.sub(r'_+', '_', result)
        # Remove leading/trailing underscores
        result = result.strip('_')
        return result
    
    return name.lower().replace(' ', '_')

def extract_gender_indicators(appellation: str) -> str:
    """
    Extract gender indicators from appellations like Mr., Mrs., etc.
    
    Args:
        appellation: The appellation to analyze
        
    Returns:
        Gender indicator ('male', 'female', or 'unknown')
    """
    appellation_lower = appellation.lower().strip()
    
    male_indicators = ['mr', 'mr.', 'sir', 'king', 'prince', 'duke', 'baron', 'earl', 'father', 'brother']
    female_indicators = ['mrs', 'mrs.', 'ms', 'ms.', 'miss', 'miss.', 'lady', 'queen', 'princess', 'duchess', 'baroness', 'mother', 'sister']
    
    for indicator in male_indicators:
        if indicator in appellation_lower:
            return 'male'
    
    for indicator in female_indicators:
        if indicator in appellation_lower:
            return 'female'
    
    return 'unknown'

def has_conflicting_gender_titles(appellations1: List[str], appellations2: List[str]) -> bool:
    """
    Check if two sets of appellations have conflicting gender-specific titles.
    
    Args:
        appellations1: First set of appellations
        appellations2: Second set of appellations
        
    Returns:
        True if there are conflicting gender titles, False otherwise
    """
    gender1_indicators = set()
    gender2_indicators = set()
    
    for app in appellations1:
        gender = extract_gender_indicators(app)
        if gender != 'unknown':
            gender1_indicators.add(gender)
    
    for app in appellations2:
        gender = extract_gender_indicators(app)
        if gender != 'unknown':
            gender2_indicators.add(gender)
    
    # If both have gender indicators and they don't overlap, there's a conflict
    if gender1_indicators and gender2_indicators and not gender1_indicators.intersection(gender2_indicators):
        return True
    
    return False

def extract_surname_from_appellation(appellation: str) -> str:
    """
    Extract surname from an appellation, handling titles.
    
    Args:
        appellation: The appellation to process
        
    Returns:
        The extracted surname or empty string
    """
    if not appellation:
        return ""
    
    words = appellation.strip().split()
    
    # Filter out titles
    filtered_words = []
    for word in words:
        clean_word = word.rstrip('.,!?;:')
        if clean_word not in TITLES:
            filtered_words.append(word)
    
    # Return the last word as surname if we have multiple words
    if len(filtered_words) > 1:
        return filtered_words[-1].rstrip('.,!?;:')
    elif len(filtered_words) == 1:
        return filtered_words[0].rstrip('.,!?;:')
    
    return ""
