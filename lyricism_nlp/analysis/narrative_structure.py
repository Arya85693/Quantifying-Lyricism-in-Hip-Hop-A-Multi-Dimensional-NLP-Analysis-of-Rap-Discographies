"""
Narrative Structure & Coherence Analysis Module.

Measures storytelling ability:
- First-person continuity (consistent narrative voice)
- Temporal progression (time references, sequence)
- Recurring entities or motifs
- Thematic cohesion across verses
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter, defaultdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import PROCESSED_DATA_DIR

# First-person indicators
FIRST_PERSON_WORDS = {'i', 'me', 'my', 'myself', 'mine', 'we', 'us', 'our', 'ours'}

# Temporal indicators
TEMPORAL_WORDS = {
    'now', 'then', 'when', 'before', 'after', 'later', 'soon', 'early', 'late',
    'today', 'yesterday', 'tomorrow', 'morning', 'afternoon', 'evening', 'night',
    'day', 'week', 'month', 'year', 'time', 'past', 'present', 'future',
    'always', 'never', 'sometimes', 'often', 'once', 'twice', 'again'
}

# Narrative action verbs
NARRATIVE_VERBS = {
    'said', 'told', 'went', 'came', 'saw', 'heard', 'felt', 'thought', 'knew',
    'did', 'made', 'took', 'gave', 'got', 'put', 'set', 'left', 'right',
    'started', 'stopped', 'began', 'ended', 'continued', 'happened', 'occurred'
}


def calculate_first_person_continuity(lines: List[List[str]]) -> float:
    """
    Calculate consistency of first-person narrative voice.
    
    Args:
        lines: List of lines
        
    Returns:
        First-person continuity score (0-1)
    """
    if not lines:
        return 0.0
    
    first_person_counts = []
    for line in lines:
        line_tokens = set(token.lower() for token in line)
        first_person_in_line = len(line_tokens.intersection(FIRST_PERSON_WORDS))
        first_person_counts.append(first_person_in_line)
    
    if not first_person_counts:
        return 0.0
    
    # Consistency: low variance in first-person usage
    mean_fp = sum(first_person_counts) / len(first_person_counts)
    if mean_fp == 0:
        return 0.0
    
    variance = sum((count - mean_fp) ** 2 for count in first_person_counts) / len(first_person_counts)
    std_dev = variance ** 0.5
    
    # Lower variance = higher consistency
    # Normalize: consistency score = 1 / (1 + normalized_std)
    normalized_std = std_dev / (mean_fp + 1)  # Add 1 to avoid division by zero
    consistency = 1.0 / (1.0 + normalized_std)
    
    # Also reward presence of first-person (can't be consistent if never used)
    presence_score = min(mean_fp / 2.0, 1.0)  # Normalize
    
    return 0.5 * consistency + 0.5 * presence_score


def calculate_temporal_progression(tokens: List[str], lines: List[List[str]]) -> float:
    """
    Calculate temporal progression indicators.
    
    Args:
        tokens: All tokens
        lines: List of lines
        
    Returns:
        Temporal progression score (0-1)
    """
    if not tokens:
        return 0.0
    
    token_set = set(token.lower() for token in tokens)
    
    # Count temporal words
    temporal_count = len(token_set.intersection(TEMPORAL_WORDS))
    temporal_density = temporal_count / len(tokens) if tokens else 0.0
    
    # Check for narrative verbs (indicating events/actions)
    narrative_count = len(token_set.intersection(NARRATIVE_VERBS))
    narrative_density = narrative_count / len(tokens) if tokens else 0.0
    
    # Combined score
    score = 0.6 * min(temporal_density * 10, 1.0) + 0.4 * min(narrative_density * 5, 1.0)
    
    return score


def calculate_recurring_motifs(tokens: List[str], min_frequency: int = 2) -> float:
    """
    Calculate presence of recurring entities or motifs.
    
    Args:
        tokens: All tokens
        min_frequency: Minimum frequency to count as motif
        
    Returns:
        Recurring motifs score (0-1)
    """
    if not tokens:
        return 0.0
    
    # Count word frequencies
    word_counts = Counter(token.lower() for token in tokens)
    
    # Find recurring words (excluding very common words)
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    recurring_words = {
        word: count for word, count in word_counts.items()
        if count >= min_frequency and word not in common_words
    }
    
    if not recurring_words:
        return 0.0
    
    # Score based on number and frequency of motifs
    num_motifs = len(recurring_words)
    avg_frequency = sum(recurring_words.values()) / num_motifs if num_motifs > 0 else 0.0
    
    # Normalize
    motif_count_score = min(num_motifs / 10.0, 1.0)
    frequency_score = min(avg_frequency / 5.0, 1.0)
    
    return 0.5 * motif_count_score + 0.5 * frequency_score


def calculate_thematic_cohesion(lines: List[List[str]]) -> float:
    """
    Calculate thematic cohesion across verses/lines.
    
    Args:
        lines: List of lines
        
    Returns:
        Thematic cohesion score (0-1)
    """
    if len(lines) < 2:
        return 0.0
    
    # Extract unique words per line
    line_vocabularies = [set(token.lower() for token in line) for line in lines]
    
    # Calculate vocabulary overlap between adjacent lines
    overlaps = []
    for i in range(len(line_vocabularies) - 1):
        vocab1 = line_vocabularies[i]
        vocab2 = line_vocabularies[i + 1]
        
        if not vocab1 or not vocab2:
            continue
        
        # Jaccard similarity
        intersection = len(vocab1.intersection(vocab2))
        union = len(vocab1.union(vocab2))
        
        if union > 0:
            similarity = intersection / union
            overlaps.append(similarity)
    
    if not overlaps:
        return 0.0
    
    # Average overlap indicates cohesion
    avg_overlap = sum(overlaps) / len(overlaps)
    
    # Also consider consistency of overlap
    if len(overlaps) > 1:
        variance = sum((overlap - avg_overlap) ** 2 for overlap in overlaps) / len(overlaps)
        consistency = 1.0 / (1.0 + variance * 10)  # Normalize
    else:
        consistency = 1.0
    
    return 0.7 * avg_overlap + 0.3 * consistency


def analyze_narrative_structure(song_data: Dict) -> float:
    """
    Compute narrative structure score for a song.
    
    Args:
        song_data: Processed song data dictionary
        
    Returns:
        Narrative structure score (0-1)
    """
    tokens = song_data.get("lyrics_tokenized", [])
    lines = song_data.get("lyrics_lines", [])
    
    if not tokens or not lines:
        return 0.0
    
    # Calculate component metrics
    first_person = calculate_first_person_continuity(lines)
    temporal = calculate_temporal_progression(tokens, lines)
    motifs = calculate_recurring_motifs(tokens)
    cohesion = calculate_thematic_cohesion(lines)
    
    # Weighted combination
    score = (
        0.25 * first_person +
        0.25 * temporal +
        0.25 * motifs +
        0.25 * cohesion
    )
    
    return min(max(score, 0.0), 1.0)


def process_all_songs() -> Dict[tuple, float]:
    """
    Process all songs and return narrative structure scores.
    
    Returns:
        Dictionary mapping (artist, song_title) to score
    """
    processed_file = Path(PROCESSED_DATA_DIR) / "all_processed_lyrics.jsonl"
    
    if not processed_file.exists():
        print(f"Processed lyrics file not found: {processed_file}")
        return {}
    
    scores = {}
    
    with open(processed_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    song_data = json.loads(line)
                    score = analyze_narrative_structure(song_data)
                    
                    key = (song_data.get("artist"), song_data.get("song_title"))
                    scores[key] = score
                except Exception as e:
                    print(f"Error processing song: {e}")
                    continue
    
    return scores


if __name__ == "__main__":
    scores = process_all_songs()
    print(f"Computed narrative structure scores for {len(scores)} songs")
    for (artist, song), score in list(scores.items())[:5]:
        print(f"{artist} - {song}: {score:.4f}")
