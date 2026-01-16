"""
Lexical & Linguistic Complexity Analysis Module.

Measures surface-level language use:
- Vocabulary richness (type-token ratio, unique words)
- Word rarity (using frequency-based metrics)
- Syllabic density (average syllables per word)
- Syntactic variation (sentence/line length variation)
"""

import json
import re
from pathlib import Path
from typing import Dict, List
from collections import Counter
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import PROCESSED_DATA_DIR

# Common English words for rarity calculation (top 1000 most common)
COMMON_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
    'is', 'was', 'are', 'were', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'will', 'would', 'shall', 'should', 'may', 'might',
    'must', 'can', 'could', 'ought', 'need', 'dare', 'used'
}


def count_syllables(word: str) -> int:
    """
    Estimate syllable count for a word.
    Simple heuristic-based approach.
    
    Args:
        word: Word to count syllables for
        
    Returns:
        Estimated syllable count
    """
    word = word.lower().strip()
    if not word:
        return 0
    
    # Remove trailing 'e' for syllable counting
    word = re.sub(r'e$', '', word)
    
    # Count vowel groups
    vowels = re.findall(r'[aeiouy]+', word)
    syllable_count = len(vowels)
    
    # Minimum 1 syllable
    if syllable_count == 0:
        syllable_count = 1
    
    return syllable_count


def calculate_type_token_ratio(tokens: List[str]) -> float:
    """
    Calculate type-token ratio (vocabulary richness).
    
    Args:
        tokens: List of tokens
        
    Returns:
        Type-token ratio (0-1)
    """
    if not tokens:
        return 0.0
    
    unique_types = len(set(tokens))
    total_tokens = len(tokens)
    
    return unique_types / total_tokens if total_tokens > 0 else 0.0


def calculate_word_rarity_score(tokens: List[str]) -> float:
    """
    Calculate word rarity score based on common word avoidance.
    
    Args:
        tokens: List of tokens
        
    Returns:
        Rarity score (0-1, higher = rarer words)
    """
    if not tokens:
        return 0.0
    
    rare_count = sum(1 for token in tokens if token.lower() not in COMMON_WORDS)
    return rare_count / len(tokens) if tokens else 0.0


def calculate_syllabic_density(tokens: List[str]) -> float:
    """
    Calculate average syllables per word.
    
    Args:
        tokens: List of tokens
        
    Returns:
        Average syllables per word
    """
    if not tokens:
        return 0.0
    
    total_syllables = sum(count_syllables(token) for token in tokens)
    return total_syllables / len(tokens) if tokens else 0.0


def calculate_syntactic_variation(lines: List[List[str]]) -> float:
    """
    Calculate syntactic variation based on line length variation.
    
    Args:
        lines: List of lines, each line is a list of tokens
        
    Returns:
        Coefficient of variation of line lengths (0-1 normalized)
    """
    if not lines or len(lines) < 2:
        return 0.0
    
    line_lengths = [len(line) for line in lines if line]
    if not line_lengths:
        return 0.0
    
    mean_length = sum(line_lengths) / len(line_lengths)
    if mean_length == 0:
        return 0.0
    
    variance = sum((length - mean_length) ** 2 for length in line_lengths) / len(line_lengths)
    std_dev = variance ** 0.5
    
    # Coefficient of variation, normalized to 0-1
    cv = std_dev / mean_length if mean_length > 0 else 0.0
    return min(cv, 1.0)  # Cap at 1.0


def analyze_lexical_complexity(song_data: Dict) -> float:
    """
    Compute lexical complexity score for a song.
    
    Args:
        song_data: Processed song data dictionary
        
    Returns:
        Lexical complexity score (0-1)
    """
    tokens = song_data.get("lyrics_tokenized", [])
    lines = song_data.get("lyrics_lines", [])
    
    if not tokens:
        return 0.0
    
    # Calculate component metrics
    ttr = calculate_type_token_ratio(tokens)
    rarity = calculate_word_rarity_score(tokens)
    syllabic_density = calculate_syllabic_density(tokens)
    syntactic_var = calculate_syntactic_variation(lines)
    
    # Normalize syllabic density (typical range 1-3, normalize to 0-1)
    normalized_syllabic = min(syllabic_density / 3.0, 1.0)
    
    # Weighted combination
    score = (
        0.30 * ttr +
        0.25 * rarity +
        0.25 * normalized_syllabic +
        0.20 * syntactic_var
    )
    
    return min(max(score, 0.0), 1.0)


def process_all_songs() -> Dict[str, float]:
    """
    Process all songs and return lexical complexity scores.
    
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
                    score = analyze_lexical_complexity(song_data)
                    
                    key = (song_data.get("artist"), song_data.get("song_title"))
                    scores[key] = score
                except Exception as e:
                    print(f"Error processing song: {e}")
                    continue
    
    return scores


if __name__ == "__main__":
    scores = process_all_songs()
    print(f"Computed lexical complexity scores for {len(scores)} songs")
    for (artist, song), score in list(scores.items())[:5]:
        print(f"{artist} - {song}: {score:.4f}")
