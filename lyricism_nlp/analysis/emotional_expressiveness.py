"""
Emotional Expressiveness & Intensity Analysis Module.

Measures emotional impact:
- Emotional range (diversity of emotions expressed)
- Intensity (strength of emotional language)
- Polarity shifts (variation in positive/negative sentiment)
Neutral to positivity/negativity; focuses on expressiveness.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import PROCESSED_DATA_DIR

# Emotional word lexicons (simplified)
POSITIVE_EMOTIONS = {
    'love', 'happy', 'joy', 'glad', 'excited', 'proud', 'grateful', 'thankful',
    'blessed', 'lucky', 'success', 'win', 'victory', 'triumph', 'celebration',
    'smile', 'laugh', 'cheer', 'hope', 'dream', 'wish', 'desire', 'want',
    'beautiful', 'amazing', 'wonderful', 'great', 'excellent', 'perfect', 'best',
    'strong', 'powerful', 'mighty', 'brave', 'courage', 'confident', 'sure'
}

NEGATIVE_EMOTIONS = {
    'hate', 'sad', 'angry', 'mad', 'furious', 'rage', 'pain', 'hurt', 'suffer',
    'cry', 'tear', 'fear', 'afraid', 'scared', 'worried', 'anxious', 'stress',
    'depressed', 'lonely', 'alone', 'empty', 'void', 'lost', 'confused', 'doubt',
    'regret', 'guilt', 'shame', 'disappointed', 'frustrated', 'upset', 'annoyed',
    'weak', 'broken', 'defeated', 'failure', 'lose', 'loss', 'death', 'die', 'kill'
}

INTENSITY_MODIFIERS = {
    'very', 'extremely', 'incredibly', 'absolutely', 'completely', 'totally',
    'really', 'truly', 'deeply', 'profoundly', 'intensely', 'strongly',
    'barely', 'hardly', 'slightly', 'somewhat', 'quite', 'rather'
}

EMOTIONAL_VERBS = {
    'feel', 'felt', 'feeling', 'emotion', 'emotional', 'express', 'express',
    'cry', 'laugh', 'smile', 'frown', 'scream', 'shout', 'whisper', 'sigh'
}


def calculate_emotional_range(tokens: List[str]) -> float:
    """
    Calculate diversity of emotions expressed.
    
    Args:
        tokens: List of tokens
        
    Returns:
        Emotional range score (0-1)
    """
    if not tokens:
        return 0.0
    
    token_set = set(token.lower() for token in tokens)
    
    positive_count = len(token_set.intersection(POSITIVE_EMOTIONS))
    negative_count = len(token_set.intersection(NEGATIVE_EMOTIONS))
    
    # Range = presence of both positive and negative
    total_emotional = positive_count + negative_count
    if total_emotional == 0:
        return 0.0
    
    # Balance score (higher when both are present)
    positive_ratio = positive_count / total_emotional
    negative_ratio = negative_count / total_emotional
    
    # Balance metric
    balance = 1.0 - abs(positive_ratio - negative_ratio)
    
    # Also reward total emotional vocabulary size
    emotional_diversity = min(total_emotional / 20.0, 1.0)
    
    return 0.6 * balance + 0.4 * emotional_diversity


def calculate_emotional_intensity(tokens: List[str]) -> float:
    """
    Calculate intensity of emotional language.
    
    Args:
        tokens: List of tokens
        
    Returns:
        Emotional intensity score (0-1)
    """
    if not tokens:
        return 0.0
    
    token_list = [token.lower() for token in tokens]
    token_set = set(token_list)
    
    # Count intensity modifiers
    intensity_count = sum(1 for token in token_list if token in INTENSITY_MODIFIERS)
    
    # Count emotional verbs (indicating active emotional expression)
    emotional_verb_count = len(token_set.intersection(EMOTIONAL_VERBS))
    
    # Count strong emotional words (both positive and negative)
    strong_emotions = token_set.intersection(POSITIVE_EMOTIONS.union(NEGATIVE_EMOTIONS))
    strong_emotion_count = len(strong_emotions)
    
    # Normalize by total tokens
    intensity_density = intensity_count / len(tokens) if tokens else 0.0
    verb_density = emotional_verb_count / len(tokens) if tokens else 0.0
    emotion_density = strong_emotion_count / len(tokens) if tokens else 0.0
    
    # Combined intensity score
    score = (
        0.3 * min(intensity_density * 20, 1.0) +
        0.3 * min(verb_density * 10, 1.0) +
        0.4 * min(emotion_density * 5, 1.0)
    )
    
    return score


def calculate_polarity_shifts(lines: List[List[str]]) -> float:
    """
    Calculate variation in emotional polarity across lines.
    
    Args:
        lines: List of lines
        
    Returns:
        Polarity shift score (0-1)
    """
    if len(lines) < 2:
        return 0.0
    
    line_polarities = []
    for line in lines:
        line_tokens = set(token.lower() for token in line)
        positive_in_line = len(line_tokens.intersection(POSITIVE_EMOTIONS))
        negative_in_line = len(line_tokens.intersection(NEGATIVE_EMOTIONS))
        
        # Calculate polarity for this line (-1 to 1)
        if positive_in_line + negative_in_line == 0:
            polarity = 0.0  # Neutral
        else:
            polarity = (positive_in_line - negative_in_line) / (positive_in_line + negative_in_line)
        
        line_polarities.append(polarity)
    
    if not line_polarities:
        return 0.0
    
    # Calculate variation in polarity
    mean_polarity = sum(line_polarities) / len(line_polarities)
    variance = sum((p - mean_polarity) ** 2 for p in line_polarities) / len(line_polarities)
    std_dev = variance ** 0.5
    
    # Higher variation = more shifts = higher score
    # Normalize to 0-1
    shift_score = min(std_dev * 2, 1.0)
    
    # Also reward presence of emotional content
    emotional_lines = sum(1 for p in line_polarities if p != 0.0)
    emotional_presence = emotional_lines / len(line_polarities) if line_polarities else 0.0
    
    return 0.6 * shift_score + 0.4 * emotional_presence


def analyze_emotional_expressiveness(song_data: Dict) -> float:
    """
    Compute emotional expressiveness score for a song.
    
    Args:
        song_data: Processed song data dictionary
        
    Returns:
        Emotional expressiveness score (0-1)
    """
    tokens = song_data.get("lyrics_tokenized", [])
    lines = song_data.get("lyrics_lines", [])
    
    if not tokens:
        return 0.0
    
    # Calculate component metrics
    emotional_range = calculate_emotional_range(tokens)
    intensity = calculate_emotional_intensity(tokens)
    polarity_shifts = calculate_polarity_shifts(lines) if lines else 0.0
    
    # Weighted combination
    score = (
        0.35 * emotional_range +
        0.35 * intensity +
        0.30 * polarity_shifts
    )
    
    return min(max(score, 0.0), 1.0)


def process_all_songs() -> Dict[tuple, float]:
    """
    Process all songs and return emotional expressiveness scores.
    
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
                    score = analyze_emotional_expressiveness(song_data)
                    
                    key = (song_data.get("artist"), song_data.get("song_title"))
                    scores[key] = score
                except Exception as e:
                    print(f"Error processing song: {e}")
                    continue
    
    return scores


if __name__ == "__main__":
    scores = process_all_songs()
    print(f"Computed emotional expressiveness scores for {len(scores)} songs")
    for (artist, song), score in list(scores.items())[:5]:
        print(f"{artist} - {song}: {score:.4f}")
