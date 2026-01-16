"""
Semantic Depth & Thematic Diversity Analysis Module.

Measures conceptual content:
- Topic diversity (vocabulary breadth across themes)
- Conceptual range (abstract vs concrete language)
- Thematic diversity (coverage of different topics)
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

# Abstract concept indicators (simplified)
ABSTRACT_WORDS = {
    'love', 'hate', 'fear', 'hope', 'dream', 'soul', 'spirit', 'mind', 'heart',
    'truth', 'lie', 'justice', 'freedom', 'power', 'strength', 'weakness',
    'beauty', 'ugliness', 'good', 'evil', 'right', 'wrong', 'moral', 'immoral',
    'existence', 'reality', 'illusion', 'perception', 'understanding', 'knowledge',
    'wisdom', 'ignorance', 'belief', 'faith', 'doubt', 'certainty', 'uncertainty',
    'emotion', 'feeling', 'passion', 'desire', 'longing', 'regret', 'remorse',
    'guilt', 'shame', 'pride', 'humility', 'gratitude', 'appreciation',
    'philosophy', 'meaning', 'purpose', 'destiny', 'fate', 'chance', 'luck',
    'time', 'eternity', 'infinity', 'nothingness', 'void', 'emptiness', 'fullness'
}

# Concrete concept indicators
CONCRETE_WORDS = {
    'car', 'house', 'money', 'cash', 'dollar', 'phone', 'gun', 'knife', 'drug',
    'food', 'water', 'drink', 'clothes', 'shoes', 'hat', 'chain', 'watch',
    'street', 'corner', 'building', 'room', 'door', 'window', 'wall', 'floor',
    'hand', 'eye', 'face', 'head', 'body', 'foot', 'leg', 'arm', 'finger',
    'red', 'blue', 'green', 'black', 'white', 'yellow', 'orange', 'purple',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'
}

# Thematic categories (simplified topic modeling)
THEMATIC_CATEGORIES = {
    'wealth': {'money', 'rich', 'poor', 'cash', 'dollar', 'million', 'billion', 'wealth', 'fortune'},
    'violence': {'gun', 'kill', 'death', 'blood', 'violence', 'fight', 'war', 'shoot', 'stab'},
    'love': {'love', 'heart', 'kiss', 'romance', 'girl', 'woman', 'baby', 'darling'},
    'success': {'success', 'win', 'victory', 'champion', 'king', 'queen', 'boss', 'leader'},
    'struggle': {'struggle', 'pain', 'suffer', 'hardship', 'difficult', 'challenge', 'obstacle'},
    'identity': {'self', 'identity', 'who', 'am', 'i', 'me', 'myself', 'person', 'individual'},
    'community': {'people', 'community', 'neighborhood', 'hood', 'city', 'town', 'street'},
    'art': {'art', 'music', 'song', 'rhyme', 'verse', 'poetry', 'lyric', 'beat'},
    'time': {'time', 'day', 'night', 'morning', 'evening', 'year', 'month', 'week', 'hour'},
    'emotion': {'feel', 'emotion', 'happy', 'sad', 'angry', 'mad', 'glad', 'excited'}
}


def calculate_topic_diversity(tokens: List[str]) -> float:
    """
    Calculate topic diversity based on thematic category coverage.
    
    Args:
        tokens: List of tokens
        
    Returns:
        Topic diversity score (0-1)
    """
    if not tokens:
        return 0.0
    
    token_set = set(token.lower() for token in tokens)
    
    # Count how many thematic categories are represented
    categories_represented = 0
    for category, keywords in THEMATIC_CATEGORIES.items():
        if token_set.intersection(keywords):
            categories_represented += 1
    
    # Diversity = proportion of categories covered
    total_categories = len(THEMATIC_CATEGORIES)
    diversity = categories_represented / total_categories if total_categories > 0 else 0.0
    
    return diversity


def calculate_conceptual_range(tokens: List[str]) -> float:
    """
    Calculate conceptual range (abstract vs concrete balance).
    
    Args:
        tokens: List of tokens
        
    Returns:
        Conceptual range score (0-1)
    """
    if not tokens:
        return 0.0
    
    token_set = set(token.lower() for token in tokens)
    
    abstract_count = len(token_set.intersection(ABSTRACT_WORDS))
    concrete_count = len(token_set.intersection(CONCRETE_WORDS))
    
    total_conceptual = abstract_count + concrete_count
    if total_conceptual == 0:
        return 0.0
    
    # Balance score: reward both abstract and concrete presence
    # Maximum when both are well-represented
    abstract_ratio = abstract_count / total_conceptual
    concrete_ratio = concrete_count / total_conceptual
    
    # Balance metric (higher when both are present)
    balance = 1.0 - abs(abstract_ratio - concrete_ratio)
    
    # Scale by total conceptual word presence
    conceptual_density = min(total_conceptual / len(tokens), 1.0) if tokens else 0.0
    
    # Combined score
    score = 0.5 * balance + 0.5 * conceptual_density
    
    return score


def calculate_vocabulary_breadth(tokens: List[str]) -> float:
    """
    Calculate vocabulary breadth (unique words relative to total).
    
    Args:
        tokens: List of tokens
        
    Returns:
        Vocabulary breadth score (0-1)
    """
    if not tokens:
        return 0.0
    
    unique_words = len(set(tokens))
    total_words = len(tokens)
    
    # Type-token ratio, but also consider absolute vocabulary size
    ttr = unique_words / total_words if total_words > 0 else 0.0
    
    # Also reward larger absolute vocabularies (normalized)
    # Typical song might have 50-200 unique words
    vocab_size_score = min(unique_words / 200.0, 1.0)
    
    # Combined
    score = 0.6 * ttr + 0.4 * vocab_size_score
    
    return score


def analyze_semantic_depth(song_data: Dict) -> float:
    """
    Compute semantic depth score for a song.
    
    Args:
        song_data: Processed song data dictionary
        
    Returns:
        Semantic depth score (0-1)
    """
    tokens = song_data.get("lyrics_tokenized", [])
    
    if not tokens:
        return 0.0
    
    # Calculate component metrics
    topic_diversity = calculate_topic_diversity(tokens)
    conceptual_range = calculate_conceptual_range(tokens)
    vocab_breadth = calculate_vocabulary_breadth(tokens)
    
    # Weighted combination
    score = (
        0.35 * topic_diversity +
        0.35 * conceptual_range +
        0.30 * vocab_breadth
    )
    
    return min(max(score, 0.0), 1.0)


def process_all_songs() -> Dict[tuple, float]:
    """
    Process all songs and return semantic depth scores.
    
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
                    score = analyze_semantic_depth(song_data)
                    
                    key = (song_data.get("artist"), song_data.get("song_title"))
                    scores[key] = score
                except Exception as e:
                    print(f"Error processing song: {e}")
                    continue
    
    return scores


if __name__ == "__main__":
    scores = process_all_songs()
    print(f"Computed semantic depth scores for {len(scores)} songs")
    for (artist, song), score in list(scores.items())[:5]:
        print(f"{artist} - {song}: {score:.4f}")
