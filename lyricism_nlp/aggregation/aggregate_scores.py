"""
Aggregation module for combining component scores into composite lyricism scores.

Normalizes scores across artists, aggregates song → album → artist,
and balances consistency with peak performance.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    AGGREGATION_WEIGHTS
)

# Import all analysis modules
from analysis.lexical_complexity import process_all_songs as get_lexical_scores
from analysis.rhyme_complexity import process_all_songs as get_rhyme_scores
from analysis.semantic_depth import process_all_songs as get_semantic_scores
from analysis.narrative_structure import process_all_songs as get_narrative_scores
from analysis.emotional_expressiveness import process_all_songs as get_emotional_scores


def normalize_scores(scores: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    """
    Normalize scores to 0-1 range using min-max normalization.
    
    Args:
        scores: Dictionary mapping (artist, song) to score
        
    Returns:
        Normalized scores
    """
    if not scores:
        return {}
    
    score_values = list(scores.values())
    min_score = min(score_values)
    max_score = max(score_values)
    
    if max_score == min_score:
        return {k: 0.5 for k in scores.keys()}
    
    normalized = {}
    for key, score in scores.items():
        normalized[key] = (score - min_score) / (max_score - min_score)
    
    return normalized


def load_song_metadata() -> Dict[Tuple[str, str], Dict]:
    """
    Load song metadata from processed lyrics.
    
    Returns:
        Dictionary mapping (artist, song_title) to metadata
    """
    processed_file = Path(PROCESSED_DATA_DIR) / "all_processed_lyrics.jsonl"
    
    if not processed_file.exists():
        return {}
    
    metadata = {}
    with open(processed_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    song_data = json.loads(line)
                    key = (song_data.get("artist"), song_data.get("song_title"))
                    metadata[key] = {
                        "album": song_data.get("album", "Unknown"),
                        "genius_song_id": song_data.get("genius_song_id"),
                    }
                except Exception as e:
                    continue
    
    return metadata


def aggregate_song_scores() -> List[Dict]:
    """
    Aggregate all component scores into song-level composite scores.
    
    Returns:
        List of song score dictionaries
    """
    print("Loading component scores...")
    
    # Get all component scores
    lexical_scores = get_lexical_scores()
    rhyme_scores = get_rhyme_scores()
    semantic_scores = get_semantic_scores()
    narrative_scores = get_narrative_scores()
    emotional_scores = get_emotional_scores()
    
    # Normalize each component
    lexical_norm = normalize_scores(lexical_scores)
    rhyme_norm = normalize_scores(rhyme_scores)
    semantic_norm = normalize_scores(semantic_scores)
    narrative_norm = normalize_scores(narrative_scores)
    emotional_norm = normalize_scores(emotional_scores)
    
    # Get all unique songs
    all_songs = set()
    all_songs.update(lexical_scores.keys())
    all_songs.update(rhyme_scores.keys())
    all_songs.update(semantic_scores.keys())
    all_songs.update(narrative_scores.keys())
    all_songs.update(emotional_scores.keys())
    
    # Load metadata
    metadata = load_song_metadata()
    
    # Aggregate scores per song
    song_results = []
    for artist, song_title in all_songs:
        # Get component scores (default to 0 if missing)
        lexical = lexical_norm.get((artist, song_title), 0.0)
        rhyme = rhyme_norm.get((artist, song_title), 0.0)
        semantic = semantic_norm.get((artist, song_title), 0.0)
        narrative = narrative_norm.get((artist, song_title), 0.0)
        emotional = emotional_norm.get((artist, song_title), 0.0)
        
        # Weighted composite score
        composite = (
            AGGREGATION_WEIGHTS["lexical_complexity"] * lexical +
            AGGREGATION_WEIGHTS["rhyme_complexity"] * rhyme +
            AGGREGATION_WEIGHTS["semantic_depth"] * semantic +
            AGGREGATION_WEIGHTS["narrative_structure"] * narrative +
            AGGREGATION_WEIGHTS["emotional_expressiveness"] * emotional
        )
        
        # Get metadata
        meta = metadata.get((artist, song_title), {})
        
        song_result = {
            "artist": artist,
            "album": meta.get("album", "Unknown"),
            "song_title": song_title,
            "lexical_complexity": lexical,
            "rhyme_complexity": rhyme,
            "semantic_depth": semantic,
            "narrative_structure": narrative,
            "emotional_expressiveness": emotional,
            "composite_score": composite
        }
        
        song_results.append(song_result)
    
    return song_results


def aggregate_album_scores(song_scores: List[Dict]) -> List[Dict]:
    """
    Aggregate song scores to album level.
    Balance consistency (mean) with peak performance (max).
    
    Args:
        song_scores: List of song score dictionaries
        
    Returns:
        List of album score dictionaries
    """
    # Group by artist and album
    album_groups = defaultdict(list)
    for song in song_scores:
        key = (song["artist"], song["album"])
        album_groups[key].append(song)
    
    album_results = []
    for (artist, album), songs in album_groups.items():
        if not songs:
            continue
        
        # Calculate mean and max for each component
        components = [
            "lexical_complexity", "rhyme_complexity", "semantic_depth",
            "narrative_structure", "emotional_expressiveness", "composite_score"
        ]
        
        album_result = {
            "artist": artist,
            "album": album,
            "num_songs": len(songs)
        }
        
        for component in components:
            values = [song[component] for song in songs]
            mean_score = sum(values) / len(values)
            max_score = max(values)
            
            # Balance: 60% consistency (mean), 40% peak (max)
            balanced_score = 0.6 * mean_score + 0.4 * max_score
            
            album_result[f"{component}_mean"] = mean_score
            album_result[f"{component}_max"] = max_score
            album_result[f"{component}_balanced"] = balanced_score
        
        album_results.append(album_result)
    
    return album_results


def aggregate_artist_scores(song_scores: List[Dict]) -> List[Dict]:
    """
    Aggregate song scores to artist level.
    Balance consistency (mean) with peak performance (max).
    
    Args:
        song_scores: List of song score dictionaries
        
    Returns:
        List of artist score dictionaries
    """
    # Group by artist
    artist_groups = defaultdict(list)
    for song in song_scores:
        artist_groups[song["artist"]].append(song)
    
    artist_results = []
    for artist, songs in artist_groups.items():
        if not songs:
            continue
        
        # Calculate mean and max for each component
        components = [
            "lexical_complexity", "rhyme_complexity", "semantic_depth",
            "narrative_structure", "emotional_expressiveness", "composite_score"
        ]
        
        artist_result = {
            "artist": artist,
            "num_songs": len(songs)
        }
        
        for component in components:
            values = [song[component] for song in songs]
            mean_score = sum(values) / len(values)
            max_score = max(values)
            
            # Balance: 60% consistency (mean), 40% peak (max)
            balanced_score = 0.6 * mean_score + 0.4 * max_score
            
            artist_result[f"{component}_mean"] = mean_score
            artist_result[f"{component}_max"] = max_score
            artist_result[f"{component}_balanced"] = balanced_score
        
        artist_results.append(artist_result)
    
    return artist_results


def export_to_csv(song_scores: List[Dict], album_scores: List[Dict], artist_scores: List[Dict]):
    """
    Export all scores to CSV files.
    
    Args:
        song_scores: Song-level scores
        album_scores: Album-level scores
        artist_scores: Artist-level scores
    """
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export song-level scores
    song_file = output_dir / "song_level_scores.csv"
    if song_scores:
        fieldnames = list(song_scores[0].keys())
        with open(song_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(song_scores)
        print(f"Exported {len(song_scores)} song scores to {song_file}")
    
    # Export album-level scores
    album_file = output_dir / "album_level_scores.csv"
    if album_scores:
        fieldnames = list(album_scores[0].keys())
        with open(album_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(album_scores)
        print(f"Exported {len(album_scores)} album scores to {album_file}")
    
    # Export artist-level scores
    artist_file = output_dir / "artist_level_scores.csv"
    if artist_scores:
        fieldnames = list(artist_scores[0].keys())
        with open(artist_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(artist_scores)
        print(f"Exported {len(artist_scores)} artist scores to {artist_file}")


def main():
    """Main aggregation function."""
    print("Starting score aggregation...")
    
    # Aggregate song scores
    song_scores = aggregate_song_scores()
    print(f"Aggregated scores for {len(song_scores)} songs")
    
    # Aggregate to album level
    album_scores = aggregate_album_scores(song_scores)
    print(f"Aggregated scores for {len(album_scores)} albums")
    
    # Aggregate to artist level
    artist_scores = aggregate_artist_scores(song_scores)
    print(f"Aggregated scores for {len(artist_scores)} artists")
    
    # Export to CSV
    export_to_csv(song_scores, album_scores, artist_scores)
    
    # Return for main.py
    return song_scores, album_scores, artist_scores


if __name__ == "__main__":
    main()
