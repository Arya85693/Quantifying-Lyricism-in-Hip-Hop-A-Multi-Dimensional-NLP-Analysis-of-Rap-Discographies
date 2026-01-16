"""
Preprocessing module for cleaning and tokenizing lyrics.
Preserves original text while creating cleaned versions for analysis.
"""

import json
import re
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def clean_text(text: str, preserve_line_breaks: bool = True) -> Dict[str, any]:
    """
    Clean and tokenize text while preserving original.
    
    Args:
        text: Raw text to clean
        preserve_line_breaks: Whether to preserve line breaks for rhyme analysis
        
    Returns:
        Dictionary with original, cleaned, and tokenized versions
    """
    if not text:
        return {
            "original": "",
            "cleaned": "",
            "tokenized": [],
            "lines": []
        }
    
    # Preserve original
    original = text
    
    # Create cleaned version
    cleaned = text.lower()
    
    # Normalize whitespace (but preserve line breaks if needed)
    if preserve_line_breaks:
        # Replace multiple spaces with single space, but keep newlines
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r' +', ' ', cleaned)
    else:
        cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove punctuation except apostrophes
    cleaned = re.sub(r"[^\w\s']", '', cleaned)
    
    # Normalize apostrophes
    cleaned = cleaned.replace("'", "'")
    cleaned = cleaned.replace("'", "'")
    
    # Tokenize
    if preserve_line_breaks:
        lines = cleaned.split('\n')
        tokenized_lines = [line.strip().split() for line in lines if line.strip()]
        tokenized = [word for line in tokenized_lines for word in line]
    else:
        tokenized = cleaned.split()
        tokenized_lines = []
    
    return {
        "original": original,
        "cleaned": cleaned,
        "tokenized": tokenized,
        "lines": tokenized_lines if preserve_line_breaks else []
    }


def load_raw_lyrics(artist_file: Path) -> List[Dict]:
    """
    Load raw lyrics from JSONL file.
    
    Args:
        artist_file: Path to JSONL file
        
    Returns:
        List of song dictionaries
    """
    songs = []
    if not artist_file.exists():
        return songs
    
    with open(artist_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    song_data = json.loads(line)
                    songs.append(song_data)
                except json.JSONDecodeError:
                    continue
    
    return songs


def process_all_artists():
    """Process lyrics for all artists."""
    raw_dir = Path(RAW_DATA_DIR)
    processed_dir = Path(PROCESSED_DATA_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    artist_files = {
        "dave": raw_dir / "dave_lyrics.jsonl",
        "drake": raw_dir / "drake_lyrics.jsonl",
        "kanye_west": raw_dir / "kanye_west_lyrics.jsonl",
        "kendrick_lamar": raw_dir / "kendrick_lamar_lyrics.jsonl",
        "lil_wayne": raw_dir / "lil_wayne_lyrics.jsonl",
    }
    
    all_processed = []
    
    for artist_name, artist_file in artist_files.items():
        print(f"Processing {artist_name}...")
        songs = load_raw_lyrics(artist_file)
        
        for song in songs:
            lyrics_raw = song.get("lyrics_raw", "")
            if not lyrics_raw:
                continue
            
            # Clean lyrics
            cleaned_data = clean_text(lyrics_raw, preserve_line_breaks=True)
            
            # Create processed song data
            processed_song = {
                "artist": song.get("artist"),
                "album": song.get("album"),
                "song_title": song.get("song_title"),
                "genius_song_id": song.get("genius_song_id"),
                "genius_url": song.get("genius_url"),
                "lyrics_original": cleaned_data["original"],
                "lyrics_cleaned": cleaned_data["cleaned"],
                "lyrics_tokenized": cleaned_data["tokenized"],
                "lyrics_lines": cleaned_data["lines"],
                "retrieval_timestamp": song.get("retrieval_timestamp"),
                "data_source": song.get("data_source")
            }
            
            all_processed.append(processed_song)
        
        print(f"Processed {len(songs)} songs for {artist_name}")
    
    # Save all processed data
    output_file = processed_dir / "all_processed_lyrics.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for song in all_processed:
            f.write(json.dumps(song, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(all_processed)} processed songs to {output_file}")
    
    return all_processed


if __name__ == "__main__":
    process_all_artists()
