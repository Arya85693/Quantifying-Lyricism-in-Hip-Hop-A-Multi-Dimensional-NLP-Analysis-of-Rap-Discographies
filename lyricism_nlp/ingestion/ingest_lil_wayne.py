"""
Ingestion script for Lil Wayne's discography.
"""

import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import RAW_DATA_DIR
from ingestion.ingest_utils import (
    search_artist,
    get_artist_songs,
    get_song_details,
    load_existing_songs,
    load_existing_normalized_titles,
    normalize_song_title,
    save_song_data,
)
import lyricsgenius


def ingest_lil_wayne():
    """Main ingestion function for Lil Wayne."""
    artist_name = "Lil Wayne"
    output_file = Path(RAW_DATA_DIR) / "lil_wayne_lyrics.jsonl"
    log_file = Path(RAW_DATA_DIR) / "ingestion_errors_lil_wayne.log"
    
    # Ensure log folder exists to prevent FileNotFoundError on Windows
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
        ]
    )
    logger = logging.getLogger(__name__)
    
    print(f"Starting ingestion for {artist_name}...")
    print(f"Errors will be logged to: {log_file}")
    
    # Load existing songs
    existing_ids = load_existing_songs(output_file)
    existing_normalized_titles = load_existing_normalized_titles(output_file)
    print(f"Found {len(existing_ids)} existing songs")
    
    # Search for artist
    artist_id = search_artist(artist_name)
    if not artist_id:
        print(f"Could not find artist: {artist_name}")
        return
    
    print(f"Found artist ID: {artist_id}")
    
    # Get all songs
    songs = get_artist_songs(artist_id, per_page=50, max_pages=15)
    print(f"Found {len(songs)} songs")
    
    # Fetch lyrics for each song
    from config.config import GENIUS_API_TOKEN
    genius = lyricsgenius.Genius(GENIUS_API_TOKEN)
    genius.remove_section_headers = True
    genius.skip_non_songs = True
    
    ingested_count = 0
    skipped_count = 0
    error_count = 0
    
    for song in songs:
        try:
            song_id = song.get("id")
            if not song_id or song_id in existing_ids:
                skipped_count += 1
                continue
            
            song_title = song.get("title", "")
            song_url = song.get("url", "")
            
            # Get full song details
            album_name = "Unknown"
            try:
                song_details = get_song_details(song_id)
                if song_details:
                    album = song_details.get("album")
                    if album:
                        album_name = album.get("name", "Unknown")
            except Exception as e:
                logger.error(f"Song: {song_title} | Exception: {type(e).__name__} | Message: {str(e)}")
                error_count += 1
                # Continue with Unknown album name
            
            # Fetch lyrics
            lyrics_raw = None
            try:
                song_obj = genius.search_song(song_title, artist_name)
                if song_obj:
                    lyrics_raw = song_obj.lyrics
            except Exception as e:
                logger.error(f"Song: {song_title} | Exception: {type(e).__name__} | Message: {str(e)}")
                error_count += 1
                # Continue with lyrics_raw=None
            
            # Check for duplicate based on normalized title
            normalized_title = normalize_song_title(song_title)
            if normalized_title and normalized_title in existing_normalized_titles:
                logger.error(f"Song: {song_title} | Skipped: Duplicate (normalized title: {normalized_title})")
                skipped_count += 1
                continue
            
            # Rate limiting: wait 1-2 seconds between song requests
            time.sleep(random.uniform(1.0, 2.0))
            
            # Prepare song data (save even if lyrics_raw is None)
            song_data = {
                "artist": artist_name,
                "album": album_name,
                "song_title": song_title,
                "genius_song_id": song_id,
                "genius_url": song_url,
                "lyrics_raw": lyrics_raw,
                "retrieval_timestamp": datetime.now().isoformat(),
                "data_source": "genius_api"
            }
            
            save_song_data(output_file, song_data)
            existing_normalized_titles.add(normalized_title)
            ingested_count += 1
            
            if ingested_count % 10 == 0:
                print(f"Progress: {ingested_count} songs ingested")
        
        except Exception as e:
            # Catch any unexpected errors to prevent script from stopping
            song_title = song.get("title", "Unknown") if song else "Unknown"
            logger.error(f"Song: {song_title} | Exception: {type(e).__name__} | Message: {str(e)}")
            error_count += 1
            continue
    
    print(f"Ingestion complete. Ingested: {ingested_count}, Skipped: {skipped_count}, Errors: {error_count}")


if __name__ == "__main__":
    ingest_lil_wayne()
