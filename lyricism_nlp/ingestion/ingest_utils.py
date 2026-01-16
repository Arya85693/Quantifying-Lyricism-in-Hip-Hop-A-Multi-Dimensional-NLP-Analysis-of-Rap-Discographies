"""
Shared utilities for Genius API ingestion.
"""

import json
import re
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.min_interval = 60.0 / max_requests_per_minute
        self.last_request_time = 0.0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()


# Rate limiter will be initialized after config import
rate_limiter = None


def make_genius_request(endpoint: str, params: Dict) -> Optional[Dict]:
    """
    Make a rate-limited request to Genius API.
    
    Args:
        endpoint: API endpoint (e.g., '/search')
        params: Query parameters
        
    Returns:
        JSON response or None if request fails
    """
    # Lazy import to avoid circular dependency
    from config.config import GENIUS_API_TOKEN, GENIUS_API_BASE_URL, GENIUS_RATE_LIMIT_PER_MINUTE
    
    global rate_limiter
    if rate_limiter is None:
        rate_limiter = RateLimiter(GENIUS_RATE_LIMIT_PER_MINUTE)
    
    rate_limiter.wait_if_needed()
    
    headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
    url = f"{GENIUS_API_BASE_URL}{endpoint}"
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None


def search_artist(artist_name: str) -> Optional[int]:
    """
    Search for artist and return Genius artist ID.
    
    Args:
        artist_name: Name of the artist
        
    Returns:
        Genius artist ID or None
    """
    params = {"q": artist_name}
    response = make_genius_request("/search", params)
    
    if not response or "response" not in response:
        return None
    
    hits = response["response"].get("hits", [])
    for hit in hits:
        result = hit.get("result", {})
        primary_artist = result.get("primary_artist", {})
        if primary_artist.get("name", "").lower() == artist_name.lower():
            return primary_artist.get("id")
    
    # Fallback: return first result if exact match not found
    if hits:
        return hits[0].get("result", {}).get("primary_artist", {}).get("id")
    
    return None


def get_artist_songs(artist_id: int, per_page: int = 50, max_pages: int = 20) -> List[Dict]:
    """
    Get all songs for an artist.
    
    Args:
        artist_id: Genius artist ID
        per_page: Songs per page
        max_pages: Maximum pages to fetch
        
    Returns:
        List of song metadata dictionaries
    """
    all_songs = []
    
    for page in range(1, max_pages + 1):
        params = {"per_page": per_page, "page": page, "sort": "popularity"}
        response = make_genius_request(f"/artists/{artist_id}/songs", params)
        
        if not response or "response" not in response:
            break
        
        songs = response["response"].get("songs", [])
        if not songs:
            break
        
        all_songs.extend(songs)
        
        # Check if there are more pages
        next_page = response["response"].get("next_page")
        if not next_page:
            break
    
    return all_songs


def get_song_lyrics(song_id: int) -> Optional[str]:
    """
    Fetch lyrics for a song using Genius song ID.
    Note: Genius API doesn't provide lyrics directly, so we'll use the web URL.
    This function returns None and we'll need to scrape or use alternative method.
    
    For now, we'll store the song metadata and URL for manual/alternative retrieval.
    """
    # Genius API doesn't provide lyrics in their API
    # We'll need to use the song URL or alternative method
    # For this implementation, we'll mark that lyrics need to be fetched separately
    return None


def get_song_details(song_id: int) -> Optional[Dict]:
    """
    Get detailed song information from Genius API.
    
    Args:
        song_id: Genius song ID
        
    Returns:
        Song details dictionary or None
    """
    response = make_genius_request(f"/songs/{song_id}", {})
    
    if not response or "response" not in response:
        return None
    
    return response["response"].get("song")


def normalize_song_title(title: str) -> str:
    """
    Normalize song title for deduplication comparison.
    Removes parenthetical content, special characters, and normalizes whitespace.
    Preserves numbers and meaningful content to distinguish distinct songs.
    
    Args:
        title: Original song title
        
    Returns:
        Normalized title for comparison
    """
    if not title:
        return ""
    
    # Remove content in parentheses (e.g., "(Remix)", "(Live)", "(Demo)", "(Mixed)")
    normalized = re.sub(r'\([^)]*\)', '', title)
    
    # Remove special characters: *, |, and other common separators
    normalized = re.sub(r'[*|]', '', normalized)
    
    # Normalize whitespace: replace multiple spaces/tabs with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Strip leading/trailing whitespace and convert to lowercase
    normalized = normalized.strip().lower()
    
    return normalized


def load_existing_songs(output_file: Path) -> set:
    """
    Load set of already ingested song IDs from JSONL file.
    
    Args:
        output_file: Path to JSONL file
        
    Returns:
        Set of Genius song IDs
    """
    if not output_file.exists():
        return set()
    
    existing_ids = set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        existing_ids.add(data.get("genius_song_id"))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        return set()
    
    return existing_ids


def load_existing_normalized_titles(output_file: Path) -> set:
    """
    Load set of normalized song titles from JSONL file for deduplication.
    
    Args:
        output_file: Path to JSONL file
        
    Returns:
        Set of normalized song titles
    """
    if not output_file.exists():
        return set()
    
    normalized_titles = set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        song_title = data.get("song_title", "")
                        if song_title:
                            normalized = normalize_song_title(song_title)
                            if normalized:
                                normalized_titles.add(normalized)
                    except json.JSONDecodeError:
                        continue
    except Exception:
        return set()
    
    return normalized_titles


def save_song_data(output_file: Path, song_data: Dict):
    """
    Append song data to JSONL file.
    
    Args:
        output_file: Path to JSONL file
        song_data: Dictionary containing song data
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(song_data, ensure_ascii=False) + "\n")


def fetch_lyrics_from_url(url: str) -> Optional[str]:
    """
    Attempt to fetch lyrics from Genius URL.
    This is a placeholder - in production, you'd use a library like lyricsgenius
    or scrape the page. For now, we'll use a simple approach.
    
    Args:
        url: Genius song URL
        
    Returns:
        Lyrics text or None
    """
    try:
        from config.config import GENIUS_API_TOKEN
        import lyricsgenius
        genius = lyricsgenius.Genius(GENIUS_API_TOKEN)
        # Extract song ID from URL if possible
        # For now, return None and we'll handle this differently
        return None
    except ImportError:
        # Fallback: try basic scraping
        try:
            global rate_limiter
            if rate_limiter is not None:
                rate_limiter.wait_if_needed()
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Simple extraction - in production use proper HTML parsing
                # This is a simplified version
                return None
        except:
            return None
    
    return None
