"""
Configuration file for lyricism NLP analysis project.
"""

# Genius API Configuration
GENIUS_API_TOKEN = "GENERATE YOUR OWN GENIUS API - withheld mine for security reasons"
GENIUS_API_BASE_URL = "https://api.genius.com"
GENIUS_RATE_LIMIT_PER_MINUTE = 60

# Artist Configuration
ARTISTS = {
    "dave": {"genius_id": None, "search_name": "Dave"},
    "drake": {"genius_id": None, "search_name": "Drake"},
    "kanye_west": {"genius_id": None, "search_name": "Kanye West"},
    "kendrick_lamar": {"genius_id": None, "search_name": "Kendrick Lamar"},
    "lil_wayne": {"genius_id": None, "search_name": "Lil Wayne"},
}

# Data Paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_DIR = "outputs"

# Analysis Configuration
RANDOM_SEED = 42

# Aggregation Weights
AGGREGATION_WEIGHTS = {
    "lexical_complexity": 0.20,
    "rhyme_complexity": 0.25,
    "semantic_depth": 0.20,
    "narrative_structure": 0.15,
    "emotional_expressiveness": 0.20,
}

