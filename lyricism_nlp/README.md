# Quantifying Lyricism in Hip-Hop: A Multi-Dimensional NLP Analysis

> Multi-dimensional NLP analysis of hip-hop discographies. Quantifies lyricism across lexical complexity, rhyme patterns, semantic depth, narrative structure, and emotional expressiveness with transparent, interpretable metrics.

A comprehensive Python project that quantitatively analyzes hip-hop lyricism across multiple artists using interpretable NLP techniques. This project decomposes "lyricism" into five measurable dimensions and aggregates them into transparent composite scores.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Outputs](#outputs)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## Project Description

This project analyzes hip-hop lyrics from five artists (Dave, Drake, Kanye West, Kendrick Lamar, and Lil Wayne) using natural language processing to measure lyricism across five key dimensions:

1. **Lexical & Linguistic Complexity**: Measures vocabulary richness, word rarity, syllabic density, and syntactic variation
2. **Rhyme & Phonetic Complexity**: Analyzes rhyme density, internal rhymes, multisyllabic rhyme chains, and phonetic pattern variation using the CMU Pronouncing Dictionary
3. **Semantic Depth & Thematic Diversity**: Evaluates topic diversity, conceptual range, and abstract vs. concrete language
4. **Narrative Structure & Coherence**: Assesses storytelling ability, first-person continuity, temporal progression, and thematic cohesion
5. **Emotional Expressiveness & Intensity**: Measures emotional content, range, polarity shifts, and emotional intensity

Each dimension is computed as a numeric score per song, then aggregated to album and artist levels. The final composite score balances consistency (mean) and peak performance (max) using a 60/40 weighting scheme.

## Installation

### Prerequisites

- **Python 3.8 or higher** (Python 3.9+ recommended)
- pip (Python package installer)

### Step 1: Clone or Download the Project

Navigate to the project directory:

```bash
cd lyricism_nlp
```

### Step 2: Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

This will install:
- `requests` - HTTP library for API calls
- `lyricsgenius` - Genius API wrapper
- `pronouncing` - CMU Pronouncing Dictionary access
- `matplotlib` - Plotting library
- `seaborn` - Statistical visualization
- `pandas` - Data manipulation
- `numpy` - Numerical computing

### Step 3: Configure Genius API (Optional)

The project includes a Genius API token in `config/config.py`. If you need to use your own token:

1. Sign up for a free account at [Genius API](https://genius.com/api-clients)
2. Generate an access token
3. Update `GENIUS_API_TOKEN` in `config/config.py`

**Note**: The free tier allows up to 60 requests per minute. The project includes rate limiting to respect these limits.

## Project Structure

```
lyricism_nlp/
├── config/
│   └── config.py              # Configuration settings (API tokens, paths, weights)
├── ingestion/
│   ├── ingest_utils.py        # Shared utilities for Genius API interaction
│   ├── ingest_dave.py         # Ingestion script for Dave
│   ├── ingest_drake.py        # Ingestion script for Drake
│   ├── ingest_kanye_west.py   # Ingestion script for Kanye West
│   ├── ingest_kendrick_lamar.py # Ingestion script for Kendrick Lamar
│   └── ingest_lil_wayne.py     # Ingestion script for Lil Wayne
├── preprocessing/
│   └── clean_lyrics.py         # Lyrics cleaning and tokenization
├── analysis/
│   ├── lexical_complexity.py      # Lexical & linguistic complexity analysis
│   ├── rhyme_complexity.py        # Rhyme & phonetic complexity analysis
│   ├── semantic_depth.py          # Semantic depth & thematic diversity analysis
│   ├── narrative_structure.py     # Narrative structure & coherence analysis
│   └── emotional_expressiveness.py # Emotional expressiveness & intensity analysis
├── aggregation/
│   └── aggregate_scores.py    # Score aggregation (song → album → artist)
├── visualization/
│   └── generate_plots.py       # Visualization generation
├── cleaning/
│   └── clean_albums.py         # Album filtering by song count threshold
├── data/
│   ├── raw/                    # Raw lyrics from Genius API (JSONL format)
│   └── processed/               # Cleaned and tokenized lyrics
├── outputs/
│   ├── song_level_scores.csv   # Per-song scores for all dimensions
│   ├── album_level_scores.csv # Per-album aggregated scores
│   ├── album_level_scores_clean.csv # Filtered albums (≥6 songs)
│   ├── artist_level_scores.csv # Per-artist aggregated scores
│   └── visualizations/         # Generated PNG visualizations
├── main.py                     # Main pipeline execution script
└── requirements.txt            # Python dependencies
```

## Usage

### Running the Full Pipeline

To run the complete analysis pipeline from ingestion to visualization:

```bash
python main.py
```

This will:
1. **Ingest** lyrics from Genius API for all artists
2. **Preprocess** and clean the lyrics
3. **Analyze** each song across five dimensions
4. **Aggregate** scores to album and artist levels
5. **Generate** visualizations (PNG files)
6. **Print** a ranked summary of artists

**Note**: The first run may take several hours due to API rate limiting. Subsequent runs are faster as lyrics are cached locally.

### Running Individual Steps

#### Ingestion Only

To fetch lyrics for a specific artist:

```bash
# PowerShell (Windows)
python -m ingestion.ingest_drake

# Terminal (Mac/Linux)
python -m ingestion.ingest_drake
```

Available scripts:
- `ingest_dave.py`
- `ingest_drake.py`
- `ingest_kanye_west.py`
- `ingest_kendrick_lamar.py`
- `ingest_lil_wayne.py`

#### Preprocessing Only

To clean and tokenize lyrics:

```bash
python -m preprocessing.clean_lyrics
```

#### Aggregation Only

To aggregate scores (includes analysis step):

```bash
python -m aggregation.aggregate_scores
```

#### Visualization Only

To generate visualizations from existing CSV files:

```bash
python -m visualization.generate_plots
```

**Note**: This requires existing CSV files in the `outputs/` directory.

#### Album Cleaning

To filter albums by song count threshold (≥6 songs):

```bash
python -m cleaning.clean_albums
```

This creates:
- `album_level_scores_clean.csv` - Filtered albums
- `albums_removed_due_to_size.csv` - Audit file of removed albums
- `album_cleaning.log` - Detailed log of filtering actions

## Outputs

### CSV Files

All CSV files are saved in the `outputs/` directory:

#### `song_level_scores.csv`
Per-song scores for all five dimensions plus composite score. Columns include:
- `artist`, `album`, `song_title`
- `lexical_complexity`, `rhyme_complexity`, `semantic_depth`, `narrative_structure`, `emotional_expressiveness`
- `composite_score`

#### `album_level_scores.csv`
Aggregated scores per album with mean, max, and balanced (60% mean, 40% max) values for each dimension. Columns include:
- `artist`, `album`, `num_songs`
- `{dimension}_mean`, `{dimension}_max`, `{dimension}_balanced` for each dimension
- `composite_score_balanced`

#### `album_level_scores_clean.csv`
Filtered version containing only albums with ≥6 songs (generated by `clean_albums.py`).

#### `artist_level_scores.csv`
Final aggregated scores per artist with mean, max, and balanced values. Same structure as album-level but aggregated across all albums.

### Visualizations

All PNG visualizations are saved in `outputs/visualizations/`:

#### Artist-Level Comparisons
- **`01_artist_composite_comparison.png`** - Bar chart ranking artists by composite score
- **`03_mean_max_balanced_comparison.png`** - Grouped bar chart comparing all components across artists
- **`03_{component}_comparison.png`** - Individual grouped bar charts for each component (Mean vs Max vs Balanced)
- **`10_radar_chart.png`** - Radar chart comparing artists across all five components

#### Song-Level Distributions
- **`04_song_distributions.png`** - Histograms of composite scores per artist
- **`05_component_boxplots.png`** - Violin plots showing component score variability by artist
- **`songs_per_artist_pie.png`** - Pie chart showing dataset composition (songs per artist)

#### Album-Level Insights
- **`11_album_distributions.png`** - Histograms of album composite scores per artist

#### Heatmaps
- **`08_component_correlation_heatmap.png`** - Correlation matrix between the five components
- **`09_artist_component_heatmap.png`** - Heatmap of component scores by artist

#### Top-3 Album Breakdowns
Located in `outputs/visualizations/album_top3/`:
- **`{artist}_top3_component_bars.png`** - Bar chart comparing top 3 albums across components
- **`{artist}_top3_radar.png`** - Radar chart comparing top 3 albums
- **`{artist}_top3_text.txt`** - Human-readable summary with strengths and weaknesses

## Customization

### Adding New Artists

1. **Add artist to config**: Edit `config/config.py`:
   ```python
   ARTISTS = {
       # ... existing artists ...
       "new_artist": {"genius_id": None, "search_name": "New Artist Name"},
   }
   ```

2. **Create ingestion script**: Copy an existing script (e.g., `ingest_drake.py`) and modify:
   - Import function name
   - Artist name variable
   - Call to `ingest_artist()` function

3. **Add to main pipeline**: Update `main.py`:
   ```python
   from ingestion.ingest_new_artist import ingest_new_artist
   
   def run_ingestion():
       # ... existing artists ...
       print("\nIngesting New Artist...")
       ingest_new_artist()
   ```

4. **Add color for visualizations**: Update `ARTIST_COLORS` in `visualization/generate_plots.py`:
   ```python
   ARTIST_COLORS = {
       # ... existing colors ...
       "New Artist": "#HEXCOLOR",
   }
   ```

### Adjusting Aggregation Weights

Edit `AGGREGATION_WEIGHTS` in `config/config.py`:

```python
AGGREGATION_WEIGHTS = {
    "lexical_complexity": 0.20,        # 20% weight
    "rhyme_complexity": 0.25,           # 25% weight
    "semantic_depth": 0.20,             # 20% weight
    "narrative_structure": 0.15,        # 15% weight
    "emotional_expressiveness": 0.20,  # 20% weight
}
```

**Note**: Weights should sum to 1.0 for proper normalization.

### Changing Visualization Options

Edit `visualization/generate_plots.py` to:
- Modify color palettes in `ARTIST_COLORS`
- Adjust figure sizes in `plt.subplots(figsize=(width, height))`
- Change output format (currently PNG only)
- Add or remove specific visualizations

### Album Filtering Threshold

To change the minimum songs per album threshold, edit `cleaning/clean_albums.py`:

```python
MIN_SONGS_PER_ALBUM = 6  # Change this value
```

## Contributing

Contributions are welcome! To contribute:

1. **Fork the repository** (if applicable)
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make your changes** following the existing code style
4. **Test your changes**: Run the pipeline to ensure everything works
5. **Document your changes**: Update relevant docstrings and comments
6. **Submit a pull request** with a clear description of changes

### Code Style Guidelines

- Follow PEP 8 Python style guide
- Use descriptive variable and function names
- Include docstrings for all functions and classes
- Add comments for complex logic
- Maintain modular structure (one function per task)

### Areas for Contribution

- Additional analysis dimensions
- Improved visualization aesthetics
- Performance optimizations
- Documentation improvements
- Bug fixes and error handling

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- **Genius API** for providing access to lyrics data
- **CMU Pronouncing Dictionary** (via `pronouncing` library) for phonetic analysis
- **Python NLP community** for excellent open-source tools

## Contact

For questions, issues, or suggestions, please open an issue in the repository or contact the project maintainer.

---

**Note**: This project is designed for research and educational purposes. The quantitative scores are not intended to declare an objective "best lyricist" but rather to demonstrate systematic analysis using interpretable NLP techniques.
