"""
Visualization module for lyricism NLP analysis results.

Generates comprehensive visualizations comparing artists across
all dimensions of lyricism analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys
import logging
from math import pi

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import OUTPUT_DIR

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Artist color palette
ARTIST_COLORS = {
    "Dave": "#FF8C00",  # Orange
    "Drake": "#1E90FF",  # Blue
    "Kanye West": "#DC143C",  # Red
    "Kendrick Lamar": "#000000",  # Black
    "Lil Wayne": "#32CD32",  # Green
}

COMPONENT_NAMES = {
    "lexical_complexity": "Lexical Complexity",
    "rhyme_complexity": "Rhyme Complexity",
    "semantic_depth": "Semantic Depth",
    "narrative_structure": "Narrative Structure",
    "emotional_expressiveness": "Emotional Expressiveness",
}


def filter_unknown_albums(album_scores: List[Dict]) -> List[Dict]:
    """
    Filter out albums where album == "Unknown" or album is empty/null.
    Logs filtering actions to visualization_logs.
    
    Args:
        album_scores: List of album score dictionaries
        
    Returns:
        Filtered list of album score dictionaries
    """
    if not album_scores:
        return []
    
    # Setup logging
    log_dir = Path(OUTPUT_DIR) / "visualization_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('album_filter')
    if not logger.handlers:
        log_file = log_dir / "album_viz.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    original_count = len(album_scores)
    filtered = []
    removed_count = 0
    
    for album in album_scores:
        album_name = album.get('album', '').strip() if album.get('album') else ''
        if album_name and album_name.lower() != 'unknown':
            filtered.append(album)
        else:
            removed_count += 1
            artist = album.get('artist', 'Unknown Artist')
            logger.info(f"Filtered out album: {artist} - '{album_name}' (Unknown or empty)")
    
    if removed_count > 0:
        logger.info(f"Filtered {removed_count} out of {original_count} albums (removed Unknown/empty albums)")
    
    return filtered


def save_figure(fig, filename: str, output_dir: Path):
    """
    Save figure as PNG only.
    
    Args:
        fig: Matplotlib figure object
        filename: Filename (should end with .png)
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure filename ends with .png
    if not filename.endswith('.png'):
        filename = f"{filename}.png"
    
    # Save as PNG
    png_path = output_dir / filename
    fig.savefig(png_path, bbox_inches='tight', format='png')
    
    plt.close(fig)


def plot_artist_composite_comparison(artist_scores: List[Dict], output_dir: Path):
    """
    Bar chart of each artist's balanced composite score.
    
    Args:
        artist_scores: List of artist score dictionaries
        output_dir: Output directory for saving plots
    """
    df = pd.DataFrame(artist_scores)
    df = df.sort_values('composite_score_balanced', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [ARTIST_COLORS.get(artist, '#808080') for artist in df['artist']]
    
    bars = ax.barh(df['artist'], df['composite_score_balanced'], color=colors)
    
    ax.set_xlabel('Composite Score (Balanced)', fontweight='bold')
    ax.set_ylabel('Artist', fontweight='bold')
    ax.set_title('Artist Comparison: Composite Lyricism Score', fontweight='bold', pad=15)
    
    # Make y-axis scale more pronounced - use data range with padding
    min_score = df['composite_score_balanced'].min()
    max_score = df['composite_score_balanced'].max()
    score_range = max_score - min_score
    padding = max(0.05, score_range * 0.1)  # At least 5% padding, or 10% of range
    ax.set_xlim(max(0, min_score - padding), min(1, max_score + padding))
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, df['composite_score_balanced'])):
        ax.text(score + padding * 0.1, i, f'{score:.3f}', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, '01_artist_composite_comparison', output_dir)


def plot_mean_max_balanced_comparison(artist_scores: List[Dict], output_dir: Path):
    """
    Create grouped bar charts for component comparisons across artists.
    Replaces line graphs with grouped bar charts for better visual comparison.
    
    Args:
        artist_scores: List of artist score dictionaries
        output_dir: Output directory for saving plots
    """
    df = pd.DataFrame(artist_scores)
    df = df.sort_values('composite_score_balanced', ascending=False)
    
    components = [
        'lexical_complexity',
        'rhyme_complexity',
        'semantic_depth',
        'narrative_structure',
        'emotional_expressiveness',
        'composite_score'
    ]
    
    component_labels = [
        'Lexical Complexity',
        'Rhyme Complexity',
        'Semantic Depth',
        'Narrative Structure',
        'Emotional Expressiveness',
        'Composite Score'
    ]
    
    artists = df['artist'].tolist()
    n_artists = len(artists)
    x = np.arange(n_artists)
    width = 0.25  # Width of bars
    
    # Create one combined graph with all 6 components as grouped bars
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (comp, label, color) in enumerate(zip(components, component_labels, colors)):
        balanced_vals = df[f'{comp}_balanced'].values
        offset = (i - len(components) / 2) * width + width / 2
        ax.bar(x + offset, balanced_vals, width, label=label, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Artist', fontweight='bold')
    ax.set_ylabel('Balanced Score', fontweight='bold')
    ax.set_title('Component Comparison Across Artists (Grouped Bar Chart)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(artists, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='best', framealpha=0.9, ncol=2, fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    save_figure(fig, '03_mean_max_balanced_comparison', output_dir)
    
    # Create individual grouped bar charts for each component (Mean, Max, Balanced)
    for comp, label in zip(components, component_labels):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(n_artists)
        mean_vals = df[f'{comp}_mean'].values
        max_vals = df[f'{comp}_max'].values
        balanced_vals = df[f'{comp}_balanced'].values
        
        width = 0.25
        ax.bar(x - width, mean_vals, width, label='Mean', color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.bar(x, max_vals, width, label='Max', color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.bar(x + width, balanced_vals, width, label='Balanced', color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Artist', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f'{label}: Mean vs Max vs Balanced', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(artists, rotation=45, ha='right')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1)
        
        # Create safe filename from component name
        safe_filename = label.lower().replace(' ', '_').replace(':', '')
        plt.tight_layout()
        save_figure(fig, f'03_{safe_filename}_comparison', output_dir)


def plot_song_distributions(song_scores: List[Dict], output_dir: Path):
    """
    Histograms of composite scores for all songs of each artist.
    
    Args:
        song_scores: List of song score dictionaries
        output_dir: Output directory for saving plots
    """
    df = pd.DataFrame(song_scores)
    
    artists = sorted(df['artist'].unique())
    n_artists = len(artists)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, artist in enumerate(artists):
        ax = axes[i]
        artist_data = df[df['artist'] == artist]['composite_score']
        color = ARTIST_COLORS.get(artist, '#808080')
        
        ax.hist(artist_data, bins=20, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Composite Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{artist}\n(n={len(artist_data)} songs)', fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
    
    # Hide unused subplots
    for i in range(n_artists, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Distribution of Composite Scores by Artist', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_figure(fig, '04_song_distributions', output_dir)


def plot_songs_per_artist_pie(song_scores: List[Dict], output_dir: Path):
    """
    Pie chart showing number of songs per artist.
    
    Args:
        song_scores: List of song score dictionaries
        output_dir: Output directory for saving plots
    """
    df = pd.DataFrame(song_scores)
    
    # Count songs per artist
    song_counts = df['artist'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get colors for each artist
    colors = [ARTIST_COLORS.get(artist, '#808080') for artist in song_counts.index]
    
    # Create pie chart with percentage labels
    wedges, texts, autotexts = ax.pie(song_counts.values, labels=None, autopct='%1.1f%%',
                                       colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    # Create legend with exact counts
    legend_labels = [f'{artist}: {count} songs' for artist, count in song_counts.items()]
    ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), 
              framealpha=0.9, fontsize=10)
    
    ax.set_title('Dataset Composition – Songs per Artist', fontweight='bold', pad=20, fontsize=14)
    
    plt.tight_layout()
    save_figure(fig, 'songs_per_artist_pie', output_dir)


def plot_component_boxplots(song_scores: List[Dict], output_dir: Path):
    """
    Violin plots showing variability of component scores by artist.
    More readable than boxplots, shows distribution shape.
    
    Args:
        song_scores: List of song score dictionaries
        output_dir: Output directory for saving plots
    """
    df = pd.DataFrame(song_scores)
    
    components = [
        'lexical_complexity',
        'rhyme_complexity',
        'semantic_depth',
        'narrative_structure',
        'emotional_expressiveness'
    ]
    
    component_labels = [COMPONENT_NAMES[comp] for comp in components]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for ax, comp, label in zip(axes[:5], components, component_labels):
        # Prepare data for violin plot
        plot_data = []
        plot_labels = []
        
        for artist in sorted(df['artist'].unique()):
            artist_data = df[df['artist'] == artist][comp].values
            plot_data.append(artist_data)
            plot_labels.append(artist)
        
        # Create violin plot
        parts = ax.violinplot(plot_data, positions=range(len(plot_labels)), 
                             showmeans=True, showmedians=True)
        
        # Color the violins
        for i, (pc, artist) in enumerate(zip(parts['bodies'], plot_labels)):
            color = ARTIST_COLORS.get(artist, '#808080')
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # Style the other elements
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if partname in parts:
                parts[partname].set_color('black')
                parts[partname].set_linewidth(1.5)
        
        ax.set_xticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels, rotation=45, ha='right')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(label, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1)
    
    # Hide last subplot
    axes[5].axis('off')
    
    plt.suptitle('Component Score Variability by Artist (Violin Plots)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_figure(fig, '05_component_boxplots', output_dir)


def plot_component_correlation_heatmap(song_scores: List[Dict], output_dir: Path):
    """
    Heatmap showing correlation between the five components across all songs.
    Includes explanation text.
    
    Args:
        song_scores: List of song score dictionaries
        output_dir: Output directory for saving plots
    """
    df = pd.DataFrame(song_scores)
    
    components = [
        'lexical_complexity',
        'rhyme_complexity',
        'semantic_depth',
        'narrative_structure',
        'emotional_expressiveness'
    ]
    
    component_labels = [COMPONENT_NAMES[comp] for comp in components]
    
    corr_matrix = df[components].corr()
    corr_matrix.index = component_labels
    corr_matrix.columns = component_labels
    
    fig, ax = plt.subplots(figsize=(10, 9))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, 
                linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Correlation Between Lyricism Components', fontweight='bold', pad=15)
    
    plt.tight_layout()
    save_figure(fig, '08_component_correlation_heatmap', output_dir)


def plot_artist_component_heatmap(artist_scores: List[Dict], output_dir: Path):
    """
    Heatmap of component scores per artist for quick comparison.
    Includes explanation text.
    
    Args:
        artist_scores: List of artist score dictionaries
        output_dir: Output directory for saving plots
    """
    df = pd.DataFrame(artist_scores)
    df = df.sort_values('composite_score_balanced', ascending=False)
    
    components = [
        'lexical_complexity_balanced',
        'rhyme_complexity_balanced',
        'semantic_depth_balanced',
        'narrative_structure_balanced',
        'emotional_expressiveness_balanced',
        'composite_score_balanced'
    ]
    
    component_labels = [COMPONENT_NAMES[comp.replace('_balanced', '')] for comp in components[:-1]]
    component_labels.append('Composite Score')
    
    heatmap_data = df[components].T
    heatmap_data.index = component_labels
    heatmap_data.columns = df['artist']
    
    fig, ax = plt.subplots(figsize=(10, 9))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                vmin=0, vmax=1, square=False, linewidths=1, 
                cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Component Scores by Artist (Balanced)', fontweight='bold', pad=15)
    ax.set_xlabel('Artist', fontweight='bold')
    ax.set_ylabel('Component', fontweight='bold')
    
    # Add explanation text
    explanation = "This heatmap shows how each artist scores across all components relative to each other.\n" \
                  "Darker colors indicate higher scores. Values are balanced scores (60% mean, 40% max)."
    ax.text(0.5, -0.15, explanation, transform=ax.transAxes, 
            fontsize=9, ha='center', style='italic', wrap=True)
    
    plt.tight_layout()
    save_figure(fig, '09_artist_component_heatmap', output_dir)


def plot_radar_chart(artist_scores: List[Dict], output_dir: Path):
    """
    Radar/spider chart comparing artists across the five components.
    Increased scale to show differences more clearly.
    
    Args:
        artist_scores: List of artist score dictionaries
        output_dir: Output directory for saving plots
    """
    from math import pi
    
    components = [
        'lexical_complexity_balanced',
        'rhyme_complexity_balanced',
        'semantic_depth_balanced',
        'narrative_structure_balanced',
        'emotional_expressiveness_balanced'
    ]
    
    component_labels = [COMPONENT_NAMES[comp.replace('_balanced', '')] for comp in components]
    
    # Number of variables
    N = len(components)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    df = pd.DataFrame(artist_scores)
    
    # Calculate range for better scale
    all_values = []
    for _, row in df.iterrows():
        values = [row[comp] for comp in components]
        all_values.extend(values)
    
    min_val = min(all_values)
    max_val = max(all_values)
    val_range = max_val - min_val
    
    # Use a narrower range with padding to make differences more visible
    y_min = max(0, min_val - val_range * 0.1)
    y_max = min(1, max_val + val_range * 0.1)
    
    for _, row in df.iterrows():
        artist = row['artist']
        values = [row[comp] for comp in components]
        values += values[:1]  # Complete the circle
        
        color = ARTIST_COLORS.get(artist, '#808080')
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=artist, color=color, markersize=8)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(component_labels, fontsize=10)
    ax.set_ylim(y_min, y_max)
    
    # More granular y-axis ticks for better readability
    num_ticks = 8
    y_ticks = np.linspace(y_min, y_max, num_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.2f}' for tick in y_ticks], fontsize=9)
    ax.grid(True, alpha=0.5)
    
    ax.set_title('Artist Comparison: Component Scores (Radar Chart)', 
                 fontweight='bold', pad=25, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, '10_radar_chart', output_dir)


def plot_album_distributions(album_scores: List[Dict], output_dir: Path):
    """
    Histograms of composite_score_balanced for all albums of each artist.
    Mirrors the style of plot_song_distributions but for albums.
    Filters out "Unknown" albums.
    
    Args:
        album_scores: List of album score dictionaries
        output_dir: Output directory for saving plots
    """
    # Filter out Unknown albums
    album_scores = filter_unknown_albums(album_scores)
    if not album_scores:
        return
    
    df = pd.DataFrame(album_scores)
    
    artists = sorted(df['artist'].unique())
    n_artists = len(artists)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, artist in enumerate(artists):
        ax = axes[i]
        artist_data = df[df['artist'] == artist]['composite_score_balanced']
        color = ARTIST_COLORS.get(artist, '#808080')
        
        ax.hist(artist_data, bins=20, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Composite Score (Balanced)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{artist}\n(n={len(artist_data)} albums)', fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
    
    # Hide unused subplots
    for i in range(n_artists, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Album-Level Score Distributions by Artist', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_figure(fig, '11_album_distributions', output_dir)


def generate_top3_album_breakdown(album_scores: List[Dict], output_base_dir: Path):
    """
    Generate top-3 album breakdowns for each artist.
    Creates bar charts, radar charts, and text summaries.
    Filters out "Unknown" albums and skips artists with no valid albums.
    
    Args:
        album_scores: List of album score dictionaries
        output_base_dir: Base output directory (will create album_top3/ subfolder)
    """
    # Filter out Unknown albums
    album_scores = filter_unknown_albums(album_scores)
    if not album_scores:
        return
    
    df = pd.DataFrame(album_scores)
    top3_dir = output_base_dir / "album_top3"
    top3_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging (only if not already configured)
    log_dir = Path(OUTPUT_DIR) / "visualization_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "album_viz.log"
    
    # Create file handler only (avoid duplicate console handlers)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Get or create logger
    logger = logging.getLogger('album_viz')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler())
    
    components = [
        'lexical_complexity_balanced',
        'rhyme_complexity_balanced',
        'semantic_depth_balanced',
        'narrative_structure_balanced',
        'emotional_expressiveness_balanced'
    ]
    
    component_labels = [COMPONENT_NAMES[comp.replace('_balanced', '')] for comp in components]
    
    artists = sorted(df['artist'].unique())
    
    for artist in artists:
        artist_df = df[df['artist'] == artist].copy()
        
        # Get top 3 albums (or all if < 3)
        top_albums = artist_df.nlargest(3, 'composite_score_balanced')
        n_albums = len(top_albums)
        
        if n_albums == 0:
            logger.warning(f"No albums found for {artist}, skipping")
            continue
        
        logger.info(f"Processing {artist}: {n_albums} album(s)")
        
        # Generate bar chart
        plot_top3_component_bars(artist, top_albums, components, component_labels, top3_dir)
        
        # Generate radar chart
        plot_top3_radar(artist, top_albums, components, component_labels, top3_dir)
        
        # Generate text summary
        generate_top3_text_summary(artist, top_albums, components, component_labels, top3_dir)


def plot_top3_component_bars(artist: str, top_albums: pd.DataFrame, 
                            components: List[str], component_labels: List[str],
                            output_dir: Path):
    """
    Bar chart showing 5 components for top 3 albums of an artist.
    
    Args:
        artist: Artist name
        top_albums: DataFrame with top albums
        components: List of component column names
        component_labels: List of component display names
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(component_labels))
    width = 0.25
    n_albums = len(top_albums)
    
    # Colors for albums
    album_colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:n_albums]
    
    for i, (_, album_row) in enumerate(top_albums.iterrows()):
        album_name = album_row['album']
        values = [album_row[comp] for comp in components]
        offset = (i - (n_albums - 1) / 2) * width
        
        ax.bar(x + offset, values, width, label=album_name, 
              color=album_colors[i], alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Component', fontweight='bold')
    ax.set_ylabel('Score (Balanced)', fontweight='bold')
    ax.set_title(f'{artist}: Top {n_albums} Albums - Component Breakdown', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(component_labels, rotation=45, ha='right')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save with artist name in filename
    safe_artist = artist.lower().replace(' ', '_')
    save_figure(fig, f'{safe_artist}_top3_component_bars', output_dir)


def plot_top3_radar(artist: str, top_albums: pd.DataFrame,
                   components: List[str], component_labels: List[str],
                   output_dir: Path):
    """
    Radar chart comparing top 3 albums across 5 components.
    
    Args:
        artist: Artist name
        top_albums: DataFrame with top albums
        components: List of component column names
        component_labels: List of component display names
        output_dir: Output directory
    """
    n_components = len(components)
    angles = [n / float(n_components) * 2 * pi for n in range(n_components)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    album_colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(top_albums)]
    
    for i, (_, album_row) in enumerate(top_albums.iterrows()):
        album_name = album_row['album']
        values = [album_row[comp] for comp in components]
        values += values[:1]  # Complete the circle
        
        color = album_colors[i]
        ax.plot(angles, values, 'o-', linewidth=2.5, label=album_name, 
               color=color, markersize=8)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(component_labels, fontsize=10)
    ax.set_ylim(0, 1)
    
    # More granular y-axis ticks
    num_ticks = 6
    y_ticks = np.linspace(0, 1, num_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.2f}' for tick in y_ticks], fontsize=9)
    ax.grid(True, alpha=0.5)
    
    ax.set_title(f'{artist}: Top {len(top_albums)} Albums Comparison (Radar Chart)', 
                 fontweight='bold', pad=25, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    
    safe_artist = artist.lower().replace(' ', '_')
    save_figure(fig, f'{safe_artist}_top3_radar', output_dir)


def generate_top3_text_summary(artist: str, top_albums: pd.DataFrame,
                               components: List[str], component_labels: List[str],
                               output_dir: Path):
    """
    Generate human-readable text summary of top albums.
    
    Args:
        artist: Artist name
        top_albums: DataFrame with top albums
        components: List of component column names
        component_labels: List of component display names
        output_dir: Output directory
    """
    safe_artist = artist.lower().replace(' ', '_')
    output_file = output_dir / f"{safe_artist}_top3_text.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{artist} - Top {len(top_albums)} Albums Analysis\n")
        f.write("=" * 60 + "\n\n")
        
        for rank, (_, album_row) in enumerate(top_albums.iterrows(), 1):
            album_name = album_row['album']
            composite_score = album_row['composite_score_balanced']
            
            f.write(f"Album {rank} – {album_name}\n")
            f.write(f"  Composite Score: {composite_score:.3f}\n")
            
            # Get component scores
            comp_scores = {}
            for comp, label in zip(components, component_labels):
                comp_scores[label] = album_row[comp]
            
            # Find top 2 components (strengths)
            sorted_components = sorted(comp_scores.items(), key=lambda x: x[1], reverse=True)
            top2 = sorted_components[:2]
            
            # Find lowest component (weakness)
            lowest = sorted_components[-1]
            
            f.write(f"  Strengths: {top2[0][0]} ({top2[0][1]:.3f}), {top2[1][0]} ({top2[1][1]:.3f})\n")
            f.write(f"  Weaknesses: {lowest[0]} ({lowest[1]:.3f})\n")
            f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("Note: Scores are balanced (60% mean, 40% max)\n")
    
    logger = logging.getLogger('album_viz')
    logger.info(f"Generated text summary for {artist}: {output_file}")


def ensure_numeric_values(data: List[Dict], numeric_keys: List[str]):
    """
    Ensure numeric keys are converted to float.
    
    Args:
        data: List of dictionaries
        numeric_keys: List of keys that should be numeric
    """
    for item in data:
        for key in numeric_keys:
            if key in item:
                try:
                    item[key] = float(item[key])
                except (ValueError, TypeError):
                    item[key] = 0.0


def generate_all_visualizations(song_scores: List[Dict], album_scores: List[Dict], 
                                artist_scores: List[Dict]):
    """
    Generate all visualizations and save to outputs/visualizations.
    
    Args:
        song_scores: List of song score dictionaries
        album_scores: List of album score dictionaries
        artist_scores: List of artist score dictionaries
    """
    # Ensure numeric values are properly typed
    song_numeric_keys = ['lexical_complexity', 'rhyme_complexity', 'semantic_depth',
                        'narrative_structure', 'emotional_expressiveness', 'composite_score']
    ensure_numeric_values(song_scores, song_numeric_keys)
    
    album_numeric_keys = [key for key in (album_scores[0].keys() if album_scores else []) 
                         if key not in ['artist', 'album', 'num_songs']]
    ensure_numeric_values(album_scores, album_numeric_keys)
    
    artist_numeric_keys = [key for key in (artist_scores[0].keys() if artist_scores else []) 
                          if key not in ['artist', 'num_songs']]
    ensure_numeric_values(artist_scores, artist_numeric_keys)
    
    output_dir = Path(OUTPUT_DIR) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    print("\nGenerating artist-level comparisons...")
    plot_artist_composite_comparison(artist_scores, output_dir)
    plot_mean_max_balanced_comparison(artist_scores, output_dir)
    
    print("Generating song-level distributions...")
    plot_song_distributions(song_scores, output_dir)
    plot_component_boxplots(song_scores, output_dir)
    plot_songs_per_artist_pie(song_scores, output_dir)
    
    print("Generating album-level insights...")
    plot_album_distributions(album_scores, output_dir)
    
    print("Generating heatmaps...")
    plot_component_correlation_heatmap(song_scores, output_dir)
    plot_artist_component_heatmap(artist_scores, output_dir)
    
    print("Generating radar chart...")
    plot_radar_chart(artist_scores, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("Files saved as PNG format.")
    
    # Generate album-level analytics (top-3 breakdowns)
    print("\n" + "=" * 60)
    print("GENERATING ALBUM-LEVEL ANALYTICS")
    print("=" * 60)
    generate_album_visualizations(album_scores, artist_scores)


def generate_album_visualizations(album_scores: List[Dict], artist_scores: List[Dict]):
    """
    Generate album-level analytics visualizations.
    Includes top-3 album breakdowns per artist.
    Filters out "Unknown" albums.
    
    Args:
        album_scores: List of album score dictionaries
        artist_scores: List of artist score dictionaries
    """
    # Filter out Unknown albums
    album_scores = filter_unknown_albums(album_scores)
    if not album_scores:
        print("\nNo valid albums found (all filtered as Unknown). Skipping album-level analytics.")
        return
    
    # Ensure numeric values
    album_numeric_keys = [key for key in (album_scores[0].keys() if album_scores else []) 
                         if key not in ['artist', 'album', 'num_songs']]
    ensure_numeric_values(album_scores, album_numeric_keys)
    
    output_dir = Path(OUTPUT_DIR) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating top-3 album breakdowns per artist...")
    generate_top3_album_breakdown(album_scores, output_dir)
    
    print(f"\nAlbum-level analytics complete!")
    print(f"Top-3 breakdowns saved to: {output_dir / 'album_top3'}")


if __name__ == "__main__":
    # For testing - load from CSV files
    import csv
    
    output_dir = Path(OUTPUT_DIR)
    
    # Load song scores
    song_scores = []
    song_file = output_dir / "song_level_scores.csv"
    if song_file.exists():
        with open(song_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            song_scores = list(reader)
    
    # Load album scores (use cleaned version if available)
    album_scores = []
    album_file_cleaned = output_dir / "album_level_scores_clean.csv"
    album_file = album_file_cleaned if album_file_cleaned.exists() else output_dir / "album_level_scores.csv"
    if album_file.exists():
        with open(album_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            album_scores = list(reader)
    
    # Load artist scores
    artist_scores = []
    artist_file = output_dir / "artist_level_scores.csv"
    if artist_file.exists():
        with open(artist_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            artist_scores = list(reader)
    
    if song_scores and album_scores and artist_scores:
        # Convert string values to float where needed
        for song in song_scores:
            for key in ['lexical_complexity', 'rhyme_complexity', 'semantic_depth',
                       'narrative_structure', 'emotional_expressiveness', 'composite_score']:
                if key in song:
                    song[key] = float(song[key])
        
        for album in album_scores:
            for key in album.keys():
                if key not in ['artist', 'album', 'num_songs']:
                    album[key] = float(album[key])
        
        for artist in artist_scores:
            for key in artist.keys():
                if key not in ['artist', 'num_songs']:
                    artist[key] = float(artist[key])
        
        generate_all_visualizations(song_scores, album_scores, artist_scores)
    else:
        print("Error: Could not load score files. Run aggregation first.")
