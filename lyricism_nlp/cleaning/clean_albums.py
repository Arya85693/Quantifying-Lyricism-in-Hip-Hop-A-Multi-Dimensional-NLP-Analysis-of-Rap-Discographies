"""
Script to clean album_level_scores.csv by filtering to albums with sufficient songs.

Filters albums based on song count threshold (default: >= 6 songs).
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import OUTPUT_DIR

# Song count threshold
MIN_SONGS_PER_ALBUM = 6


def clean_album_scores_by_count(input_file: Path, output_file: Path, 
                                audit_file: Path, min_songs: int = MIN_SONGS_PER_ALBUM) -> Tuple[pd.DataFrame, int]:
    """
    Clean album-level scores CSV by filtering to albums with sufficient songs.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output cleaned CSV file
        audit_file: Path to audit file for removed albums
        min_songs: Minimum number of songs required (default: 6)
        
    Returns:
        Tuple of (cleaned DataFrame, original count)
    """
    # Load CSV
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    df = pd.read_csv(input_file)
    original_count = len(df)
    logging.info(f"Loaded {original_count} album entries from {input_file}")
    
    # Get unique artists
    artists = df['artist'].unique()
    logging.info(f"Found {len(artists)} artists: {', '.join(artists)}")
    
    # Filter by song count
    mask = df['num_songs'] >= min_songs
    kept_df = df[mask].copy()
    removed_df = df[~mask].copy()
    
    # Save audit file with removed albums
    if len(removed_df) > 0:
        audit_df = removed_df[['artist', 'album', 'num_songs']].copy()
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(audit_file, index=False)
        logging.info(f"Saved {len(removed_df)} removed albums to {audit_file}")
    
    # Backup original file
    backup_file = input_file.with_suffix('.csv.backup')
    if input_file.exists() and not backup_file.exists():
        import shutil
        shutil.copy2(input_file, backup_file)
        logging.info(f"Backed up original file to: {backup_file}")
    
    # Save cleaned CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    kept_df.to_csv(output_file, index=False)
    logging.info(f"Saved {len(kept_df)} kept albums to {output_file}")
    
    # Also overwrite the original file used by visualizations (as per requirement 8)
    kept_df.to_csv(input_file, index=False)
    logging.info(f"Overwrote {input_file} with cleaned data (original backed up)")
    
    # Generate summary report
    print_summary_report(df, kept_df, removed_df, min_songs)
    
    return kept_df, original_count


def print_summary_report(original_df: pd.DataFrame, kept_df: pd.DataFrame, 
                         removed_df: pd.DataFrame, min_songs: int):
    """
    Print summary report for each artist.
    
    Args:
        original_df: Original DataFrame
        kept_df: DataFrame with kept albums
        removed_df: DataFrame with removed albums
        min_songs: Minimum songs threshold
    """
    logging.info(f"\n{'='*60}")
    logging.info("CLEANING SUMMARY REPORT")
    logging.info(f"{'='*60}")
    logging.info(f"Threshold: Albums with >= {min_songs} songs are kept")
    logging.info(f"{'='*60}\n")
    
    artists = sorted(original_df['artist'].unique())
    
    for artist in artists:
        artist_original = original_df[original_df['artist'] == artist]
        artist_kept = kept_df[kept_df['artist'] == artist]
        artist_removed = removed_df[removed_df['artist'] == artist]
        
        total_before = len(artist_original)
        albums_kept = len(artist_kept)
        albums_removed = len(artist_removed)
        
        # Find smallest kept album size
        if len(artist_kept) > 0:
            smallest_kept = artist_kept['num_songs'].min()
        else:
            smallest_kept = "N/A (no albums kept)"
        
        logging.info(f"{artist}:")
        logging.info(f"  Total albums before: {total_before}")
        logging.info(f"  Albums kept: {albums_kept}")
        logging.info(f"  Albums removed: {albums_removed}")
        logging.info(f"  Smallest kept album size: {smallest_kept}")
        
        # List removed albums
        if len(artist_removed) > 0:
            logging.info(f"  Removed albums:")
            for _, row in artist_removed.iterrows():
                logging.info(f"    - {row['album']} ({row['num_songs']} songs)")
        logging.info("")
    
    # Overall summary
    logging.info(f"{'='*60}")
    logging.info("OVERALL SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"Original albums: {len(original_df)}")
    logging.info(f"Albums kept: {len(kept_df)}")
    logging.info(f"Albums removed: {len(removed_df)}")
    logging.info(f"{'='*60}")


def main():
    """Main execution function."""
    # Setup logging
    log_file = Path(OUTPUT_DIR) / "album_cleaning.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # File paths
    input_file = Path(OUTPUT_DIR) / "album_level_scores.csv"
    output_file = Path(OUTPUT_DIR) / "album_level_scores_clean.csv"
    audit_file = Path(OUTPUT_DIR) / "albums_removed_due_to_size.csv"
    
    print(f"\n{'='*60}")
    print("ALBUM CLEANING SCRIPT (Song Count Threshold)")
    print(f"{'='*60}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Audit file: {audit_file}")
    print(f"Log file: {log_file}")
    print(f"Threshold: Albums with >= {MIN_SONGS_PER_ALBUM} songs")
    print(f"{'='*60}\n")
    
    try:
        # Clean the CSV
        cleaned_df, original_count = clean_album_scores_by_count(
            input_file, 
            output_file, 
            audit_file,
            min_songs=MIN_SONGS_PER_ALBUM
        )
        
        removed_count = original_count - len(cleaned_df)
        
        print(f"\n{'='*60}")
        print("CLEANING COMPLETE")
        print(f"{'='*60}")
        print(f"Cleaned CSV saved to: {output_file}")
        print(f"Audit file saved to: {audit_file}")
        print(f"Log file saved to: {log_file}")
        print(f"Original albums: {original_count}")
        print(f"Cleaned albums: {len(cleaned_df)}")
        print(f"Removed albums: {removed_count}")
        
    except Exception as e:
        logging.error(f"Error during cleaning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
