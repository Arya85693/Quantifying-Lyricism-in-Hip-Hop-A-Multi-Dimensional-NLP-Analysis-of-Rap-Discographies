"""
Main execution script for lyricism NLP analysis pipeline.

Runs ingestion, preprocessing, analysis, and aggregation in sequence.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ingestion.ingest_dave import ingest_dave
from ingestion.ingest_drake import ingest_drake
from ingestion.ingest_kanye_west import ingest_kanye_west
from ingestion.ingest_kendrick_lamar import ingest_kendrick_lamar
from ingestion.ingest_lil_wayne import ingest_lil_wayne
from preprocessing.clean_lyrics import process_all_artists
from aggregation.aggregate_scores import main as aggregate_main
from visualization.generate_plots import generate_all_visualizations


def run_ingestion():
    """Run all ingestion scripts."""
    print("=" * 60)
    print("STEP 1: INGESTION")
    print("=" * 60)
    
    print("\nIngesting Dave...")
    ingest_dave()
    
    print("\nIngesting Drake...")
    ingest_drake()
    
    print("\nIngesting Kanye West...")
    ingest_kanye_west()
    
    print("\nIngesting Kendrick Lamar...")
    ingest_kendrick_lamar()
    
    print("\nIngesting Lil Wayne...")
    ingest_lil_wayne()
    
    print("\nIngestion complete!")


def run_preprocessing():
    """Run preprocessing."""
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING")
    print("=" * 60)
    
    process_all_artists()
    print("Preprocessing complete!")


def run_analysis():
    """Run all analysis modules."""
    print("\n" + "=" * 60)
    print("STEP 3: ANALYSIS")
    print("=" * 60)
    
    print("Analysis modules will be executed during aggregation...")
    print("Analysis complete!")


def run_aggregation():
    """Run aggregation."""
    print("\n" + "=" * 60)
    print("STEP 4: AGGREGATION")
    print("=" * 60)
    
    song_scores, album_scores, artist_scores = aggregate_main()
    
    return song_scores, album_scores, artist_scores


def print_summary(artist_scores):
    """Print ranked summary of artists."""
    print("\n" + "=" * 60)
    print("FINAL RANKINGS")
    print("=" * 60)
    
    # Sort by balanced composite score
    sorted_artists = sorted(
        artist_scores,
        key=lambda x: x.get("composite_score_balanced", 0.0),
        reverse=True
    )
    
    print("\nRank | Artist                | Composite Score | Songs")
    print("-" * 60)
    
    for rank, artist_data in enumerate(sorted_artists, 1):
        artist = artist_data["artist"]
        score = artist_data.get("composite_score_balanced", 0.0)
        num_songs = artist_data.get("num_songs", 0)
        
        print(f"{rank:4d} | {artist:20s} | {score:15.4f} | {num_songs:5d}")
    
    print("\n" + "=" * 60)
    print("Component Breakdown:")
    print("=" * 60)
    
    for artist_data in sorted_artists:
        artist = artist_data["artist"]
        print(f"\n{artist}:")
        print(f"  Lexical Complexity:      {artist_data.get('lexical_complexity_balanced', 0.0):.4f}")
        print(f"  Rhyme Complexity:        {artist_data.get('rhyme_complexity_balanced', 0.0):.4f}")
        print(f"  Semantic Depth:           {artist_data.get('semantic_depth_balanced', 0.0):.4f}")
        print(f"  Narrative Structure:      {artist_data.get('narrative_structure_balanced', 0.0):.4f}")
        print(f"  Emotional Expressiveness: {artist_data.get('emotional_expressiveness_balanced', 0.0):.4f}")
        print(f"  Composite Score:          {artist_data.get('composite_score_balanced', 0.0):.4f}")


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("LYRICISM NLP ANALYSIS PIPELINE")
    print("=" * 60)
    print("\nThis pipeline will:")
    print("1. Ingest lyrics from Genius API")
    print("2. Preprocess and clean lyrics")
    print("3. Analyze five dimensions of lyricism")
    print("4. Aggregate scores and generate rankings")
    print("\n" + "=" * 60 + "\n")
    
    try:
        # Step 1: Ingestion
        run_ingestion()
        
        # Step 2: Preprocessing
        run_preprocessing()
        
        # Step 3: Analysis (happens during aggregation)
        run_analysis()
        
        # Step 4: Aggregation
        song_scores, album_scores, artist_scores = run_aggregation()
        
        # Step 5: Generate visualizations
        try:
            generate_all_visualizations(song_scores, album_scores, artist_scores)
        except Exception as e:
            print(f"\nWarning: Visualization generation failed: {e}")
            print("CSV outputs are still available.")
            import traceback
            traceback.print_exc()
        
        # Print summary
        print_summary(artist_scores)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print("\nResults saved to outputs/ directory:")
        print("  - song_level_scores.csv")
        print("  - album_level_scores.csv")
        print("  - artist_level_scores.csv")
        print("  - visualizations/ (PNG and PDF files)")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
