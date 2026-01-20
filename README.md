Quantifying Lyricism in Hip-Hop

A Multi-Dimensional NLP Analysis of Rap Discographies
Overview
This project explores whether aspects of “lyricism” in hip-hop can be quantified using natural language processing techniques. I built a Python-based analysis pipeline that processes song lyrics across multiple artists and extracts linguistic features such as vocabulary richness, repetition, and structural complexity. The goal was not to rank artists definitively, but to experiment with how different NLP metrics behave at a discography level and to understand their strengths and limitations.
This project was developed as a learning exercise in text preprocessing, feature extraction, and exploratory data analysis using real-world, unstructured text data.

Motivation
Hip-hop discussions often involve subjective claims about lyrical ability. I was curious whether commonly discussed qualities — such as word diversity or repetition — could be approximated using quantitative methods. At the same time, I wanted hands-on experience working with:
Messy text data
NLP libraries
End-to-end data analysis workflows

Data
Source: Publicly available song lyrics scraped from online lyric databases
Scope: Multiple artists with full or near-full discographies
Granularity: Song-level data aggregated to artist-level summaries
Lyrics were cleaned to remove annotations, punctuation noise, and formatting artifacts before analysis.

Methodology
1. Text Preprocessing
Lowercasing and tokenization
Removal of punctuation and non-lyrical artifacts
Stopword handling (with experimentation on inclusion vs exclusion)
This step required iteration, as small preprocessing choices significantly affected downstream metrics.

2. Feature Extraction
For each song and artist, I computed several interpretable NLP metrics, including:
Lexical diversity (e.g., type-token ratios)
Word frequency distributions
Repetition indicators
Average line and song length metrics
Metrics were chosen for interpretability rather than model performance, since the goal was analysis rather than prediction.

3. Aggregation & Comparison
Song-level metrics were aggregated to artist-level statistics
Results were compared across artists to observe trends and contrasts
Visualizations were used to sanity-check and interpret outputs

Results & Observations
Different metrics often emphasize different aspects of lyrical style
Some commonly cited measures of “lyricism” are highly sensitive to preprocessing choices
No single metric meaningfully captures lyrical complexity on its own
These results reinforced the idea that quantitative analysis can complement, but not replace, subjective interpretation in creative domains.

Limitations
Lyrics are treated purely as text, ignoring delivery, cadence, and production
Metrics are sensitive to data quality and preprocessing assumptions
The analysis does not account for temporal changes within an artist’s career

Future Improvements
If I were to extend this project, I would:
Store processed lyrics and metrics in a SQL database for easier querying
Add unit tests for preprocessing and feature extraction steps
Explore time-based analysis across album releases
Compare traditional NLP metrics with embedding-based representations

Tech Stack
Language: Python
Libraries: pandas, nltk / spaCy, matplotlib / seaborn
Workflow: Data ingestion → preprocessing → feature extraction → aggregation → visualization

What I Learned
How fragile NLP pipelines can be without careful preprocessing
How to reason about metric validity rather than blindly trusting outputs
How to structure an exploratory analysis project from raw data to insights
