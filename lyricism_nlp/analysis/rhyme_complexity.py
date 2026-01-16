"""
Rhyme & Phonetic Complexity Analysis Module.

Measures phonetic technique:
- Rhyme density (end rhymes, internal rhymes)
- Multisyllabic rhyme chains
- Phonetic pattern variation
Emphasizes internal and overlapping rhymes over simple end rhymes.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import PROCESSED_DATA_DIR

try:
    import pronouncing
    PHONETIC_AVAILABLE = True
except ImportError:
    PHONETIC_AVAILABLE = False
    print("Warning: pronouncing library not available. Falling back to orthographic rhyme detection.")


def get_phonemes(word: str) -> Optional[List[str]]:
    """
    Get phonemes for a word using CMU Pronouncing Dictionary.
    
    Args:
        word: Word to get phonemes for
        
    Returns:
        List of phonemes or None if word not found
    """
    if not PHONETIC_AVAILABLE:
        return None
    
    word = word.lower().strip()
    # Remove punctuation that might interfere
    word = re.sub(r"[^a-z']", "", word)
    
    if not word:
        return None
    
    pronunciations = pronouncing.phones_for_word(word)
    if pronunciations:
        # Use first pronunciation (most common)
        return pronunciations[0].split()
    return None


def count_syllables_from_phonemes(phonemes: List[str]) -> int:
    """
    Count syllables from phoneme sequence.
    Syllables are indicated by phonemes ending in digits (0, 1, 2).
    
    Args:
        phonemes: List of phonemes
        
    Returns:
        Number of syllables
    """
    if not phonemes:
        return 0
    
    syllable_count = 0
    for phoneme in phonemes:
        # Phonemes ending in 0, 1, or 2 indicate vowels (0=unstressed, 1=primary stress, 2=secondary stress)
        if phoneme and phoneme[-1].isdigit():
            syllable_count += 1
    
    return syllable_count if syllable_count > 0 else 1  # Minimum 1 syllable


def extract_rhyming_phoneme_suffix(phonemes: List[str]) -> Optional[str]:
    """
    Extract rhyming phoneme suffix: from last stressed vowel to end.
    The last stressed vowel is the phoneme with the highest stress number
    (2=secondary, 1=primary) closest to the end, or any vowel if no stress marked.
    
    Args:
        phonemes: List of phonemes
        
    Returns:
        Rhyming suffix as string of phonemes, or None
    """
    if not phonemes:
        return None
    
    # Find the last vowel (phoneme ending in digit) and extract from there
    last_vowel_idx = -1
    for i in range(len(phonemes) - 1, -1, -1):
        if phonemes[i] and phonemes[i][-1].isdigit():
            last_vowel_idx = i
            break
    
    # If no vowel found, use last phoneme
    if last_vowel_idx == -1:
        last_vowel_idx = len(phonemes) - 1
    
    # Extract from last vowel to end
    rhyming_suffix = " ".join(phonemes[last_vowel_idx:])
    return rhyming_suffix


def extract_rhyming_sound(word: str) -> str:
    """
    Extract the rhyming sound from a word using phonetic analysis.
    Uses CMU Pronouncing Dictionary to extract rhyming phoneme suffix.
    Falls back to orthographic method if word not found in dictionary.
    
    Args:
        word: Word to extract rhyme from
        
    Returns:
        Rhyming sound pattern (phonetic suffix or orthographic fallback)
    """
    word = word.lower().strip()
    if not word:
        return ""
    
    # Try phonetic method first
    phonemes = get_phonemes(word)
    if phonemes:
        rhyming_suffix = extract_rhyming_phoneme_suffix(phonemes)
        if rhyming_suffix:
            return rhyming_suffix
    
    # Fallback to orthographic method if word not in CMU dictionary
    # This handles proper nouns, slang, or words not in the dictionary
    if len(word) <= 2:
        return word
    
    # Use last 2-3 characters as rhyme pattern (orthographic fallback)
    if len(word) >= 3:
        return word[-3:]
    return word[-2:]


def find_end_rhymes(lines: List[List[str]]) -> int:
    """
    Count end rhymes (words at end of lines that rhyme).
    
    Args:
        lines: List of lines, each line is a list of tokens
        
    Returns:
        Number of end rhyme pairs
    """
    if len(lines) < 2:
        return 0
    
    end_words = [line[-1].lower() if line else "" for line in lines]
    end_words = [w for w in end_words if w]
    
    if len(end_words) < 2:
        return 0
    
    rhyme_groups = defaultdict(list)
    for i, word in enumerate(end_words):
        rhyme_sound = extract_rhyming_sound(word)
        rhyme_groups[rhyme_sound].append(i)
    
    # Count pairs (each group of n words creates n*(n-1)/2 pairs)
    total_pairs = 0
    for group in rhyme_groups.values():
        if len(group) > 1:
            total_pairs += len(group) * (len(group) - 1) // 2
    
    return total_pairs


def find_internal_rhymes(line: List[str]) -> int:
    """
    Find internal rhymes within a single line.
    
    Args:
        line: List of tokens in a line
        
    Returns:
        Number of internal rhyme pairs
    """
    if len(line) < 2:
        return 0
    
    rhyme_groups = defaultdict(list)
    for i, word in enumerate(line):
        rhyme_sound = extract_rhyming_sound(word.lower())
        rhyme_groups[rhyme_sound].append(i)
    
    # Count pairs within the line
    total_pairs = 0
    for group in rhyme_groups.values():
        if len(group) > 1:
            total_pairs += len(group) * (len(group) - 1) // 2
    
    return total_pairs


def find_multisyllabic_rhymes(lines: List[List[str]], min_syllables: int = 2) -> int:
    """
    Find multisyllabic rhyme chains (consecutive words that rhyme).
    Uses actual syllable counts from phonemes when available.
    
    Args:
        lines: List of lines
        min_syllables: Minimum syllables to consider multisyllabic
        
    Returns:
        Number of multisyllabic rhyme chains
    """
    total_chains = 0
    
    for line in lines:
        if len(line) < 2:
            continue
        
        # Look for consecutive rhyming words
        i = 0
        while i < len(line) - 1:
            word1 = line[i].lower()
            word2 = line[i + 1].lower()
            
            # Check if they rhyme (phonetic or orthographic)
            rhyme1 = extract_rhyming_sound(word1)
            rhyme2 = extract_rhyming_sound(word2)
            
            if rhyme1 == rhyme2:
                # Check if multisyllabic using actual syllable count
                phonemes1 = get_phonemes(word1)
                phonemes2 = get_phonemes(word2)
                
                # Use phonetic syllable count if available, otherwise estimate from rhyme pattern
                if phonemes1 and phonemes2:
                    syllables1 = count_syllables_from_phonemes(phonemes1)
                    syllables2 = count_syllables_from_phonemes(phonemes2)
                    # Use the minimum to be conservative
                    syllable_count = min(syllables1, syllables2)
                else:
                    # Fallback: estimate from rhyme pattern length (rough heuristic)
                    # Phoneme strings are space-separated, so count spaces + 1
                    syllable_count = rhyme1.count(" ") + 1 if " " in rhyme1 else 1
                
                if syllable_count >= min_syllables:
                    # Found a chain, count it
                    chain_length = 2
                    j = i + 2
                    while j < len(line):
                        word3 = line[j].lower()
                        rhyme3 = extract_rhyming_sound(word3)
                        if rhyme3 == rhyme1:
                            chain_length += 1
                            j += 1
                        else:
                            break
                    
                    # Reward longer chains more
                    total_chains += chain_length * (chain_length - 1) // 2
                    i = j
                else:
                    i += 1
            else:
                i += 1
    
    return total_chains


def calculate_rhyme_density(lines: List[List[str]]) -> float:
    """
    Calculate overall rhyme density.
    
    Args:
        lines: List of lines
        
    Returns:
        Rhyme density score (0-1)
    """
    if not lines:
        return 0.0
    
    total_words = sum(len(line) for line in lines)
    if total_words == 0:
        return 0.0
    
    # Count different types of rhymes
    end_rhymes = find_end_rhymes(lines)
    internal_rhymes = sum(find_internal_rhymes(line) for line in lines)
    multisyllabic_rhymes = find_multisyllabic_rhymes(lines)
    
    # Weight internal rhymes more heavily
    total_rhyme_score = (
        1.0 * end_rhymes +
        2.0 * internal_rhymes +
        3.0 * multisyllabic_rhymes
    )
    
    # Normalize by total words
    density = total_rhyme_score / total_words if total_words > 0 else 0.0
    
    # Cap at reasonable maximum
    return min(density, 1.0)


def calculate_phonetic_variation(lines: List[List[str]]) -> float:
    """
    Calculate phonetic pattern variation.
    
    Args:
        lines: List of lines
        
    Returns:
        Phonetic variation score (0-1)
    """
    if not lines:
        return 0.0
    
    all_rhyme_patterns = []
    for line in lines:
        for word in line:
            pattern = extract_rhyming_sound(word.lower())
            all_rhyme_patterns.append(pattern)
    
    if not all_rhyme_patterns:
        return 0.0
    
    # Calculate diversity (unique patterns / total)
    unique_patterns = len(set(all_rhyme_patterns))
    total_patterns = len(all_rhyme_patterns)
    
    diversity = unique_patterns / total_patterns if total_patterns > 0 else 0.0
    
    return diversity


def analyze_rhyme_complexity(song_data: Dict) -> float:
    """
    Compute rhyme complexity score for a song.
    
    Args:
        song_data: Processed song data dictionary
        
    Returns:
        Rhyme complexity score (0-1)
    """
    lines = song_data.get("lyrics_lines", [])
    
    if not lines:
        return 0.0
    
    # Calculate component metrics
    rhyme_density = calculate_rhyme_density(lines)
    phonetic_variation = calculate_phonetic_variation(lines)
    
    # Weighted combination (emphasize density)
    score = (
        0.70 * rhyme_density +
        0.30 * phonetic_variation
    )
    
    return min(max(score, 0.0), 1.0)


def process_all_songs() -> Dict[Tuple[str, str], float]:
    """
    Process all songs and return rhyme complexity scores.
    
    Returns:
        Dictionary mapping (artist, song_title) to score
    """
    processed_file = Path(PROCESSED_DATA_DIR) / "all_processed_lyrics.jsonl"
    
    if not processed_file.exists():
        print(f"Processed lyrics file not found: {processed_file}")
        return {}
    
    scores = {}
    
    with open(processed_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    song_data = json.loads(line)
                    score = analyze_rhyme_complexity(song_data)
                    
                    key = (song_data.get("artist"), song_data.get("song_title"))
                    scores[key] = score
                except Exception as e:
                    print(f"Error processing song: {e}")
                    continue
    
    return scores


if __name__ == "__main__":
    scores = process_all_songs()
    print(f"Computed rhyme complexity scores for {len(scores)} songs")
    for (artist, song), score in list(scores.items())[:5]:
        print(f"{artist} - {song}: {score:.4f}")
