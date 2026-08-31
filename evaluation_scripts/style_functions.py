#!/usr/bin/env python3
"""
Style Metrics and Utility Functions

This module contains all shared functions for style evaluation including:
- Readability metrics (Gunning Fog Index)
- Syntactic complexity metrics (Parse Tree Depth)
- Lexical diversity metrics (TTR, Yule's K)
- Structural similarity metrics (EMD on POS tags)
- Text processing utilities (tokenization, truncation)
- Filename parsing utilities
"""

import os
import re
import numpy as np
import nltk
from collections import Counter
from scipy.stats import wasserstein_distance
from transformers import AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from our_utils import MODEL_PATHS


# ==============================================================================
# TEXT PREPROCESSING
# ==============================================================================

def clean_delimiter_artifacts(text):
    """
    Replace delimiter artifacts like '//' with periods.

    Args:
        text: Input text that may contain delimiter artifacts

    Returns:
        Cleaned text with delimiters replaced
    """
    return re.sub(r'//', '.', text)


# ==============================================================================
# READABILITY METRICS
# ==============================================================================

def flesch_reading_ease(text):
    """
    Calculate Flesch Reading Ease score.

    Formula: 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
    Higher scores = easier to read
    Typical scores:
        90-100: Very easy (5th grade)
        60-70: Standard (8th-9th grade)
        0-30: Very difficult (college graduate)

    Args:
        text: Input text to analyze

    Returns:
        Flesch Reading Ease score
    """
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return 0

    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalnum()]

    if not words:
        return 0

    def count_syllables(word):
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        if word.endswith('e'):
            syllable_count -= 1

        if syllable_count == 0:
            syllable_count = 1

        return syllable_count

    total_syllables = sum(count_syllables(word) for word in words)
    total_words = len(words)
    total_sentences = len(sentences)

    flesch_score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)

    return flesch_score


def gunning_fog_index(text):
    """
    Calculate Gunning Fog index for readability.

    Formula: 0.4 * ((words/sentences) + 100 * (complex_words/words))
    Complex words = words with 3+ syllables
    Lower values = more readable (easier to understand)

    Args:
        text: Input text to analyze

    Returns:
        Gunning Fog Index score
    """
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return 0

    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalnum()]

    if not words:
        return 0

    def count_syllables(word):
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        if word.endswith('e'):
            syllable_count -= 1

        if syllable_count == 0:
            syllable_count = 1

        return syllable_count

    complex_words = sum(1 for word in words if count_syllables(word) >= 3)
    avg_sentence_length = len(words) / len(sentences)
    percentage_complex = 100 * (complex_words / len(words))
    fog_index = 0.4 * (avg_sentence_length + percentage_complex)

    return fog_index


# ==============================================================================
# SYNTACTIC COMPLEXITY METRICS
# ==============================================================================

def parse_tree_depth(text):
    """
    Calculate average parse tree depth using POS-tag heuristic.

    Higher values indicate more complex syntactic structures.

    Args:
        text: Input text to analyze

    Returns:
        Average parse tree depth across sentences
    """
    sentences = nltk.sent_tokenize(text)
    depths = []

    for sent in sentences:
        tokens = nltk.word_tokenize(sent)
        pos_tags = nltk.pos_tag(tokens)

        depth = 1
        clause_markers = ['IN', 'WDT', 'WP', 'WP$', 'WRB']
        for _, tag in pos_tags:
            if tag in clause_markers:
                depth += 1

        depth += sum(1 for _, tag in pos_tags if tag == ',') * 0.5
        depths.append(depth)

    return np.mean(depths) if depths else 0


# ==============================================================================
# LEXICAL DIVERSITY METRICS
# ==============================================================================

def type_token_ratio(text):
    """
    Calculate Type-Token Ratio (TTR) for lexical diversity.

    TTR = number of unique words / total number of words
    Higher values indicate greater vocabulary diversity.
    Range: 0 to 1, where 1 means all words are unique.

    Args:
        text: Input text to analyze

    Returns:
        Type-Token Ratio
    """
    words = nltk.word_tokenize(text.lower())
    words = [w for w in words if w.isalnum()]

    if not words:
        return 0

    unique_words = len(set(words))
    total_words = len(words)

    return unique_words / total_words


def yules_k(text):
    """
    Calculate Yule's K statistic for lexical diversity.

    Yule's K is less sensitive to text length than TTR.
    Lower values indicate greater lexical diversity.

    Formula: K = 10^4 * (M2 - M1) / (M1^2)
    where M1 = number of tokens, M2 = sum of (freq^2 * number of types with that freq)

    Args:
        text: Input text to analyze

    Returns:
        Yule's K statistic
    """
    words = nltk.word_tokenize(text.lower())
    words = [w for w in words if w.isalnum()]

    if len(words) < 2:
        return 0

    word_freq = Counter(words)
    M1 = len(words)
    freq_freq = Counter(word_freq.values())
    M2 = sum(freq**2 * count for freq, count in freq_freq.items())

    if M1 == 0:
        return 0

    K = 10000 * (M2 - M1) / (M1 ** 2)

    return K


# ==============================================================================
# STRUCTURAL SIMILARITY METRICS
# ==============================================================================

def emd_pos(hypothesis, reference):
    """
    Calculate Earth Mover's Distance between PoS tag distributions.

    Lower values indicate more similar syntactic structure.

    Args:
        hypothesis: First text for comparison
        reference: Second text for comparison

    Returns:
        Wasserstein distance between POS tag distributions
    """
    hyp_tokens = nltk.word_tokenize(hypothesis)
    ref_tokens = nltk.word_tokenize(reference)

    hyp_pos = nltk.pos_tag(hyp_tokens)
    ref_pos = nltk.pos_tag(ref_tokens)

    hyp_tags = [tag for _, tag in hyp_pos]
    ref_tags = [tag for _, tag in ref_pos]

    all_tags = sorted(set(hyp_tags + ref_tags))

    hyp_counts = Counter(hyp_tags)
    ref_counts = Counter(ref_tags)

    hyp_total = len(hyp_tags)
    ref_total = len(ref_tags)

    hyp_probs = np.array([hyp_counts.get(tag, 0) / hyp_total for tag in all_tags])
    ref_probs = np.array([ref_counts.get(tag, 0) / ref_total for tag in all_tags])

    positions = np.arange(len(all_tags))

    return wasserstein_distance(positions, positions, hyp_probs, ref_probs)


def pos_tag_frequency(text):
    """
    Calculate the frequency of each POS tag in the given text.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary mapping POS tags to their frequency counts
    """
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    tags = [tag for _, tag in pos_tags]
    return dict(Counter(tags))


# ==============================================================================
# TOKENIZATION AND TEXT PROCESSING
# ==============================================================================

def load_tokenizer(model_name):
    """
    Load the appropriate tokenizer for the given model.

    Args:
        model_name: Name of the model (must exist in MODEL_PATHS)

    Returns:
        Hugging Face tokenizer instance
    """
    model_path = MODEL_PATHS.get(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def truncate_to_n_tokens(text, tokenizer, max_tokens=200):
    """
    Truncate text to a maximum number of tokens.

    Args:
        text: Input text
        tokenizer: Hugging Face tokenizer
        max_tokens: Maximum number of tokens

    Returns:
        Truncated text
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


# ==============================================================================
# FILENAME PARSING
# ==============================================================================

def parse_filename(filepath):
    """
    Parse the filename to extract scheme, task, and model.

    Examples:
        KGW-g0.5-d2-llama3-ehr_procedures.pkl.json -> (KGW-g0.5-d2, ehr_procedures, llama3)
        SIR-llama3-medal_llama3.pkl.json -> (SIR, medal_llama3, llama3)

    Args:
        filepath: Path to the file

    Returns:
        Tuple of (scheme, task, model)
    """
    filename = os.path.basename(filepath)
    filename = filename.replace('.pkl.json', '')

    parts = filename.split('-')

    model = None
    model_idx = -1
    for i, part in enumerate(parts):
        if part in MODEL_PATHS.keys():
            model = part
            model_idx = i
            break

    if model is None:
        raise ValueError(f"Could not identify model in filename: {filepath}")

    scheme = '-'.join(parts[:model_idx])
    task = '-'.join(parts[model_idx + 1:])

    return scheme, task, model


# Fixed min/max ranges per task for HELP score normalization.
# Computed from watermark_samples_6.csv (W+U combined per task).
HELP_METRIC_RANGES = {
    'ehr_nursing_note': {
        'ACC': (0.055556, 0.723306),   # METEOR
        'EMD': (0.214286, 2.462155),   # EMD vs Natural
        'READ': (0.200000, 13.607692), # GFI
    },
    'ehr_progress_note': {
        'ACC': (0.128952, 0.720867),
        'EMD': (0.205369, 2.842774),
        'READ': (4.230769, 14.751515),
    },
    'all': {
        'ACC': (0.055556, 0.723306),
        'EMD': (0.205369, 2.842774),
        'READ': (0.200000, 14.751515),
    },
}

# Linear regression coefficients.
# Fitted with leave-one-task-out cross validation on human ratings: the human Helpfulness
# score is regressed on the human Accuracy, Syntactic and Readability scores, all min-max
# normalized to [0,1]. The coefficients are then transferred to the automatic features
# ACC = METEOR, EMD = EMD vs natural text, READ = Gunning Fog Index, normalized with the ranges above.
HELP_INTERCEPT = -0.078
HELP_COEF_ACC = 0.574
HELP_COEF_EMD = 0.215
HELP_COEF_READ = 0.282


def calculate_help_score(acc, emd, read, task='all'):
    """
    Calculate HELP score using linear regression with normalized metrics.

    Uses fixed per-task min/max ranges for normalization to [0,1],
    matching the min-max normalization used in the regression training.

    Args:
        acc: Accuracy metric value(s) (e.g., METEOR). Scalar or array.
        emd: EMD value(s). Scalar or array.
        read: Readability metric value(s) (e.g., GFI). Scalar or array.
        task: Task name for selecting normalization ranges.
              One of 'ehr_nursing_note', 'ehr_progress_note', 'all'.

    Returns:
        HELP score (scalar or array, matching input shape)
    """
    ranges = HELP_METRIC_RANGES[task]

    acc_min, acc_max = ranges['ACC']
    emd_min, emd_max = ranges['EMD']
    read_min, read_max = ranges['READ']

    acc_norm = (np.asarray(acc) - acc_min) / (acc_max - acc_min)
    emd_norm = (np.asarray(emd) - emd_min) / (emd_max - emd_min)
    read_norm = (np.asarray(read) - read_min) / (read_max - read_min)

    help_score = HELP_INTERCEPT + HELP_COEF_ACC * acc_norm + HELP_COEF_EMD * emd_norm + HELP_COEF_READ * read_norm

    return help_score