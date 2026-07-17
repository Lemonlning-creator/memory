"""
Evaluation Metrics Module

Centralized implementation of all evaluation metrics used across experiments:
  - Style Similarity (lexical + semantic)
  - Portrait Entropy (profile uncertainty)
  - Emotion / Sentiment Accuracy
  - Exploration Question Ratio
  - Profile Completeness
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from .epistemic_decay import (
    PROFILE_LAYERS,
    compute_portrait_entropy,
    compute_profile_completeness,
)


# ---------------------------------------------------------------------------
# Style Similarity
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split into tokens."""
    return text.lower().translate(_PUNCT_TABLE).split()


def compute_rouge_l(reference: str, candidate: str) -> float:
    """Compute ROUGE-L F1 score between reference and candidate.

    ROUGE-L measures the longest common subsequence (LCS) between two texts.
    Returns F1 score in [0, 1].
    """
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)

    if not ref_tokens or not cand_tokens:
        return 0.0

    # LCS via dynamic programming
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / n
    recall = lcs_len / m
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def compute_lexical_overlap(reference: str, candidate: str) -> float:
    """Compute token-level Jaccard similarity."""
    ref_set = set(_tokenize(reference))
    cand_set = set(_tokenize(candidate))
    if not ref_set and not cand_set:
        return 1.0
    if not ref_set or not cand_set:
        return 0.0
    intersection = ref_set & cand_set
    union = ref_set | cand_set
    return round(len(intersection) / len(union), 4)


def compute_style_similarity(reference: str, candidate: str) -> Dict[str, float]:
    """Compute multiple style similarity metrics.

    Returns:
        Dict with rouge_l, lexical_overlap, and a combined style_similarity score.
    """
    rouge = compute_rouge_l(reference, candidate)
    lexical = compute_lexical_overlap(reference, candidate)
    combined = round(0.6 * rouge + 0.4 * lexical, 4)
    return {
        "rouge_l": rouge,
        "lexical_overlap": lexical,
        "style_similarity": combined,
    }


# ---------------------------------------------------------------------------
# Exploration Question Detection
# ---------------------------------------------------------------------------

EXPLORATION_PATTERNS = [
    r"(?:how|what|why|when|where|who)\s+(?:do|did|does|are|is|was|were|can|could|would|will|should)\s+you",
    r"(?:tell\s+me|share|what|how)\s+(?:about|more)",
    r"(?:how\s+do\s+you\s+feel|how\s+are\s+you\s+feeling|how\s+was\s+your)",
    r"(?:what\s+(?:do|did)\s+you\s+(?:think|feel|want|need))",
    r"(?:can\s+you\s+(?:tell|share|explain))",
    r"(?:why\s+do\s+you\s+(?:think|feel|say))",
    r"(?:what\s+makes\s+you)",
    r"(?:is\s+that\s+why|is\s+that\s+because)",
    r"(?:do\s+you\s+(?:often|usually|tend))",
    r"(?:what\s+happened\s+(?:next|after))",
    r"(?:你(?:觉得|感觉|为什么|怎么|是不是))",
    r"(?:(?:能|可以)(?:告诉|分享|说说))",
    r"(?:(?:什么|为什么|怎么).*(?:让|使|导致))",
    r"(?:(?:你|您).*(?:感受|想法|看法|意见))",
]


def detect_exploration_question(text: str) -> bool:
    """Detect if a text contains exploration questions (probing for more info about the user).

    Uses regex patterns to identify questions that seek to learn more about
    the user's thoughts, feelings, experiences, or motivations.
    """
    text_lower = text.lower()
    for pattern in EXPLORATION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    # Also check for question marks combined with user-directed words
    if "?" in text or "？" in text:
        user_words = {"you", "your", "yourself", "你", "你的", "您"}
        words = set(_tokenize(text))
        if words & user_words:
            return True
    return False


def compute_exploration_ratio(responses: List[str]) -> float:
    """Compute the ratio of exploration questions in a list of responses.

    Returns:
        Ratio in [0, 1].
    """
    if not responses:
        return 0.0
    exploration_count = sum(1 for r in responses if detect_exploration_question(r))
    return round(exploration_count / len(responses), 4)


# ---------------------------------------------------------------------------
# Re-exports for convenience
# ---------------------------------------------------------------------------

__all__ = [
    "compute_rouge_l",
    "compute_lexical_overlap",
    "compute_style_similarity",
    "detect_exploration_question",
    "compute_exploration_ratio",
    "compute_portrait_entropy",
    "compute_profile_completeness",
    "compute_emotion_similarity",
    "match_emotion",
]


# ---------------------------------------------------------------------------
# Semantic Emotion Matching
# ---------------------------------------------------------------------------

# Plutchik's emotion wheel adjacency similarity.
# Each emotion maps to a position on the wheel; similarity decreases with distance.
# This captures that "joy" and "amusement" are similar, while "joy" and "anger" are not.
EMOTION_CATEGORIES = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
    "amusement", "guilt", "curiosity", "neutral",
]

# Pre-computed pairwise similarity (0.0 = opposite, 1.0 = identical)
# Based on Plutchik's wheel + common psychological grouping
_EMOTION_SIMILARITY = {
    # Joy cluster
    ("joy", "joy"): 1.0, ("joy", "amusement"): 0.85, ("joy", "trust"): 0.6,
    ("joy", "anticipation"): 0.5, ("joy", "curiosity"): 0.4,
    # Sadness cluster
    ("sadness", "sadness"): 1.0, ("sadness", "guilt"): 0.7, ("sadness", "fear"): 0.5,
    ("sadness", "disgust"): 0.4,
    # Anger cluster
    ("anger", "anger"): 1.0, ("anger", "disgust"): 0.7, ("anger", "sadness"): 0.4,
    # Fear cluster
    ("fear", "fear"): 1.0, ("fear", "surprise"): 0.6, ("fear", "sadness"): 0.5,
    # Surprise cluster
    ("surprise", "surprise"): 1.0, ("surprise", "anticipation"): 0.6,
    ("surprise", "fear"): 0.6,
    # Trust cluster
    ("trust", "trust"): 1.0, ("trust", "joy"): 0.6,
    # Anticipation cluster
    ("anticipation", "anticipation"): 1.0, ("anticipation", "curiosity"): 0.7,
    ("anticipation", "surprise"): 0.6,
    # Amusement cluster
    ("amusement", "amusement"): 1.0, ("amusement", "joy"): 0.85,
    # Guilt cluster
    ("guilt", "guilt"): 1.0, ("guilt", "sadness"): 0.7,
    # Curiosity cluster
    ("curiosity", "curiosity"): 1.0, ("curiosity", "anticipation"): 0.7,
    ("curiosity", "surprise"): 0.4,
    # Disgust
    ("disgust", "disgust"): 1.0, ("disgust", "anger"): 0.7,
    # Neutral
    ("neutral", "neutral"): 1.0,
}


def compute_emotion_similarity(emotion_a: str, emotion_b: str) -> float:
    """Compute semantic similarity between two emotion labels.

    Returns:
        Float in [0, 1]. 1.0 = identical, 0.0 = dissimilar/opposite.
    """
    a = emotion_a.lower().strip()
    b = emotion_b.lower().strip()

    if a == b:
        return 1.0

    # Check direct lookup
    sim = _EMOTION_SIMILARITY.get((a, b))
    if sim is not None:
        return sim
    sim = _EMOTION_SIMILARITY.get((b, a))
    if sim is not None:
        return sim

    # Handle common variants
    variant_map = {
        "happy": "joy", "excited": "joy", "excitement": "joy",
        "afraid": "fear", "anxious": "fear",
        "annoyed": "anger", "frustrated": "anger",
        "disappointed": "sadness", "depressed": "sadness",
        "shocked": "surprise", "astonished": "surprise",
        "interested": "curiosity", "intrigued": "curiosity",
        "looking forward": "anticipation", "hopeful": "anticipation",
        "remorseful": "guilt", "ashamed": "guilt",
        "content": "trust", "secure": "trust",
        "relieved": "trust",
        "neutral": "neutral",
    }
    a_mapped = variant_map.get(a, a)
    b_mapped = variant_map.get(b, b)
    if a_mapped == b_mapped:
        return 0.9

    sim = _EMOTION_SIMILARITY.get((a_mapped, b_mapped))
    if sim is not None:
        return sim
    sim = _EMOTION_SIMILARITY.get((b_mapped, a_mapped))
    if sim is not None:
        return sim

    # Fallback: check valence alignment (positive vs negative)
    positive_emotions = {"joy", "amusement", "trust", "anticipation", "curiosity"}
    negative_emotions = {"sadness", "anger", "fear", "disgust", "guilt"}
    if a in positive_emotions and b in positive_emotions:
        return 0.3
    if a in negative_emotions and b in negative_emotions:
        return 0.3

    return 0.0


def match_emotion(predicted: str, ground_truth: str, threshold: float = 0.5) -> float:
    """Compute emotion match score using semantic similarity.

    Args:
        predicted: Predicted emotion label.
        ground_truth: Ground truth emotion label.
        threshold: Minimum similarity to count as a match.

    Returns:
        1.0 if similarity >= threshold, else the similarity score.
    """
    sim = compute_emotion_similarity(predicted, ground_truth)
    return 1.0 if sim >= threshold else round(sim, 4)
