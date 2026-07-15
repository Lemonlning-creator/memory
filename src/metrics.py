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
]
