"""
Evaluation metrics for report generation quality.
Implements BLEU, ROUGE, and METEOR metrics.
"""

import numpy as np
from typing import List, Dict
from collections import Counter
import re


def tokenize(text: str) -> List[str]:
    """Simple tokenization for metric computation."""
    # Convert to lowercase and split on non-alphanumeric
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def compute_ngrams(tokens: List[str], n: int) -> Counter:
    """Compute n-grams from token list."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return Counter(ngrams)


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Compute BLEU-4 score.
    
    Args:
        reference: Ground truth text
        hypothesis: Generated text
        max_n: Maximum n-gram order (default 4 for BLEU-4)
    
    Returns:
        BLEU score (0-100)
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    
    if len(hyp_tokens) == 0:
        return 0.0
    
    # Compute precision for each n-gram order
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = compute_ngrams(ref_tokens, n)
        hyp_ngrams = compute_ngrams(hyp_tokens, n)
        
        if len(hyp_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        # Count matches
        matches = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = sum(hyp_ngrams.values())
        
        precision = matches / total if total > 0 else 0.0
        precisions.append(precision)
    
    # Geometric mean of precisions
    if 0.0 in precisions:
        return 0.0
    
    geo_mean = np.exp(np.mean(np.log(precisions)))
    
    # Brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
    
    bleu = bp * geo_mean * 100  # Scale to 0-100
    return bleu


def longest_common_subsequence(tokens1: List[str], tokens2: List[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(tokens1), len(tokens2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i-1] == tokens2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def rouge_l_score(reference: str, hypothesis: str) -> float:
    """
    Compute ROUGE-L score (F1-measure based on LCS).
    
    Args:
        reference: Ground truth text
        hypothesis: Generated text
    
    Returns:
        ROUGE-L score (0-1)
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0
    
    lcs_length = longest_common_subsequence(ref_tokens, hyp_tokens)
    
    # Precision and recall based on LCS
    precision = lcs_length / len(hyp_tokens) if len(hyp_tokens) > 0 else 0.0
    recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    # F1 score
    if precision + recall == 0:
        return 0.0
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def meteor_score(reference: str, hypothesis: str) -> float:
    """
    Simplified METEOR score (unigram F-mean with penalty).
    
    Args:
        reference: Ground truth text
        hypothesis: Generated text
    
    Returns:
        METEOR score (0-1)
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0
    
    # Count matching unigrams
    ref_counter = Counter(ref_tokens)
    hyp_counter = Counter(hyp_tokens)
    
    matches = sum(min(hyp_counter[token], ref_counter[token]) for token in hyp_counter)
    
    # Precision and recall
    precision = matches / len(hyp_tokens) if len(hyp_tokens) > 0 else 0.0
    recall = matches / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    # Harmonic mean (F1)
    f_mean = (10 * precision * recall) / (9 * precision + recall)
    
    # Simplified penalty (no chunk detection in this version)
    penalty = 0.5
    
    meteor = f_mean * (1 - penalty)
    return meteor


def compute_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Compute all metrics for a batch of reports.
    
    Args:
        references: List of ground truth reports
        hypotheses: List of generated reports
    
    Returns:
        Dictionary with average BLEU, ROUGE-L, METEOR scores
    """
    assert len(references) == len(hypotheses), "Must have same number of references and hypotheses"
    
    bleu_scores = []
    rouge_scores = []
    meteor_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        bleu_scores.append(bleu_score(ref, hyp))
        rouge_scores.append(rouge_l_score(ref, hyp))
        meteor_scores.append(meteor_score(ref, hyp))
    
    return {
        'bleu_4': np.mean(bleu_scores),
        'rouge_l': np.mean(rouge_scores),
        'meteor': np.mean(meteor_scores),
        'num_samples': len(references)
    }


def evaluate_report_quality(reference: str, generated: str) -> Dict[str, float]:
    """
    Evaluate a single report across all metrics.
    
    Args:
        reference: Ground truth report
        generated: Generated report
    
    Returns:
        Dictionary with individual metric scores
    """
    return {
        'bleu_4': bleu_score(reference, generated),
        'rouge_l': rouge_l_score(reference, generated),
        'meteor': meteor_score(reference, generated)
    }
