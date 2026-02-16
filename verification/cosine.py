"""Cosine similarity verification (Layer 1)."""
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)


def interpret_similarity(score: float) -> str:
    """Convert similarity score to verdict."""
    if score >= 0.70:
        return "MATCH"
    elif score <= 0.45:
        return "MISMATCH"
    else:
        return "UNCERTAIN"
