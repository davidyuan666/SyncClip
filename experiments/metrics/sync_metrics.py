"""
Temporal synchronization error metrics (Eq. 16 in paper).
Reports mean absolute error and RMS error in seconds and milliseconds.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class SyncMetrics:
    mean_abs_error_s: float = 0.0
    mean_abs_error_ms: float = 0.0
    rms_error_s: float = 0.0
    rms_error_ms: float = 0.0
    median_error_ms: float = 0.0
    p95_error_ms: float = 0.0
    n_alignment_pairs: int = 0
    per_genre: Dict[str, "SyncMetrics"] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        base = {
            "mean_abs_error_s": self.mean_abs_error_s,
            "mean_abs_error_ms": self.mean_abs_error_ms,
            "rms_error_s": self.rms_error_s,
            "rms_error_ms": self.rms_error_ms,
            "median_error_ms": self.median_error_ms,
            "p95_error_ms": self.p95_error_ms,
            "n_alignment_pairs": self.n_alignment_pairs,
        }
        if self.per_genre:
            base["per_genre"] = {g: m.to_dict() for g, m in self.per_genre.items()}
        return base

    def __repr__(self):
        return f"SyncMetrics(MAE={self.mean_abs_error_ms:.1f}ms, RMS={self.rms_error_ms:.1f}ms, n={self.n_alignment_pairs})"


def compute_sync_error(
    predicted_alignments: List[Dict],
    reference_alignments: List[Dict],
) -> SyncMetrics:
    """
    Compute temporal synchronization error.

    Args:
        predicted_alignments: [{"video_time": float, "audio_time": float}, ...]
        reference_alignments: [{"video_time": float, "audio_time": float}, ...]

    The error is |t_video - t_audio| for each matched pair.
    When mismatched lengths, align pairs by closest timestamp matching.
    """
    if not predicted_alignments or not reference_alignments:
        return SyncMetrics(n_alignment_pairs=0)

    errors = []
    for pred in predicted_alignments:
        p_t = pred.get("video_time", pred.get("start", 0.0))
        best_error = float("inf")
        for ref in reference_alignments:
            r_t = ref.get("video_time", ref.get("start", 0.0))
            err = abs(p_t - r_t)
            if err < best_error:
                best_error = err
        errors.append(best_error)

    errors = np.array(errors)
    n = len(errors)

    return SyncMetrics(
        mean_abs_error_s=round(float(np.mean(errors)), 4),
        mean_abs_error_ms=round(float(np.mean(errors) * 1000), 1),
        rms_error_s=round(float(np.sqrt(np.mean(errors ** 2))), 4),
        rms_error_ms=round(float(np.sqrt(np.mean(errors ** 2)) * 1000), 1),
        median_error_ms=round(float(np.median(errors) * 1000), 1),
        p95_error_ms=round(float(np.percentile(errors, 95) * 1000), 1),
        n_alignment_pairs=n,
    )


def compute_genre_sync_breakdown(
    predictions_by_genre: Dict[str, List[Dict]],
    references_by_genre: Dict[str, List[Dict]],
) -> Dict[str, SyncMetrics]:
    return {
        genre: compute_sync_error(
            predictions_by_genre.get(genre, []),
            references_by_genre.get(genre, []),
        )
        for genre in references_by_genre
    }
