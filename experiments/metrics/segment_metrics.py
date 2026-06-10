"""
Segment selection metrics: Precision, Recall, F1-score.
Matches the paper's definitions (Eq. 13-15).
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class SegmentMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    per_genre: Dict[str, "SegmentMetrics"] = field(default_factory=dict)
    per_fps: Dict[int, "SegmentMetrics"] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        base = {"precision": self.precision, "recall": self.recall, "f1_score": self.f1_score,
                "tp": self.tp, "fp": self.fp, "fn": self.fn}
        if self.per_genre:
            base["per_genre"] = {g: m.to_dict() for g, m in self.per_genre.items()}
        if self.per_fps:
            base["per_fps"] = {str(f): m.to_dict() for f, m in self.per_fps.items()}
        return base

    def __repr__(self):
        return f"SegmentMetrics(P={self.precision:.3f}, R={self.recall:.3f}, F1={self.f1_score:.3f})"


def _match_segments(
    predicted: List[Dict],
    reference: List[Dict],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """
    Match predicted segments to reference using temporal IoU.
    A predicted segment tp if IoU > iou_threshold with a reference segment.
    """
    matched_ref = set()
    tp = 0

    for pred in predicted:
        p_start, p_end = pred["start"], pred["end"]
        best_iou = 0.0
        best_idx = -1
        for j, ref in enumerate(reference):
            if j in matched_ref:
                continue
            r_start, r_end = ref["start"], ref["end"]
            inter = max(0.0, min(p_end, r_end) - max(p_start, r_start))
            union = max(p_end, r_end) - min(p_start, r_start)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou >= iou_threshold:
            tp += 1
            matched_ref.add(best_idx)

    fp = len(predicted) - tp
    fn = len(reference) - len(matched_ref)
    return tp, fp, fn


def compute_segment_metrics(
    predicted_segments: List[Dict],
    reference_segments: List[Dict],
    iou_threshold: float = 0.5,
) -> SegmentMetrics:
    tp, fp, fn = _match_segments(predicted_segments, reference_segments, iou_threshold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return SegmentMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        tp=tp, fp=fp, fn=fn,
    )


def compute_genre_breakdown(
    all_predictions: Dict[str, List[Dict]],
    all_references: Dict[str, List[Dict]],
    iou_threshold: float = 0.5,
) -> Dict[str, SegmentMetrics]:
    return {
        genre: compute_segment_metrics(
            all_predictions.get(genre, []),
            all_references.get(genre, []),
            iou_threshold,
        )
        for genre in all_references
    }


def compute_fps_breakdown(
    predictions_by_fps: Dict[int, List[Dict]],
    reference_segments: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[int, SegmentMetrics]:
    return {
        fps: compute_segment_metrics(segs, reference_segments, iou_threshold)
        for fps, segs in predictions_by_fps.items()
    }
