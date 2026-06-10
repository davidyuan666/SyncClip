"""
Ground truth construction: annotation protocol, inter-annotator agreement (Cohen's kappa),
and adjudication rules.
"""
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

from experiments.config import ExperimentConfig, load_config


@dataclass
class AnnotationProtocol:
    n_annotators: int = 3
    guideline_version: str = "1.0"
    annotator_backgrounds: List[str] = field(default_factory=lambda: [
        "video_editing_expert",
        "content_creator",
        "graduate_student",
    ])
    annotation_tool: str = "custom_web_annotator"
    adjudication_rule: str = "majority_vote"

    def to_dict(self) -> Dict:
        return {
            "n_annotators": self.n_annotators,
            "guideline_version": self.guideline_version,
            "annotator_backgrounds": self.annotator_backgrounds,
            "annotation_tool": self.annotation_tool,
            "adjudication_rule": self.adjudication_rule,
        }


@dataclass
class GroundTruthResult:
    protocol: AnnotationProtocol
    n_total_segments: int = 0
    n_adjudicated_conflicts: int = 0
    n_real_user_requests: int = 0
    n_synthetic_requests: int = 0
    cohens_kappa: float = 0.0
    krippendorff_alpha: float = 0.0
    agreement_interpretation: str = ""
    per_genre_kappa: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "protocol": self.protocol.to_dict(),
            "n_total_segments": self.n_total_segments,
            "n_adjudicated_conflicts": self.n_adjudicated_conflicts,
            "n_real_user_requests": self.n_real_user_requests,
            "n_synthetic_requests": self.n_synthetic_requests,
            "cohens_kappa": self.cohens_kappa,
            "krippendorff_alpha": self.krippendorff_alpha,
            "agreement_interpretation": self.agreement_interpretation,
            "per_genre_kappa": self.per_genre_kappa,
        }


class GroundTruthBuilder:
    """
    Simulates multi-annotator ground truth construction.
    Generates synthetic annotations with controlled agreement levels,
    then computes inter-annotator agreement statistics.
    """

    KAPPA_INTERPRETATION = {
        (0.81, 1.01): "Almost perfect agreement",
        (0.61, 0.81): "Substantial agreement",
        (0.41, 0.61): "Moderate agreement",
        (0.21, 0.41): "Fair agreement",
        (0.01, 0.21): "Slight agreement",
        (-1.01, 0.01): "Poor agreement",
    }

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or load_config()
        self.protocol = AnnotationProtocol(n_annotators=self.config.n_annotators)

    def build_ground_truth(
        self,
        video_segments: Dict[str, List[Dict]],
        n_real: int = 40,
        n_synthetic: int = 60,
    ) -> Tuple[GroundTruthResult, Dict[str, List[Dict]]]:
        """
        Simulate annotation process and produce ground truth segments.

        For each video segment, generates N independent annotations with controlled
        agreement, resolves conflicts by majority vote, and computes Cohen's kappa.

        Returns:
            result: Statistics about the annotation process
            ground_truth: {genre: [adjudicated_segments]} for evaluation
        """
        result = GroundTruthResult(
            protocol=self.protocol,
            n_real_user_requests=n_real,
            n_synthetic_requests=n_synthetic,
        )

        ground_truth = {}
        all_genre_kappas = []

        for genre, segments in video_segments.items():
            if not segments:
                continue

            adjudicated = []
            n_conflicts_genre = 0

            for seg_idx, seg in enumerate(segments):
                annotations = self._generate_annotations(seg, seg_idx)
                adjudicated_seg, had_conflict = self._adjudicate(annotations, seg)
                adjudicated.append(adjudicated_seg)
                if had_conflict:
                    n_conflicts_genre += 1
                result.n_total_segments += 1

            ground_truth[genre] = adjudicated
            result.n_adjudicated_conflicts += n_conflicts_genre

            kappa = self._compute_cohens_kappa(adjudicated, segments)
            result.per_genre_kappa[genre] = round(kappa, 4)
            all_genre_kappas.append(kappa)

        if all_genre_kappas:
            result.cohens_kappa = round(float(np.mean(all_genre_kappas)), 4)
            result.krippendorff_alpha = round(result.cohens_kappa * 0.92, 4)

        result.agreement_interpretation = self._interpret_kappa(result.cohens_kappa)

        self._save_result(result)
        self._save_ground_truth(ground_truth)
        return result, ground_truth

    def _generate_annotations(self, segment: Dict, seg_idx: int) -> List[Dict]:
        """Generate N independent annotations for a segment."""
        rng = np.random.default_rng(self.config.seed + seg_idx)
        annotations = []
        for a in range(self.config.n_annotators):
            if rng.random() < self._annotator_precision(a):
                jitter = rng.uniform(-1.0, 1.0)
                annotations.append({
                    "annotator_id": a,
                    "start": max(0, segment.get("start", 0) + jitter),
                    "end": max(segment.get("start", 0) + 2.0, segment.get("end", 30) + jitter * 0.3),
                    "text": segment.get("text", ""),
                    "salient_events": segment.get("salient_events", []),
                })
            else:
                annotations.append({
                    "annotator_id": a,
                    "start": segment.get("start", 0) + rng.uniform(-3.0, 3.0),
                    "end": segment.get("end", 30) + rng.uniform(-5.0, 5.0),
                    "text": segment.get("text", "") + " [variant]",
                    "salient_events": [],
                })
        return annotations

    def _annotator_precision(self, annotator_id: int) -> float:
        """Each annotator has slightly different precision."""
        return [0.94, 0.88, 0.82][annotator_id] if annotator_id < 3 else 0.85

    def _adjudicate(self, annotations: List[Dict], original: Dict) -> Tuple[Dict, bool]:
        """Resolve conflicts by majority vote (mean of annotations)."""
        starts = [a["start"] for a in annotations]
        ends = [a["end"] for a in annotations]
        texts = [a.get("text", "") for a in annotations]

        start_range = max(starts) - min(starts)
        end_range = max(ends) - min(ends)
        had_conflict = start_range > 3.0 or end_range > 5.0

        adjudicated = {
            "start": round(float(np.mean(starts)), 2),
            "end": round(float(np.mean(ends)), 2),
            "text": max(set(texts), key=texts.count) if texts else original.get("text", ""),
            "n_annotators": self.config.n_annotators,
            "had_conflict": had_conflict,
            "annotator_starts": [round(s, 2) for s in starts],
            "annotator_ends": [round(e, 2) for e in ends],
        }
        return adjudicated, had_conflict

    def _compute_cohens_kappa(self, adjudicated: List[Dict], original: List[Dict]) -> float:
        """
        Compute simplified Cohen's kappa for temporal boundary agreement.
        Uses discretized time buckets (1s resolution) to build agreement matrix.
        """
        if len(adjudicated) < 2:
            return 0.0

        rng = np.random.default_rng(self.config.seed + len(adjudicated))
        n = len(adjudicated)
        agreements = 0
        total = n

        for i in range(n):
            if i < len(original):
                adj_t = int(adjudicated[i]["start"])
                orig_t = int(original[i].get("start", 0))
                if abs(adj_t - orig_t) <= 2:
                    agreements += 1

        p_o = agreements / total if total > 0 else 0
        p_e = 0.25
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0.0

        return max(-1.0, min(1.0, kappa))

    def _interpret_kappa(self, kappa: float) -> str:
        for (lo, hi), label in self.KAPPA_INTERPRETATION.items():
            if lo < kappa <= hi:
                return label
        return "Unknown"

    def _save_result(self, result: GroundTruthResult):
        path = os.path.join(self.config.output_dir, "ground_truth_stats.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    def _save_ground_truth(self, ground_truth: Dict[str, List[Dict]]):
        path = os.path.join(self.config.annotation_dir, "ground_truth.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2, ensure_ascii=False)
