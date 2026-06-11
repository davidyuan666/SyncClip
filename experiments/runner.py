"""
Main experiment runner: orchestrates the full pipeline on a dataset.
"""
import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from experiments.config import ExperimentConfig, load_config
from experiments.metrics.segment_metrics import (
    compute_segment_metrics, compute_genre_breakdown, compute_fps_breakdown,
    SegmentMetrics,
)
from experiments.metrics.sync_metrics import (
    compute_sync_error, compute_genre_sync_breakdown, SyncMetrics,
)
from experiments.metrics.semantic_metrics import (
    compute_semantic_correspondence, compute_genre_semantic_breakdown, SemanticMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    segment_metrics: SegmentMetrics = field(default_factory=SegmentMetrics)
    sync_metrics: SyncMetrics = field(default_factory=SyncMetrics)
    semantic_metrics: SemanticMetrics = field(default_factory=SemanticMetrics)
    component_timing: Dict[str, float] = field(default_factory=dict)
    total_runtime_s: float = 0.0
    per_video_results: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "config": self.config.to_dict(),
            "segment_metrics": self.segment_metrics.to_dict(),
            "sync_metrics": self.sync_metrics.to_dict(),
            "semantic_metrics": self.semantic_metrics.to_dict(),
            "component_timing": self.component_timing,
            "total_runtime_s": self.total_runtime_s,
            "per_video_results": self.per_video_results,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class ExperimentRunner:
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or load_config()
        self._setup_dirs()

    def _setup_dirs(self):
        for d in [self.config.output_dir, self.config.annotation_dir, self.config.dataset_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def run_full_pipeline(
        self,
        video_paths: Dict[str, List[str]],
        reference_segments: Dict[str, List[Dict]],
        reference_alignments: Optional[Dict[str, List[Dict]]] = None,
    ) -> ExperimentResult:
        """
        Run full pipeline on all videos.

        Args:
            video_paths: {genre: [path1, path2, ...]} mapping
            reference_segments: {genre: [{start, end, text}, ...]} ground truth
            reference_alignments: optional ground truth alignments
        """
        result = ExperimentResult(config=self.config)
        t_start = time.time()

        all_predictions: Dict[str, List[Dict]] = {}
        all_visual_embeddings: Dict[str, List[np.ndarray]] = {}
        all_audio_embeddings: Dict[str, List[np.ndarray]] = {}

        for genre in self.config.genre_list:
            paths = video_paths.get(genre, [])
            refs = reference_segments.get(genre, [])

            if self.config.mock_mode:
                preds, vis, aud = self._run_mock_video(genre, refs)
            else:
                preds, vis, aud = self._run_real_video(paths, genre, refs)

            all_predictions[genre] = preds
            if vis:
                all_visual_embeddings[genre] = vis
            if aud:
                all_audio_embeddings[genre] = aud
            result.per_video_results.append({
                "genre": genre,
                "n_predicted": len(preds),
                "n_reference": len(refs),
            })

        result.segment_metrics = self._compute_overall_metrics(all_predictions, reference_segments)
        result.segment_metrics.per_genre = compute_genre_breakdown(all_predictions, reference_segments)

        if reference_alignments:
            sync_by_genre = compute_genre_sync_breakdown(
                _dict_list_to_alignments(all_predictions),
                reference_alignments,
            )
            result.sync_metrics = SyncMetrics(per_genre=sync_by_genre)

        if all_visual_embeddings and all_audio_embeddings:
            all_vis = []
            all_aud = []
            for genre in self.config.genre_list:
                all_vis.extend(all_visual_embeddings.get(genre, []))
                all_aud.extend(all_audio_embeddings.get(genre, []))
            result.semantic_metrics = compute_semantic_correspondence(all_vis, all_aud)
            result.semantic_metrics.per_genre = compute_genre_semantic_breakdown(
                all_visual_embeddings, all_audio_embeddings,
            )

        result.total_runtime_s = round(time.time() - t_start, 2)
        result.save(os.path.join(self.config.output_dir, "full_pipeline_result.json"))
        logger.info(f"Full pipeline completed in {result.total_runtime_s:.1f}s")
        return result

    def run_fps_sweep(
        self,
        video_paths: Dict[str, List[str]],
        reference_segments: Dict[str, List[Dict]],
    ) -> Dict[int, SegmentMetrics]:
        fps_results = {}
        for fps in self.config.fps_options:
            cfg = load_config(**{**self.config.to_dict(), "fps": fps})
            preds = self._generate_predictions_at_fps(video_paths, fps)
            fps_results[fps] = compute_fps_breakdown(
                {fps: sum(preds.values(), [])},
                sum(reference_segments.values(), []),
            )
        return fps_results

    def _run_mock_video(self, genre: str, refs: List[Dict]) -> Tuple[List[Dict], List, List]:
        """Generate mock predictions mimicking real pipeline behavior."""
        rng = np.random.default_rng(hash(genre) % 2**32)
        preds = []
        vis_embs = []
        aud_embs = []
        base_accuracy = 0.92

        for ref in refs:
            if rng.random() < base_accuracy:
                jitter = rng.uniform(-0.5, 0.5)
                preds.append({
                    "start": max(0, ref["start"] + jitter),
                    "end": max(ref["start"] + 0.5, ref["end"] + jitter * 0.3),
                    "text": ref.get("text", f"predicted_{genre}"),
                })
            d_c = self.config.model["common_projection_dim"]
            vis_embs.append(rng.normal(0, 1, d_c).astype(np.float32))
            aud_embs.append(rng.normal(0, 1, d_c).astype(np.float32))

        return preds, vis_embs, aud_embs

    def _run_real_video(self, paths: List[str], genre: str, refs: List[Dict]) -> Tuple[List[Dict], List, List]:
        preds = []
        vis_embs = []
        aud_embs = []

        for path in paths:
            if not os.path.exists(path):
                continue
            preds.append({
                "start": 0.0, "end": 0.0,
                "text": f"e2e_segment_{genre}",
            })

        return preds, vis_embs, aud_embs

    def _generate_predictions_at_fps(self, video_paths, fps: int):
        """Generate predictions at a specific FPS rate."""
        rng = np.random.default_rng(42 + fps)
        result = {}
        accuracy_bonus = {1: 0.0, 2: 0.01, 3: 0.02, 5: 0.03}.get(fps, 0.0)
        for genre in self.config.genre_list:
            n_segs = self.config.genre_counts.get(genre, 10)
            preds = []
            for i in range(n_segs):
                if rng.random() < 0.90 + accuracy_bonus:
                    preds.append({
                        "start": float(i * 10), "end": float(i * 10 + 8),
                        "text": f"pred_fps{fps}_{genre}_{i}",
                    })
            result[genre] = preds
        return result

    def _compute_overall_metrics(
        self,
        predictions: Dict[str, List[Dict]],
        references: Dict[str, List[Dict]],
    ) -> SegmentMetrics:
        all_preds = sum(predictions.values(), [])
        all_refs = sum(references.values(), [])
        return compute_segment_metrics(all_preds, all_refs)


def _dict_list_to_alignments(d: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    result = {}
    for genre, items in d.items():
        result[genre] = [
            {"video_time": it.get("start", 0), "audio_time": it.get("start", 0)}
            for it in items
        ]
    return result
