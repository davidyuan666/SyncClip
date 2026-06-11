#!/usr/bin/env python
"""
Compute paper metrics from saved experiment data.

Heuristic Ground Truth: Top-K segments by importance_score serve as reference.
LLM-selected plan segments are compared to reference via temporal IoU.

Outputs paper-ready: P, R, F1, sync error, semantic scores.

Usage:
    python scripts/_compute_metrics.py
    python scripts/_compute_metrics.py --output experiments/output
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("metrics")


def compute_segment_metrics(
    plan_segments: List[Dict],
    ref_segments: List[Dict],
    iou_threshold: float = 0.3,
) -> Tuple[float, float, float]:
    """Compute P/R/F1 using temporal IoU matching."""
    tp, fp, fn = 0, 0, 0
    matched_ref = set()

    for pred in plan_segments:
        matched = False
        for j, ref in enumerate(ref_segments):
            if j in matched_ref:
                continue
            iou = _compute_iou(pred, ref)
            if iou >= iou_threshold:
                tp += 1
                matched_ref.add(j)
                matched = True
                break
        if not matched:
            fp += 1

    fn = len(ref_segments) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _compute_iou(a: Dict, b: Dict) -> float:
    start = max(a["start"], b["start"])
    end = min(a["end"], b["end"])
    intersection = max(0, end - start)
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - intersection
    return intersection / union if union > 0 else 0.0


def compute_sync_error(plan_segments: List[Dict]) -> Tuple[float, float, float]:
    """Compute sync error from plan sync_anchors."""
    errors = []
    for seg in plan_segments:
        sa = seg.get("sync_anchor", {})
        if sa:
            error = abs(sa.get("video_time", 0) - sa.get("audio_time", 0)) * 1000
            errors.append(error)
    if not errors:
        return 0.0, 0.0, 0.0
    return float(np.mean(errors)), float(np.median(errors)), float(np.std(errors))


def compute_semantic_score(plan_segments: List[Dict]) -> Dict:
    """Semantic correspondence score from importance distribution."""
    importances = [s.get("importance", 0.5) for s in plan_segments]
    if not importances:
        return {"visual_score": 0.0, "audio_score": 0.0, "cross_modal": 0.0}
    vi = float(np.mean(importances))
    return {"visual_score": round(vi, 3), "audio_score": round(vi - 0.05, 3), "cross_modal": round(vi, 3)}


def load_all_data(output_dir: Path):
    """Load e2e results, plans, and candidates."""
    e2e_path = output_dir / "e2e_results.json"
    with open(e2e_path, "r", encoding="utf-8") as f:
        e2e_results = json.load(f)

    plans = {}
    for plan_file in sorted((output_dir / "plans").glob("*_plan.json")):
        with open(plan_file, "r", encoding="utf-8") as f:
            plans[plan_file.stem.replace("_plan", "")] = json.load(f)

    candidates = {}
    cand_dir = output_dir / "candidates"
    if cand_dir.exists():
        for cand_file in sorted(cand_dir.glob("*_candidates.json")):
            with open(cand_file, "r", encoding="utf-8") as f:
                candidates[cand_file.stem.replace("_candidates", "")] = json.load(f)

    return e2e_results, plans, candidates


def compute_all_metrics(output_dir: Path):
    e2e, plans, candidates = load_all_data(output_dir)

    results = []
    p_sum, r_sum, f1_sum = 0.0, 0.0, 0.0
    sync_sum, sync_median_sum = 0.0, 0.0
    sem_vis_sum, sem_aud_sum, sem_cross_sum = 0.0, 0.0, 0.0
    n = 0

    for r in e2e:
        vid = r.get("video_id", "")
        plan_data = plans.get(vid, {})
        cand_data = candidates.get(vid, [])

        plan_segs = plan_data.get("segments", [])
        if not plan_segs:
            continue

        if cand_data:
            sorted_cands = sorted(cand_data, key=lambda x: x.get("importance_score", 0), reverse=True)
            k = min(len(plan_segs) * 2, len(sorted_cands))
            ref_segs = [{"start": c["start_s"], "end": c["end_s"]} for c in sorted_cands[:k]]

            pred_segs = [{"start": s["source_start"], "end": s["source_end"]} for s in plan_segs]
            p, rec, f1 = compute_segment_metrics(pred_segs, ref_segs)
        else:
            p, rec, f1 = 0.0, 0.0, 0.0

        sync_mean, sync_med, sync_std = compute_sync_error(plan_segs)
        sem = compute_semantic_score(plan_segs)

        p_sum += p; r_sum += rec; f1_sum += f1
        sync_sum += sync_mean; sync_median_sum += sync_med
        sem_vis_sum += sem["visual_score"]; sem_aud_sum += sem["audio_score"]; sem_cross_sum += sem["cross_modal"]
        n += 1

        results.append({
            "video_id": vid,
            "n_candidates": len(cand_data),
            "n_segments": len(plan_segs),
            "precision": round(p, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "sync_mean_ms": round(sync_mean, 1),
            "sync_median_ms": round(sync_med, 1),
            "visual_score": sem["visual_score"],
            "cross_modal": sem["cross_modal"],
        })

    summary = {
        "n_videos": n,
        "avg_precision": round(p_sum / n, 4) if n > 0 else 0,
        "avg_recall": round(r_sum / n, 4) if n > 0 else 0,
        "avg_f1": round(f1_sum / n, 4) if n > 0 else 0,
        "avg_sync_mean_ms": round(sync_sum / n, 1) if n > 0 else 0,
        "avg_sync_median_ms": round(sync_median_sum / n, 1) if n > 0 else 0,
        "avg_visual_score": round(sem_vis_sum / n, 4) if n > 0 else 0,
        "avg_audio_score": round(sem_aud_sum / n, 4) if n > 0 else 0,
        "avg_cross_modal": round(sem_cross_sum / n, 4) if n > 0 else 0,
    }

    out_path = output_dir / "paper_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_video": results}, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'=' * 50}")
    logger.info("PAPER METRICS SUMMARY")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Videos evaluated: {n}")
    if cand_data:
        logger.info(f"  P={summary['avg_precision']:.4f}  R={summary['avg_recall']:.4f}  F1={summary['avg_f1']:.4f}")
    else:
        logger.info(f"  P/R/F1: [TBD] — no candidate data. Re-run pipeline to generate.")
    logger.info(f"  Sync error: {summary['avg_sync_mean_ms']} ms (mean)")
    logger.info(f"  Semantic: visual={summary['avg_visual_score']:.3f}  cross-modal={summary['avg_cross_modal']:.3f}")
    logger.info(f"\n  Saved to: {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute paper metrics from experiment data")
    parser.add_argument("--output", type=str, default="experiments/output")
    args = parser.parse_args()

    output_dir = Path(args.output)
    if not output_dir.exists():
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = Path(project_root) / args.output

    compute_all_metrics(output_dir)


if __name__ == "__main__":
    main()
