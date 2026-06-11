#!/usr/bin/env python
"""
Human evaluation helper for SyncCLIPAgent experiments.

Generates a CSV template for manual scoring of rendered video outputs,
then merges completed scores back into the experiment results.

Usage:
    python scripts/human_eval.py --generate     Create human_eval_template.csv
    python scripts/human_eval.py --merge        Merge filled scores into e2e_results.json

Scoring dimensions (1=poor, 5=excellent):
    overall_score       Overall impression of the edited video
    segment_quality     Relevance of selected clips to the editing goal
    pacing               Rhythm, flow, and transition quality
    sync_quality         Audiovisual synchronization accuracy
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCORE_COLUMNS = ["overall_score", "segment_quality", "pacing", "sync_quality", "notes"]
TEMPLATE_COLUMNS = [
    "video_id", "n_segments_planned", "planned_duration_s",
    "validation_passed", "render_success", "preprocess_s", "clip_s",
    "whisper_s", "llm_plan_s", "render_s", "total_s",
] + SCORE_COLUMNS


def _find_output_dir(base: str = "experiments/output") -> str:
    candidate = Path(base)
    if (candidate / "e2e_results.json").exists():
        return str(candidate.resolve())
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = Path(project_root) / "experiments" / "output"
    if (candidate / "e2e_results.json").exists():
        return str(candidate.resolve())
    return str(candidate.resolve())


def generate_template(output_dir: str = "") -> str:
    output_dir = output_dir or _find_output_dir()
    e2e_path = os.path.join(output_dir, "e2e_results.json")
    if not os.path.exists(e2e_path):
        print(f"ERROR: {e2e_path} not found. Run experiments first.")
        sys.exit(1)

    with open(e2e_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    template_path = os.path.join(output_dir, "human_eval_template.csv")
    with open(template_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=TEMPLATE_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for r in results:
            timing = r.get("timing", {})
            row = {
                "video_id": r.get("video_id", "?"),
                "n_segments_planned": r.get("n_segments_planned", 0),
                "planned_duration_s": r.get("planned_duration_s", 0),
                "validation_passed": r.get("validation_passed", False),
                "render_success": r.get("render_success", False),
                "preprocess_s": timing.get("preprocess_s", ""),
                "clip_s": timing.get("clip_s", ""),
                "whisper_s": timing.get("whisper_s", ""),
                "llm_plan_s": timing.get("llm_plan_s", ""),
                "render_s": timing.get("render_s", ""),
                "total_s": timing.get("total_s", ""),
                "overall_score": "",
                "segment_quality": "",
                "pacing": "",
                "sync_quality": "",
                "notes": "",
            }
            writer.writerow(row)

    print(f"\nTemplate generated: {template_path}")
    print(f"  {len(results)} rows (one per video)")
    print(f"\n  Scoring guidelines (1=poor, 5=excellent):")
    print(f"    overall_score    - Overall impression of the edited video")
    print(f"    segment_quality  - Relevance of selected clips to the editing goal")
    print(f"    pacing           - Rhythm, flow, and transition quality")
    print(f"    sync_quality     - Audiovisual synchronization accuracy")
    print(f"    notes            - Optional comments on specific issues")
    print(f"\n  After filling scores, run:")
    print(f"    python scripts/human_eval.py --merge")
    return template_path


def merge_scores(output_dir: str = "") -> None:
    output_dir = output_dir or _find_output_dir()
    e2e_path = os.path.join(output_dir, "e2e_results.json")
    template_path = os.path.join(output_dir, "human_eval_template.csv")

    if not os.path.exists(template_path):
        print(f"ERROR: {template_path} not found. Run --generate first.")
        sys.exit(1)
    if not os.path.exists(e2e_path):
        print(f"ERROR: {e2e_path} not found.")
        sys.exit(1)

    scores: Dict[str, Dict] = {}
    with open(template_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", "").strip()
            if not vid:
                continue
            score_row = {col: row.get(col, "") for col in SCORE_COLUMNS}
            for col in SCORE_COLUMNS[:-1]:
                try:
                    score_row[col] = int(score_row[col])
                except (ValueError, TypeError):
                    score_row[col] = None
            scores[vid] = score_row

    with open(e2e_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    merged = 0
    for r in results:
        vid = r.get("video_id", "")
        if vid in scores:
            merged += 1
            r["human_eval"] = {k: v for k, v in scores[vid].items() if v is not None}

    with open(e2e_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = _compute_eval_summary(results)
    summary_path = os.path.join(output_dir, "human_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nMerged scores for {merged}/{len(results)} videos")
    print(f"Summary saved to: {summary_path}")
    _print_summary(summary)


def _compute_eval_summary(results: List[Dict]) -> Dict:
    dims = ["overall_score", "segment_quality", "pacing", "sync_quality"]
    summary: Dict = {"n_evaluated": 0, "dimensions": {}}

    scores_by_dim = {d: [] for d in dims}
    for r in results:
        he = r.get("human_eval", {})
        if not he:
            continue
        summary["n_evaluated"] += 1
        for d in dims:
            val = he.get(d)
            if val is not None and isinstance(val, (int, float)):
                scores_by_dim[d].append(float(val))

    for d in dims:
        vals = scores_by_dim[d]
        if vals:
            summary["dimensions"][d] = {
                "mean": round(sum(vals) / len(vals), 2),
                "min": min(vals),
                "max": max(vals),
                "std": round(_std(vals), 2),
            }
    return summary


def _print_summary(summary: Dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"Human Evaluation Summary ({summary['n_evaluated']} videos)")
    print(f"{'=' * 50}")
    for dim, stats in summary.get("dimensions", {}).items():
        print(f"  {dim:20s}  mean={stats['mean']:.2f}  "
              f"min={stats['min']:.0f}  max={stats['max']:.0f}  std={stats['std']:.2f}")


def _std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5


def main():
    parser = argparse.ArgumentParser(description="SyncCLIPAgent Human Evaluation Helper")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true", help="Generate CSV template for scoring")
    group.add_argument("--merge", action="store_true", help="Merge completed scores into e2e_results.json")
    parser.add_argument("--output", type=str, default="", help="Output directory (default: experiments/output)")
    args = parser.parse_args()

    out = args.output or _find_output_dir()

    if args.generate:
        generate_template(out)
    elif args.merge:
        merge_scores(out)


if __name__ == "__main__":
    main()
