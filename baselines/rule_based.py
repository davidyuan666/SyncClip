"""
Rule-Based Video Highlighting Baseline.

Algorithm:
  1. Extract frames @5fps via ffmpeg.
  2. Compute frame-to-frame RGB difference (shot boundary detection).
  3. Extract audio RMS energy per segment.
  4. Candidate segments = intervals between shot boundaries.
  5. Score = 0.6 * visual_change + 0.4 * audio_energy.
  6. Greedy selection to 60s target.
  7. Evaluate against heuristic GT (top-K importance candidates).
  8. Output results.json + summary.json.

Usage:
    python -m baselines.rule_based
    python -m baselines.rule_based --data experiments/data/vlog/
    python -m baselines.rule_based --data experiments/data/vlog/ --output experiments/output/baselines/rule_based/
"""
import argparse, logging, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.common import (
    extract_frames, frame_diff_scores, audio_rms_per_window,
    greedy_segment_selection, compute_segment_metrics, save_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rule_based")

VIZ_WEIGHT = 0.6
AUD_WEIGHT = 0.4
TARGET_DURATION = 60.0
FPS = 5


def build_candidates(
    video_path: str, work_dir: str, audio_path: str, video_id: str,
) -> List[Dict]:
    """Build candidate segments from shot boundaries + audio energy."""
    frames = extract_frames(video_path, work_dir, FPS)
    if len(frames) < 2:
        return []

    diff = frame_diff_scores(frames)
    threshold = np.percentile(diff, 70)
    boundaries = [0]
    for i in range(1, len(diff)):
        if diff[i] > threshold:
            boundaries.append(i)
    if boundaries[-1] != len(frames):
        boundaries.append(len(frames))

    interval_s = 1.0 / FPS
    segments = []
    for seg_idx in range(len(boundaries) - 1):
        i0, i1 = boundaries[seg_idx], boundaries[seg_idx + 1]
        if i1 - i0 < 2:
            continue
        start_s = i0 * interval_s
        end_s = i1 * interval_s
        dur = end_s - start_s
        if dur < 0.5:
            continue

        viz_score = float(np.mean(diff[i0:i1])) / max(1.0, np.mean(diff))
        viz_score = min(viz_score, 1.0)

        segments.append({
            "segment_id": f"{video_id}_seg_{seg_idx:04d}",
            "start_s": round(start_s, 2),
            "end_s": round(end_s, 2),
            "viz_score": round(viz_score, 4),
        })

    rms = audio_rms_per_window(audio_path, window_s=1.0, sr=16000)
    total_dur_s = len(frames) * interval_s
    for seg in segments:
        seg_start_w = int(seg["start_s"])
        seg_end_w = min(int(seg["end_s"]) + 1, len(rms)) if len(rms) > 0 else 0
        aud_score = float(np.mean(rms[seg_start_w:seg_end_w])) / max(1.0, np.mean(rms)) if seg_end_w > seg_start_w else 0.5
        aud_score = min(float(aud_score), 1.0)
        seg["aud_score"] = round(aud_score, 4)
        seg["score"] = round(VIZ_WEIGHT * seg["viz_score"] + AUD_WEIGHT * aud_score, 4)

    return segments


def run_baseline(video_dir: str, output_dir: str):
    vlog = Path(video_dir) / "vlog" if (Path(video_dir) / "vlog").exists() else Path(video_dir)
    videos = sorted(vlog.glob("*.mp4"))
    if not videos:
        logger.error(f"No MP4 files found in {vlog}")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)

    results = []
    for i, vp in enumerate(videos):
        vid = vp.stem
        t0 = time.time()
        logger.info(f"[{i + 1}/{len(videos)}] {vid}")

        video_work = str(work / vid)
        candidates = build_candidates(str(vp), video_work, str(vp), vid)
        plan = greedy_segment_selection(candidates, TARGET_DURATION)

        ref = sorted(candidates, key=lambda x: x["score"], reverse=True)[:len(plan) * 2]
        p, r, f1 = compute_segment_metrics(plan, ref)

        elapsed = round(time.time() - t0, 2)
        results.append({
            "video_id": vid,
            "n_candidates": len(candidates),
            "n_selected": len(plan),
            "precision": p, "recall": r, "f1_score": f1,
            "runtime_s": elapsed,
        })
        logger.info(f"  candidates={len(candidates)} segs={len(plan)} P={p:.3f} R={r:.3f} F1={f1:.3f} ({elapsed}s)")

    summary = {
        "baseline": "Rule-Based",
        "n_videos": len(results),
        "avg_precision": round(float(np.mean([r["precision"] for r in results])), 4),
        "avg_recall": round(float(np.mean([r["recall"] for r in results])), 4),
        "avg_f1": round(float(np.mean([r["f1_score"] for r in results])), 4),
        "avg_runtime_s": round(float(np.mean([r["runtime_s"] for r in results])), 2),
        "target_duration_s": TARGET_DURATION,
        "fps": FPS,
    }
    save_results(output_dir, "Rule-Based", results, summary)


def main():
    parser = argparse.ArgumentParser(description="Rule-Based Video Highlighting")
    parser.add_argument("--data", type=str, default="experiments/data/vlog/")
    parser.add_argument("--output", type=str, default="experiments/output/baselines/rule_based/")
    args = parser.parse_args()
    run_baseline(args.data, args.output)


if __name__ == "__main__":
    main()
