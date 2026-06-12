"""
SVM-Based Video Highlighting Baseline.

Algorithm:
  1. Extract frames @5fps + audio via ffmpeg.
  2. Visual features: frame-to-frame diff + HSV histogram variance + edge density.
  3. Audio features: RMSE per 1s window.
  4. Pseudo-label: top 20% segments = positive, bottom 20% = negative.
  5. Train sklearn LinearSVC.
  6. SVM scores all candidates.
  7. Greedy selection to 60s target.
  8. Evaluate against heuristic GT (top-K importance candidates).
  9. Output results.json + summary.json.

Usage:
    python -m baselines.svm_based
    python -m baselines.svm_based --data experiments/data/vlog/
    python -m baselines.svm_based --output experiments/output/baselines/svm_based/
"""
import argparse, logging, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.common import (
    extract_frames, frame_diff_scores, audio_rms_per_window,
    color_histogram_features, edge_density,
    greedy_segment_selection, compute_segment_metrics, save_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("svm_based")

TARGET_DURATION = 60.0
FPS = 5
POSITIVE_RATIO = 0.20
NEGATIVE_RATIO = 0.20


def build_candidates_svm(
    video_path: str, work_dir: str, audio_path: str, video_id: str,
) -> List[Dict]:
    """Build candidates using visual + audio features for SVM scoring."""
    frames = extract_frames(video_path, work_dir, FPS)
    if len(frames) < 2:
        return []

    diff = frame_diff_scores(frames)
    color_hist = color_histogram_features(frames)
    edges = edge_density(frames)

    diff_norm = diff / max(1.0, np.max(diff))
    color_norm = color_hist / max(1.0, np.max(color_hist))
    edge_norm = edges / max(1.0, np.max(edges))

    interval_s = 1.0 / FPS
    total_dur = len(frames) * interval_s

    boundaries = [0]
    threshold = np.percentile(diff, 70)
    for i in range(1, len(diff)):
        if diff[i] > threshold:
            boundaries.append(i)
    if boundaries[-1] != len(frames):
        boundaries.append(len(frames))

    rms = audio_rms_per_window(audio_path, window_s=1.0, sr=16000)
    aud_norm = rms / max(1.0, np.max(rms)) if len(rms) > 0 else np.array([])

    features_list = []
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

        f_viz_diff = float(np.mean(diff_norm[i0:i1]))
        f_color = float(np.mean(color_norm[i0:i1]))
        f_edge = float(np.mean(edge_norm[i0:i1]))

        aud_start = int(start_s)
        aud_end = min(int(end_s) + 1, len(aud_norm)) if len(aud_norm) > 0 else 0
        f_audio = float(np.mean(aud_norm[aud_start:aud_end])) if aud_end > aud_start else 0.5

        features_list.append([f_viz_diff, f_color, f_edge, f_audio])
        segments.append({
            "segment_id": f"{video_id}_seg_{seg_idx:04d}",
            "start_s": round(start_s, 2),
            "end_s": round(end_s, 2),
        })

    if len(features_list) < 5:
        for s in segments:
            s["score"] = 0.5
        return segments

    X = np.array(features_list)
    combined = np.mean(X, axis=1)
    n_pos = max(1, int(len(X) * POSITIVE_RATIO))
    n_neg = max(1, int(len(X) * NEGATIVE_RATIO))
    pos_idx = np.argsort(combined)[-n_pos:]
    neg_idx = np.argsort(combined)[:n_neg]
    y = np.zeros(len(X))
    y[pos_idx] = 1
    y[neg_idx] = 0

    mask = np.array([i in pos_idx or i in neg_idx for i in range(len(X))])
    X_train, y_train = X[mask], y[mask]

    if len(np.unique(y_train)) < 2:
        for s in segments:
            s["score"] = round(float(combined[i]), 4)
        return segments

    from sklearn.svm import LinearSVC
    clf = LinearSVC(C=1.0, max_iter=1000, dual="auto", random_state=42)
    clf.fit(X_train, y_train)
    scores = clf.decision_function(X)

    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    for i, s in enumerate(segments):
        s["score"] = round(float(scores[i]), 4)
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
        candidates = build_candidates_svm(str(vp), video_work, str(vp), vid)
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
        "baseline": "SVM-Based",
        "n_videos": len(results),
        "avg_precision": round(float(np.mean([r["precision"] for r in results])), 4),
        "avg_recall": round(float(np.mean([r["recall"] for r in results])), 4),
        "avg_f1": round(float(np.mean([r["f1_score"] for r in results])), 4),
        "avg_runtime_s": round(float(np.mean([r["runtime_s"] for r in results])), 2),
        "target_duration_s": TARGET_DURATION,
        "fps": FPS,
    }
    save_results(output_dir, "SVM-Based", results, summary)


def main():
    parser = argparse.ArgumentParser(description="SVM-Based Video Highlighting")
    parser.add_argument("--data", type=str, default="experiments/data/vlog/")
    parser.add_argument("--output", type=str, default="experiments/output/baselines/svm_based/")
    args = parser.parse_args()
    run_baseline(args.data, args.output)


if __name__ == "__main__":
    main()
