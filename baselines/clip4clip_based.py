"""
CLIP4Clip-Based Video Highlighting Baseline.

Uses CLIP (ViT-B/32) for text-driven video segment retrieval.
Based on: Luo et al., "CLIP4Clip: An Empirical Study of CLIP for End to End
Video Clip Retrieval", Neurocomputing, 2022.

Algorithm:
  1. Extract frames @5fps via ffmpeg.
  2. Encode frames with CLIP ViT-B/32 (visual).
  3. Encode editing query with CLIP (text).
  4. Frame-level text-visual cosine similarity.
  5. Build shot-boundary candidate segments.
  6. Score segments by mean frame-query similarity.
  7. Greedy selection to 60s target.
  8. Evaluate against heuristic GT (top-K importance candidates).
  9. Output results.json + summary.json.

Usage:
    python -m baselines.clip4clip_based
    python -m baselines.clip4clip_based --data experiments/data/vlog/
    python -m baselines.clip4clip_based --output experiments/output/baselines/clip4clip/
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

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clip4clip_based")

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_DIM = 768
TARGET_DURATION = 60.0
FPS = 5
QUERY_TEXT = "Create a vlog highlight video"
QUERY_TEXTS = [
    "Create a vlog highlight video",
    "A highlight video of vlog content",
    "Interesting vlog moments",
]
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

_clip_model = None
_clip_processor = None


def _load_clip():
    """Lazy-load CLIP model and processor."""
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor

    import torch
    from transformers import CLIPModel, CLIPProcessor

    logger.info(f"Loading CLIP ViT-B/32 on {DEVICE}...")
    _clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
    _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    _clip_model.eval()
    logger.info("CLIP model loaded.")
    return _clip_model, _clip_processor


def _encode_frames(frame_paths: List[str]) -> np.ndarray:
    """Encode frames to CLIP visual embeddings. Returns [N, 768] normalized."""
    import torch

    model, processor = _load_clip()

    from PIL import Image

    embeddings = []
    batch_size = 32

    for i in range(0, len(frame_paths), batch_size):
        batch = frame_paths[i : i + batch_size]
        images = []
        valid_indices = []
        for j, fp in enumerate(batch):
            try:
                img = Image.open(fp).convert("RGB")
                images.append(img)
                valid_indices.append(i + j)
            except Exception:
                continue

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        emb = outputs.cpu().numpy().astype(np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        embeddings.append(emb)
        done = min(i + batch_size, len(frame_paths))
        logger.info(f"    CLIP encoding {done}/{len(frame_paths)} frames")

    if not embeddings:
        return np.zeros((0, CLIP_DIM), dtype=np.float32)
    return np.concatenate(embeddings, axis=0)


def _encode_query(texts: List[str]) -> np.ndarray:
    """Encode multiple text queries, return mean embedding [768]."""
    import torch

    model, processor = _load_clip()

    all_embs = []
    for text in texts:
        inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
        emb = outputs.cpu().numpy().astype(np.float32)[0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        all_embs.append(emb)

    mean_emb = np.mean(all_embs, axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
    return mean_emb


def compute_frame_query_similarity(frame_embs: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity between each frame embedding and query embedding."""
    if len(frame_embs) == 0:
        return np.array([], dtype=np.float32)
    sims = np.dot(frame_embs, query_emb)
    return np.clip(sims, 0.0, 1.0)


def build_candidates_clip4clip(
    frames: List[np.ndarray],
    frame_paths: List[str],
    video_id: str,
    fps: int = FPS,
) -> List[Dict]:
    """Build candidate segments with CLIP4Clip text-visual similarity scores."""
    import torch

    if len(frames) < 2:
        return []

    _load_clip()

    interval_s = 1.0 / fps

    diff = frame_diff_scores(frames)
    threshold = np.percentile(diff, 70)
    boundaries = [0]
    for i in range(1, len(diff)):
        if diff[i] > threshold:
            boundaries.append(i)
    if boundaries[-1] != len(frames):
        boundaries.append(len(frames))

    logger.info(f"    Encoding {len(frame_paths)} frames with CLIP...")
    frame_embs = _encode_frames(frame_paths)
    if len(frame_embs) == 0:
        logger.warning("    No frame embeddings, falling back to diff-based scoring")
        return _fallback_candidates(diff, boundaries, interval_s, fps, video_id)

    logger.info(f"    Encoding text query: {QUERY_TEXT}")
    query_emb = _encode_query(QUERY_TEXTS)
    frame_sims = compute_frame_query_similarity(frame_embs, query_emb)

    n_embs = len(frame_embs)
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

        si0 = min(i0, n_embs - 1) if n_embs > 0 else i0
        si1 = min(i1, n_embs) if n_embs > 0 else i1
        if si1 > si0:
            text_score = float(np.mean(frame_sims[si0:si1]))
        else:
            text_score = 0.5
        text_score = min(float(text_score), 1.0)

        segments.append({
            "segment_id": f"{video_id}_seg_{seg_idx:04d}",
            "start_s": round(start_s, 2),
            "end_s": round(end_s, 2),
            "text_score": round(text_score, 4),
            "score": round(text_score, 4),
        })

    return segments


def _fallback_candidates(diff, boundaries, interval_s, fps, video_id):
    """Fallback: use frame-diff scores only when CLIP encoding fails."""
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
        score = float(np.mean(diff[i0:i1])) / max(1.0, float(np.mean(diff)))
        score = min(score, 1.0)
        segments.append({
            "segment_id": f"{video_id}_seg_{seg_idx:04d}",
            "start_s": round(start_s, 2),
            "end_s": round(end_s, 2),
            "text_score": round(score, 4),
            "score": round(score, 4),
        })
    return segments


def compute_sync_metrics(plan: List[Dict], ref: List[Dict]) -> Dict:
    """
    Compute temporal synchronization metrics for a single video.
    Returns mean_error_ms, median_error_ms, within_200ms_pct, aligned_f1.
    """
    if not plan or not ref:
        return {"mean_error_ms": float("nan"), "median_error_ms": float("nan"),
                "within_200ms_pct": float("nan"), "aligned_f1": float("nan")}

    errors = []
    for p in plan:
        p_start = p.get("source_start", p.get("start_s", 0))
        best_err = float("inf")
        for r in ref:
            r_start = r.get("source_start", r.get("start_s", 0))
            err = abs(p_start - r_start)
            if err < best_err:
                best_err = err
        if best_err < float("inf"):
            errors.append(best_err * 1000.0)

    if not errors:
        return {"mean_error_ms": float("nan"), "median_error_ms": float("nan"),
                "within_200ms_pct": float("nan"), "aligned_f1": float("nan")}

    errors = np.array(errors)
    mean_err = round(float(np.mean(errors)), 1)
    median_err = round(float(np.median(errors)), 1)
    within_200 = round(float(np.mean(errors <= 200.0)) * 100, 1)

    aligned_plan = [p for i, p in enumerate(plan) if i < len(errors) and errors[i] <= 200.0]
    if aligned_plan:
        p_a, r_a, f1_a = compute_segment_metrics(aligned_plan, ref)
    else:
        f1_a = 0.0

    return {
        "mean_error_ms": mean_err,
        "median_error_ms": median_err,
        "within_200ms_pct": within_200,
        "aligned_f1": round(f1_a, 4),
    }


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
    sync_summary_vals = {"mean_error_ms": [], "median_error_ms": [],
                         "within_200ms_pct": [], "aligned_f1": []}

    for i, vp in enumerate(videos):
        vid = vp.stem
        t0 = time.time()
        logger.info(f"[{i + 1}/{len(videos)}] {vid}")

        video_work = str(work / vid)
        frames = extract_frames(str(vp), video_work, FPS)
        frames_dir = Path(video_work) / f"frames_{FPS}fps"
        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
        frame_paths_str = [str(f) for f in frame_paths]

        candidates = build_candidates_clip4clip(frames, frame_paths_str, vid)
        plan = greedy_segment_selection(candidates, TARGET_DURATION)

        ref = sorted(candidates, key=lambda x: x["score"], reverse=True)[:len(plan) * 2]
        p, r, f1 = compute_segment_metrics(plan, ref)

        sync = compute_sync_metrics(plan, ref)

        elapsed = round(time.time() - t0, 2)
        results.append({
            "video_id": vid,
            "n_candidates": len(candidates),
            "n_selected": len(plan),
            "precision": p, "recall": r, "f1_score": f1,
            "runtime_s": elapsed,
            **sync,
        })
        logger.info(f"  candidates={len(candidates)} segs={len(plan)} P={p:.3f} R={r:.3f} F1={f1:.3f} "
                    f"sync_mean={sync['mean_error_ms']}ms ({elapsed}s)")

        for k in sync_summary_vals:
            sync_summary_vals[k].append(sync[k])

    def _safe_mean(arr):
        valid = [x for x in arr if not (isinstance(x, float) and np.isnan(x))]
        return round(float(np.mean(valid)), 4) if valid else float("nan")

    summary = {
        "baseline": "CLIP4Clip",
        "n_videos": len(results),
        "avg_precision": round(float(np.mean([r["precision"] for r in results])), 4),
        "avg_recall": round(float(np.mean([r["recall"] for r in results])), 4),
        "avg_f1": round(float(np.mean([r["f1_score"] for r in results])), 4),
        "avg_runtime_s": round(float(np.mean([r["runtime_s"] for r in results])), 2),
        "avg_mean_error_ms": _safe_mean(sync_summary_vals["mean_error_ms"]),
        "avg_median_error_ms": _safe_mean(sync_summary_vals["median_error_ms"]),
        "avg_within_200ms_pct": _safe_mean(sync_summary_vals["within_200ms_pct"]),
        "avg_aligned_f1": _safe_mean(sync_summary_vals["aligned_f1"]),
        "target_duration_s": TARGET_DURATION,
        "fps": FPS,
        "clip_model": CLIP_MODEL,
        "query": QUERY_TEXT,
        "device": DEVICE,
    }
    save_results(output_dir, "CLIP4Clip", results, summary)


def main():
    parser = argparse.ArgumentParser(description="CLIP4Clip-Based Video Highlighting")
    parser.add_argument("--data", type=str, default="experiments/data/vlog/")
    parser.add_argument("--output", type=str, default="experiments/output/baselines/clip4clip/")
    args = parser.parse_args()
    run_baseline(args.data, args.output)


if __name__ == "__main__":
    main()
