"""Shared utilities for baseline implementations."""
import json, logging, os, subprocess, tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def extract_frames(video_path: str, work_dir: str, fps: int = 5) -> List[np.ndarray]:
    """Extract frames from video via ffmpeg at given fps. Returns RGB numpy arrays."""
    frames_dir = Path(work_dir) / f"frames_{fps}fps"
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "frame_%05d.jpg")

    logger.info(f"    ffmpeg: extracting @{fps}fps...")
    try:
        import ffmpeg
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.filter(stream, "fps", fps=fps)
        stream = ffmpeg.output(stream, pattern, start_number=0)
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=False, capture_stderr=False)
    except Exception:
        cmd = ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", "-q:v", "2", pattern, "-y"]
        subprocess.run(cmd, timeout=300)

    from PIL import Image
    frames = []
    for fp in sorted(frames_dir.glob("frame_*.jpg")):
        try:
            frames.append(np.array(Image.open(fp).convert("RGB"), dtype=np.float32))
        except Exception:
            continue
    logger.info(f"    extracted {len(frames)} frames @{fps}fps")
    return frames


def frame_diff_scores(frames: List[np.ndarray]) -> np.ndarray:
    """Compute per-frame visual change via mean RGB difference."""
    n = len(frames)
    scores = np.zeros(n)
    for i in range(1, n):
        scores[i] = float(np.mean(np.abs(frames[i] - frames[i - 1])))
    return scores


def audio_rms_per_window(audio_path: str, window_s: float, sr: int = 16000) -> np.ndarray:
    """Extract audio RMS energy per window (window_s seconds)."""
    if not audio_path or not os.path.exists(audio_path):
        return np.array([])

    tmp_wav = os.path.join(tempfile.gettempdir(), "__baseline_audio.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-i", audio_path, "-vn", "-ar", str(sr), "-ac", "1", tmp_wav, "-y"],
            capture_output=True, timeout=30,
        )
        import scipy.io.wavfile as wav
        rate, data = wav.read(tmp_wav)
        if rate != sr:
            import scipy.signal
            data = scipy.signal.resample(data, int(len(data) * sr / rate))
        data = data.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
    except Exception:
        return np.array([])
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    win_samples = int(window_s * sr)
    n_windows = max(1, len(data) // win_samples)
    rms = np.array([np.sqrt(np.mean(data[i * win_samples:(i + 1) * win_samples] ** 2))
                     for i in range(n_windows)])
    return rms


def color_histogram_features(frames: List[np.ndarray]) -> np.ndarray:
    """Compute HSV histogram variance for each frame."""
    feats = np.zeros(len(frames))
    for i, f in enumerate(frames):
        from PIL import Image
        img = Image.fromarray(f.astype(np.uint8)).convert("HSV")
        h = np.array(img).reshape(-1, 3).astype(np.float32)
        hist = np.histogram(h[:, 0], bins=16, range=(0, 256))[0]
        feats[i] = float(np.var(hist)) / max(1, np.mean(hist))
    return feats


def edge_density(frames: List[np.ndarray]) -> np.ndarray:
    """Sobel edge density for each frame."""
    from scipy import ndimage
    dens = np.zeros(len(frames))
    for i, f in enumerate(frames):
        gray = np.mean(f, axis=2).astype(np.float32)
        sx = ndimage.sobel(gray, axis=0)
        sy = ndimage.sobel(gray, axis=1)
        dens[i] = float(np.mean(np.sqrt(sx ** 2 + sy ** 2)))
    return dens


def greedy_segment_selection(
    scored_segments: List[Dict],
    target_duration: float = 60.0,
    min_gap: float = 0.5,
) -> List[Dict]:
    """Greedy selection: highest score first, under duration constraint."""
    sorted_segs = sorted(scored_segments, key=lambda s: s["score"], reverse=True)
    selected = []
    current_dur = 0.0

    for seg in sorted_segs:
        dur = seg["end_s"] - seg["start_s"]
        if dur < 0.3:
            continue
        if current_dur + dur > target_duration + 2:
            continue
        if selected and seg["start_s"] - selected[-1]["source_end"] < min_gap:
            continue
        selected.append({
            "segment_id": seg["segment_id"],
            "source_start": round(seg["start_s"], 2),
            "source_end": round(seg["end_s"], 2),
            "importance": round(seg["score"], 4),
            "transition": "cut",
        })
        current_dur += dur

    return selected


def compute_segment_metrics(
    predictions: List[Dict],
    references: List[Dict],
    iou_threshold: float = 0.3,
) -> Tuple[float, float, float]:
    """Temporal IoU matching for P/R/F1."""
    tp, fp = 0, 0
    matched = set()
    for p in predictions:
        found = False
        p_start = p.get("source_start", p.get("start", 0))
        p_end = p.get("source_end", p.get("end", 0))
        for j, r in enumerate(references):
            if j in matched:
                continue
            r_start = r.get("source_start", r.get("start", r.get("start_s", 0)))
            r_end = r.get("source_end", r.get("end", r.get("end_s", 0)))
            inter = max(0, min(p_end, r_end) - max(p_start, r_start))
            union_a = p_end - p_start
            union_b = r_end - r_start
            iou = inter / (union_a + union_b - inter) if (union_a + union_b - inter) > 0 else 0
            if iou >= iou_threshold:
                tp += 1
                matched.add(j)
                found = True
                break
        if not found:
            fp += 1
    fn = len(references) - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def save_results(output_dir: str, name: str, results: List[Dict], summary: Dict):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\n{'=' * 50}")
    logger.info(f"[{name}] Baseline Summary")
    logger.info(f"{'=' * 50}")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
