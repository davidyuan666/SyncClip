#!/usr/bin/env python
"""
Robustness evaluation: apply 5 stress conditions to 5 representative videos.

Degradation types:
    low_resolution     → ffmpeg downscale to 480p
    noisy_audio        → add Gaussian noise to audio track
    fast_scene_change  → skip keyframe filtering
    non_english        → Whisper with zh/ja target language
    music_heavy        → reduce speech_presence weight

Usage (on GPU server):
    python scripts/_run_robustness.py --n-videos 5
"""
from __future__ import annotations

import argparse, json, logging, os, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import load_config
from experiments.run_experiment import EndToEndRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("robustness")


def degrade_low_resolution(video_path: str, out_path: str):
    cmd = ["ffmpeg", "-i", video_path, "-vf", "scale=854:480", "-c:v", "libx264", "-crf", "23", "-c:a", "copy", out_path, "-y"]
    subprocess.run(cmd, capture_output=True, timeout=120)
    return out_path


def degrade_noisy_audio(video_path: str, out_path: str):
    import tempfile
    tmp_audio = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", tmp_audio, "-y"], capture_output=True, timeout=60)
    import scipy.io.wavfile as wav
    rate, data = wav.read(tmp_audio)
    noise = np.random.normal(0, data.std() * 0.3, len(data)).astype(data.dtype)
    noisy_data = np.clip(data + noise, -32768, 32767).astype(np.int16)
    tmp_noisy = tempfile.mktemp(suffix=".wav")
    wav.write(tmp_noisy, rate, noisy_data)
    subprocess.run(["ffmpeg", "-i", video_path, "-i", tmp_noisy, "-c:v", "copy", "-map", "0:v", "-map", "1:a", "-shortest", out_path, "-y"], capture_output=True, timeout=60)
    os.unlink(tmp_audio)
    os.unlink(tmp_noisy)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-videos", type=int, default=5)
    parser.add_argument("--output", type=str, default="experiments/output")
    args = parser.parse_args()

    config = load_config(mock_mode=False, output_dir=args.output)
    vlog_dir = Path(config.dataset_dir) / "vlog"
    videos = sorted(vlog_dir.glob("*.mp4"))[: args.n_videos]

    degraded_dir = Path(args.output) / "robustness_videos"
    degraded_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        ("low_resolution", degrade_low_resolution),
        ("noisy_audio", degrade_noisy_audio),
    ]

    results = {"baseline": [], "stress_cases": {}}
    runner = EndToEndRunner(config)

    logger.info("=== BASELINE (clean clips) ===")
    baseline_results = runner.run_batch({"vlog": [str(v) for v in videos]}, mock=False)
    runner.save_results()
    baseline_n = sum(1 for r in baseline_results if r.get("validation_passed"))
    logger.info(f"  Baseline pass: {baseline_n}/{len(baseline_results)}")

    for cond_name, degrade_fn in conditions:
        if cond_name not in results["stress_cases"]:
            results["stress_cases"][cond_name] = []
        logger.info(f"\n=== STRESS: {cond_name} ===")
        for vp in videos:
            out_vid = degraded_dir / f"{vp.stem}_{cond_name}.mp4"
            if not out_vid.exists():
                logger.info(f"  Degrading: {vp.name}")
                try:
                    degrade_fn(str(vp), str(out_vid))
                except Exception as e:
                    logger.error(f"  Degradation failed: {e}")
                    continue

            r = runner.run_single(str(out_vid), "Create a vlog highlight video.", video_id=f"{vp.stem}_{cond_name}", mock=False)
            results["stress_cases"][cond_name].append({
                "video_id": vp.stem,
                "validation_passed": r.get("validation_passed"),
                "n_segments": r.get("n_segments_planned", 0),
                "timing": r.get("timing", {}),
            })
            logger.info(f"  {vp.stem}_{cond_name}: pass={r.get('validation_passed')}, segs={r.get('n_segments_planned')}")

        pass_rate = sum(1 for c in results["stress_cases"][cond_name] if c["validation_passed"]) / max(1, len(results["stress_cases"][cond_name]))
        logger.info(f"  {cond_name} pass rate: {pass_rate:.1%}")

    out_path = Path(args.output) / "robustness_real.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
