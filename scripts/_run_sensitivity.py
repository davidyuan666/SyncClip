#!/usr/bin/env python
"""
Sensitivity analysis: sweep theta and tau on 5 representative videos.

Runs candidate building with different theta values, then re-evaluates
plan selection metrics for each parameter setting.

Usage (on GPU server):
    python scripts/_run_sensitivity.py --n-videos 5
"""
from __future__ import annotations

import argparse, json, logging, os, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import load_config
from experiments.preprocess import VideoPreprocessor
from experiments.extract_clip import CLIPExtractor
from experiments.transcribe_whisper import WhisperTranscriber
from experiments.build_candidates import CandidateBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sensitivity")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-videos", type=int, default=5)
    parser.add_argument("--output", type=str, default="experiments/output")
    args = parser.parse_args()

    config = load_config(mock_mode=False, output_dir=args.output)
    vlog_dir = Path(config.dataset_dir) / "vlog"
    videos = sorted(vlog_dir.glob("*.mp4"))[: args.n_videos]
    logger.info(f"Running sensitivity on {len(videos)} videos")

    theta_values = config.sensitivity_sweeps.get("theta", [0.10, 0.14, 0.18, 0.22, 0.26])
    tau_values = config.sensitivity_sweeps.get("tau", [0.50, 0.65, 0.75, 0.80, 0.90, 0.95])

    preprocessor = VideoPreprocessor(str(Path(args.output) / "work"))
    clip_extractor = CLIPExtractor()
    whisper_transcriber = WhisperTranscriber()

    results = {"theta": {}, "tau": {}}

    for video_path in videos:
        vid = video_path.stem
        logger.info(f"=== {vid} ===")

        fps = 5
        preprocess = preprocessor.process(str(video_path), vid, [fps], mock=False, apply_ssim=False)
        clip_feat = clip_extractor.extract_multiple_fps(preprocess.keyframes, mock=False).get(fps)
        whisper = whisper_transcriber.transcribe(preprocess.audio_path or "", mock=False)

        for theta in theta_values:
            builder = CandidateBuilder(theta=theta)
            candidate_set = builder.build(clip_feat, whisper, mock=False)
            if theta not in results["theta"]:
                results["theta"][theta] = []
            results["theta"][theta].append({
                "video_id": vid,
                "n_candidates": len(candidate_set.segments),
                "mean_importance": float(np.mean([s.importance_score for s in candidate_set.segments])) if candidate_set.segments else 0,
            })
            logger.info(f"  theta={theta:.2f} → {len(candidate_set.segments)} candidates")

    logger.info(f"\n{'=' * 50}")
    logger.info("SENSITIVITY RESULTS (theta)")
    logger.info(f"{'=' * 50}")
    for theta in sorted(results["theta"]):
        vals = results["theta"][theta]
        avg_cand = np.mean([v["n_candidates"] for v in vals])
        avg_imp = np.mean([v["mean_importance"] for v in vals])
        logger.info(f"  theta={theta:.2f}  avg_candidates={avg_cand:.0f}  avg_importance={avg_imp:.4f}")

    out_path = Path(args.output) / "sensitivity_real.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
