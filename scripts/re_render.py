#!/usr/bin/env python
"""
Re-render edited videos from saved edit plans and source video files.
No CLIP/Whisper/LLM needed — reads plans/*.json and renders with ffmpeg.

Usage:
    python scripts/re_render.py                  # Re-render all 22 videos
    python scripts/re_render.py --video-id 442d84071b0a   # Single video
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.render_ffmpeg import FFmpegRenderer, RenderResult
from experiments.llm_planner import (
    EditDecision, EditSegment, SyncAnchor, SubtitleSpec, AudioMixSpec,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("re_render")

DEFAULT_OUTPUT_DIR = "experiments/output"
DEFAULT_RENDERED_DIR = "rendered"


def _find_output_dir(base: str = DEFAULT_OUTPUT_DIR) -> Path:
    for candidate in [Path(base), Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / base]:
        e2e = candidate / "e2e_results.json"
        plans = candidate / "plans"
        if e2e.exists() and plans.exists():
            return candidate.resolve()
    return Path(base).resolve()


def load_e2e_results(output_dir: Path) -> List[Dict]:
    path = output_dir / "e2e_results.json"
    if not path.exists():
        logger.error(f"e2e_results.json not found at {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_plan(output_dir: Path, video_id: str) -> Optional[EditDecision]:
    plan_path = output_dir / "plans" / f"{video_id}_plan.json"
    if not plan_path.exists():
        logger.warning(f"Plan not found: {plan_path}")
        return None
    with open(plan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = []
    for s in data.get("segments", []):
        sync = None
        if s.get("sync_anchor"):
            sync = SyncAnchor(
                video_time=s["sync_anchor"].get("video_time", 0),
                audio_time=s["sync_anchor"].get("audio_time", 0),
            )
        segments.append(EditSegment(
            segment_id=s.get("segment_id", ""),
            source_start=s.get("source_start", 0),
            source_end=s.get("source_end", 0),
            target_start=s.get("target_start", 0),
            target_end=s.get("target_end", 0),
            transition=s.get("transition", "cut"),
            sync_anchor=sync,
            importance=s.get("importance", 0.5),
        ))

    subtitle = None
    if data.get("subtitle") and data["subtitle"].get("enabled"):
        subtitle = SubtitleSpec(**{k: v for k, v in data["subtitle"].items()
                                    if k in SubtitleSpec.__dataclass_fields__})

    audio_mix = None
    if data.get("audio_mix") and data["audio_mix"].get("narration_enabled"):
        audio_mix = AudioMixSpec(**{k: v for k, v in data["audio_mix"].items()
                                     if k in AudioMixSpec.__dataclass_fields__})

    return EditDecision(
        request_id=data.get("request_id", ""),
        target_duration=data.get("target_duration", 60.0),
        segments=segments, subtitle=subtitle, audio_mix=audio_mix,
        render_backend=data.get("render_backend", "ffmpeg"),
        notes=data.get("notes", ""),
        revision_count=data.get("revision_count", 0),
        validation_passed=data.get("validation_passed", False),
        validation_errors=data.get("validation_errors", []),
    )


def main():
    parser = argparse.ArgumentParser(description="Re-render videos from saved edit plans")
    parser.add_argument("--output", type=str, default="", help="Output directory with e2e_results.json and plans/")
    parser.add_argument("--video-id", type=str, default="", help="Re-render a single video by ID")
    args = parser.parse_args()

    output_dir = _find_output_dir(args.output or "")
    rendered_dir = output_dir / DEFAULT_RENDERED_DIR
    rendered_dir.mkdir(parents=True, exist_ok=True)

    e2e_results = load_e2e_results(output_dir)

    if args.video_id:
        e2e_results = [r for r in e2e_results if r.get("video_id") == args.video_id]
        if not e2e_results:
            logger.error(f"Video {args.video_id} not found in e2e_results.json")
            sys.exit(1)

    renderer = FFmpegRenderer(str(rendered_dir))
    success_count = 0
    skipped_count = 0
    fail_count = 0

    for i, result in enumerate(e2e_results):
        video_id = result.get("video_id", f"unknown_{i}")
        video_path = result.get("video_path", "")
        output_path = str(rendered_dir / f"{video_id}_edited.mp4")

        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            logger.info(f"[{i + 1}/{len(e2e_results)}] {video_id}: already rendered, skipping")
            skipped_count += 1
            continue

        if not os.path.exists(video_path):
            logger.warning(f"[{i + 1}/{len(e2e_results)}] {video_id}: source video not found at {video_path}")
            fail_count += 1
            continue

        plan = load_plan(output_dir, video_id)
        if plan is None:
            fail_count += 1
            continue

        logger.info(f"[{i + 1}/{len(e2e_results)}] {video_id}: re-rendering ({len(plan.segments)} segments)...")
        render_result = renderer.render(plan, video_path, output_path=output_path, mock=False)

        if render_result.success:
            logger.info(f"  [{video_id}] OK -> {output_path}")
            success_count += 1
        else:
            logger.error(f"  [{video_id}] FAIL: {render_result.error_message}")
            fail_count += 1

    logger.info(f"\n===== Done: {success_count} OK, {fail_count} failed, {skipped_count} skipped =====")
    logger.info(f"Rendered videos: {rendered_dir}")
    for mp4 in sorted(rendered_dir.glob("*_edited.mp4")):
        size_mb = mp4.stat().st_size / 1024 / 1024
        logger.info(f"  {mp4.name:40s} {size_mb:6.1f} MB")


if __name__ == "__main__":
    main()
