"""
End-to-end experiment orchestrator for the full SyncCLIPAgent pipeline:
  preprocess → CLIP → Whisper → build_candidates → align → LLM_plan → validate → render

Outputs:
  - Processed videos with edit decisions
  - RQ1-RQ4 CSV result files
  - Validation reports
  - Runtime profiling

Usage:
    python -m experiments.run_experiment --video-dir ./data/videos --output ./results
    python -m experiments.run_experiment --video data/video.mp4 --request "create 60s highlight" --mock
    python -m experiments.run_experiment --video-dir ./data --request-file requests.json --all
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.config import ExperimentConfig, load_config
from experiments.preprocess import VideoPreprocessor, PreprocessResult
from experiments.extract_clip import CLIPExtractor, CLIPFeatures
from experiments.transcribe_whisper import WhisperTranscriber, WhisperResult
from experiments.build_candidates import CandidateBuilder, CandidateSet
from experiments.llm_planner import LLMPlanner, EditDecision, EditSegment
from experiments.validate_plan import PlanValidator, validate_and_revise, ValidationReport
from experiments.render_ffmpeg import FFmpegRenderer, RenderResult
from experiments.csv_output import CSVAggregator, save_all_csvs
from src.models.cross_modal_projection import CrossModalProjection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logger = logging.getLogger("run_experiment")


class EndToEndRunner:
    """Full pipeline: raw video → edit decision → rendered video → CSV metrics."""

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or load_config()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor = VideoPreprocessor(str(self.output_dir / "work"))
        self.clip_extractor = CLIPExtractor()
        self.whisper_transcriber = WhisperTranscriber()
        self.candidate_builder = CandidateBuilder(theta=self.config.parameters["theta"])
        self.llm_planner = LLMPlanner()
        self.validator = PlanValidator()
        self.renderer = FFmpegRenderer()
        self.csv_agg = CSVAggregator()

        self.all_results: List[Dict] = []

    def run_single(
        self, video_path: str, user_request: str = "",
        video_id: str = "", request_id: str = "",
        target_duration: float = 60.0, fps_options: Optional[List[int]] = None,
        mock: bool = False,
    ) -> Dict:
        t0 = time.time()
        video_id = video_id or Path(video_path).stem
        request_id = request_id or f"req_{video_id[:8]}"
        user_request = user_request or "Create an engaging highlight video from the most important moments."
        fps_options = fps_options or [5]

        timing = {}
        logger.info(f"=== [{video_id}] Starting end-to-end pipeline ===")

        # 1. Preprocess
        logger.info(f"  [{video_id}] [1/8] Preprocessing frames (ffmpeg)...")
        t = time.time()
        preprocess = self.preprocessor.process(video_path, video_id, fps_options, mock=mock, apply_ssim=False)
        timing["preprocess_s"] = round(time.time() - t, 2)
        logger.info(f"  [{video_id}] [1/8] done: {len(preprocess.keyframes.get(5, []))} frames @5fps ({timing['preprocess_s']}s)")

        # 2. CLIP
        logger.info(f"  [{video_id}] [2/8] CLIP ViT-B/32 encoding...")
        t = time.time()
        clip_by_fps = self.clip_extractor.extract_multiple_fps(preprocess.keyframes, mock=mock)
        timing["clip_s"] = round(time.time() - t, 2)
        clip_5fps = clip_by_fps.get(5, clip_by_fps.get(1))
        logger.info(f"  [{video_id}] [2/8] done: {len(clip_5fps)} embeddings ({timing['clip_s']}s)")
        self._cleanup_frames(preprocess)

        # 3. Whisper
        logger.info(f"  [{video_id}] [3/8] Whisper large-v3 transcribing...")
        t = time.time()
        audio_path = preprocess.audio_path or ""
        whisper = self.whisper_transcriber.transcribe(audio_path, mock=mock)
        timing["whisper_s"] = round(time.time() - t, 2)
        logger.info(f"  [{video_id}] [3/8] done: {len(whisper)} segments ({timing['whisper_s']}s)")
        self._cleanup_audio(preprocess)

        # 4. Build candidates
        logger.info(f"  [{video_id}] [4/8] Building candidates...")
        t = time.time()
        candidates_by_fps = {}
        for fps, clip_feat in clip_by_fps.items():
            candidates_by_fps[fps] = self.candidate_builder.build(clip_feat, whisper, mock=mock)
        timing["candidates_s"] = round(time.time() - t, 2)
        candidates_5fps = candidates_by_fps.get(5, list(candidates_by_fps.values())[0] if candidates_by_fps else None)
        logger.info(f"  [{video_id}] [4/8] done: {len(candidates_5fps.segments) if candidates_5fps else 0} segments ({timing['candidates_s']}s)")

        if not candidates_5fps or not candidates_5fps.segments:
            return self._empty_result(video_id, request_id, timing, "No candidates generated")

        # 5. Cross-modal projection
        t = time.time()
        vis_embs = np.array([s.visual_embedding for s in candidates_5fps.segments if s.visual_embedding is not None])
        aud_embs = np.array([s.audio_embedding for s in candidates_5fps.segments if s.audio_embedding is not None])
        if len(vis_embs) > 1 and len(aud_embs) > 1:
            projector = CrossModalProjection(d_common=self.config.model["common_projection_dim"])
            projector.fit(vis_embs, aud_embs)
        timing["projection_s"] = round(time.time() - t, 2)

        # 6. LLM planning
        logger.info(f"  [{video_id}] [6/8] LLM planning (DeepSeek API)...")
        t = time.time()
        plan = self.llm_planner.plan(candidates_5fps, user_request, request_id, target_duration, mock=mock)
        timing["llm_plan_s"] = round(time.time() - t, 2)
        logger.info(f"  [{video_id}] [6/8] done: {len(plan.segments)} segments, {plan.total_target_duration:.1f}s ({timing['llm_plan_s']}s)")

        # 7. Validate + revise
        logger.info(f"  [{video_id}] [7/8] Validating plan...")
        t = time.time()
        plan, report = validate_and_revise(plan, candidates_5fps, self.llm_planner, self.validator, mock=mock)
        timing["validate_s"] = round(time.time() - t, 2)
        logger.info(f"  [{video_id}] [7/8] done: {'PASS' if report.passed else 'FAIL'} (revisions={plan.revision_count}, {timing['validate_s']}s)")

        # 8. Render
        logger.info(f"  [{video_id}] [8/8] FFmpeg rendering...")
        t = time.time()
        output_video = str(self.output_dir / "rendered" / f"{video_id}_edited.mp4")
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        render = self.renderer.render(plan, video_path, output_path=output_video, mock=mock)
        timing["render_s"] = round(time.time() - t, 2)
        logger.info(f"  [{video_id}] [8/8] done: {'OK' if render.success else 'FAIL'} ({timing['render_s']}s)")

        total_s = round(time.time() - t0, 2)
        timing["total_s"] = total_s

        result = {
            "video_id": video_id, "request_id": request_id,
            "video_path": video_path, "request": user_request,
            "target_duration_s": target_duration,
            "n_candidates": len(candidates_5fps.segments) if candidates_5fps else 0,
            "n_segments_planned": len(plan.segments),
            "planned_duration_s": plan.total_target_duration,
            "validation_passed": report.passed,
            "revision_count": plan.revision_count,
            "render_success": render.success,
            "render_output": render.output_path,
            "hard_errors": report.hard_errors,
            "soft_warnings": report.soft_warnings,
            "timing": timing,
        }

        self._save_plan(plan, video_id)
        self._save_report(report, video_id)
        self.all_results.append(result)
        logger.info(f"  [{video_id}] Done in {total_s}s")
        return result

    def run_batch(
        self, video_paths: Dict[str, List[str]], requests: Optional[Dict[str, str]] = None,
        mock: bool = False,
    ) -> List[Dict]:
        requests = requests or {}
        results = []
        total = sum(len(p) for p in video_paths.values())
        count = 0
        for genre, paths in video_paths.items():
            for vp in paths:
                count += 1
                vid = Path(vp).stem
                req = requests.get(genre, f"Create a {genre} highlight video.")
                logger.info(f"[{count}/{total}] Processing video: {vid}")
                result = self.run_single(vp, req, video_id=vid, mock=mock)
                result["genre"] = genre
                results.append(result)
        return results

    def save_results(self):
        save_path = self.output_dir / "e2e_results.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)

        csv_paths = save_all_csvs(self.all_results, str(self.output_dir))
        for name, path in csv_paths.items():
            logger.info(f"  CSV saved: {name} -> {path}")

        summary_path = self.output_dir / "e2e_summary.json"
        summary = self._compute_summary()
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return self.all_results

    def _save_plan(self, plan: EditDecision, video_id: str):
        path = self.output_dir / "plans" / f"{video_id}_plan.json"
        os.makedirs(path.parent, exist_ok=True)
        plan.save(str(path))

    def _save_report(self, report: ValidationReport, video_id: str):
        path = self.output_dir / "reports" / f"{video_id}_validation.json"
        os.makedirs(path.parent, exist_ok=True)
        report.save(str(path))

    def _compute_summary(self) -> Dict:
        results = self.all_results
        if not results:
            return {}

        n = len(results)
        pass_rate = sum(1 for r in results if r.get("validation_passed")) / n
        render_ok = sum(1 for r in results if r.get("render_success")) / n
        avg_revisions = np.mean([r.get("revision_count", 0) for r in results])
        avg_total_s = np.mean([r.get("timing", {}).get("total_s", 0) for r in results])
        avg_candidates = np.mean([r.get("n_candidates", 0) for r in results])
        avg_planned = np.mean([r.get("n_segments_planned", 0) for r in results])

        return {
            "n_videos": n, "validation_pass_rate": round(pass_rate, 3),
            "render_success_rate": round(render_ok, 3),
            "avg_revisions": round(float(avg_revisions), 2),
            "avg_total_time_s": round(float(avg_total_s), 1),
            "avg_candidates": round(float(avg_candidates), 1),
            "avg_segments_planned": round(float(avg_planned), 1),
        }

    def _empty_result(self, vid: str, rid: str, timing: Dict, error: str) -> Dict:
        return {
            "video_id": vid, "request_id": rid, "error": error,
            "validation_passed": False, "render_success": False,
            "timing": timing,
        }

    def _cleanup_frames(self, preprocess: "PreprocessResult"):
        from experiments.preprocess import PreprocessResult
        for fps_dir in Path(preprocess.work_dir).glob("frames_*fps"):
            try:
                import shutil
                shutil.rmtree(str(fps_dir), ignore_errors=True)
            except Exception:
                pass

    def _cleanup_audio(self, preprocess: "PreprocessResult"):
        try:
            audio = Path(preprocess.audio_path) if preprocess.audio_path else None
            if audio and audio.exists():
                audio.unlink()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="SyncCLIPAgent End-to-End Experiment Runner")
    parser.add_argument("--video", type=str, help="Single video file path")
    parser.add_argument("--video-dir", type=str, help="Directory of video files (organized by genre)")
    parser.add_argument("--request", type=str, default="", help="User editing request text")
    parser.add_argument("--request-file", type=str, help="JSON file with per-genre editing requests")
    parser.add_argument("--target-duration", type=float, default=60.0, help="Target video duration in seconds")
    parser.add_argument("--fps", type=int, nargs="+", default=[1, 2, 3, 5], help="Frame sampling rates")
    parser.add_argument("--output", type=str, default="experiments/output", help="Output directory")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--real", action="store_true", help="Use real models (requires GPU)")
    args = parser.parse_args()

    config = load_config(mock_mode=args.mock and not args.real, output_dir=args.output)

    runner = EndToEndRunner(config)

    if args.video:
        runner.run_single(args.video, args.request, target_duration=args.target_duration,
                          fps_options=args.fps, mock=config.mock_mode)
    elif args.video_dir:
        video_paths = _scan_video_dir(args.video_dir)
        requests = {}
        if args.request_file:
            with open(args.request_file, "r") as f:
                requests = json.load(f)
        runner.run_batch(video_paths, requests, mock=config.mock_mode)
    else:
        logger.info("No --video or --video-dir specified. Running single mock demo.")
        import tempfile
        mock_video = os.path.join(tempfile.gettempdir(), "mock_demo_video.mp4")
        Path(mock_video).touch()
        runner.run_single(mock_video, "Create a 60-second highlight reel.", mock=True)

    runner.save_results()
    logger.info(f"Results saved to {config.output_dir}")


def _scan_video_dir(dir_path: str) -> Dict[str, List[str]]:
    paths: Dict[str, List[str]] = {}
    root = Path(dir_path)

    genres = ["vlog"]
    for genre in genres:
        genre_dir = root / genre
        if genre_dir.exists():
            paths[genre] = sorted(str(p) for p in genre_dir.glob("*.mp4"))

    if not paths:
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]:
            for p in sorted(root.glob(ext)):
                paths.setdefault("vlog", []).append(str(p))

    return paths


if __name__ == "__main__":
    main()
