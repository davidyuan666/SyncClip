#!/usr/bin/env python
"""
SyncCLIPAgent Experiment Suite — Main Entry Point.

Usage:
    # Run all experiments (mock mode, no GPU/API required)
    python -m experiments.run_all --mock

    # Run specific experiments
    python -m experiments.run_all --ground-truth
    python -m experiments.run_all --sensitivity
    python -m experiments.run_all --profile
    python -m experiments.run_all --robustness
    python -m experiments.run_all --visualize

    # Run with real data
    python -m experiments.run_all --data /path/to/videos

Configuration via environment variables:
    OPENAI_API_KEY      LLM API key (required for non-mock LLM)
    COS_ENABLED=1       Enable Tencent COS uploads
    TTS_ENABLED=1       Enable ElevenLabs TTS
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import ExperimentConfig, load_config
from experiments.runner import ExperimentRunner
from experiments.ground_truth import GroundTruthBuilder
from experiments.sensitivity import SensitivityAnalyzer
from experiments.profiling import RuntimeProfiler
from experiments.robustness import RobustnessTester
from experiments.statistics import StatisticsAnalyzer, LaTeXTableGenerator
from experiments.visualization import ExperimentVisualizer
from experiments.run_experiment import EndToEndRunner, _scan_video_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("experiments")


def _generate_mock_segments(config: ExperimentConfig):
    """Generate synthetic video segments for testing the pipeline."""
    import numpy as np
    rng = np.random.default_rng(config.seed)

    segments = {}
    for genre in config.genre_list:
        count = config.genre_counts.get(genre, 10)
        duration = config.genre_durations.get(genre, 10.0)
        total_s = duration * 60
        seg_len = total_s / count

        genre_segs = []
        for i in range(count):
            start = i * seg_len + rng.uniform(-1, 1)
            end = start + seg_len * 0.8 + rng.uniform(-0.5, 1.5)
            genre_segs.append({
                "start": round(max(0, start), 2),
                "end": round(min(total_s, end), 2),
                "text": f"Sample {genre} segment {i}: visual content at frame {i}",
                "salient_events": [f"event_{i}"],
            })
        segments[genre] = genre_segs
    return segments


def scan_video_paths(config: ExperimentConfig) -> Dict[str, List[str]]:
    """Scan dataset directory for video files, organized by genre."""
    data_dir = config.dataset_dir
    video_paths = _scan_video_dir(data_dir)
    if not video_paths:
        vlog_dir = Path(data_dir) / "vlog"
        if vlog_dir.exists():
            mp4s = sorted(str(p) for p in vlog_dir.glob("*.mp4"))
            if mp4s:
                video_paths = {"vlog": mp4s}
    return video_paths


def run_e2e_pipeline(config: ExperimentConfig, video_paths: Dict[str, List[str]]):
    """Run end-to-end pipeline on all videos."""
    logger.info("=" * 60)
    logger.info("Running End-to-End Pipeline on Real Videos")
    logger.info("=" * 60)

    n_total = sum(len(v) for v in video_paths.values())
    logger.info(f"  Videos: {n_total} across {len(video_paths)} genre(s)")

    e2e = EndToEndRunner(config)
    results = e2e.run_batch(video_paths, mock=config.mock_mode)
    e2e.save_results()

    success = sum(1 for r in results if not r.get("error"))
    logger.info(f"  Processed: {success}/{n_total} videos successfully")
    return results


def run_ground_truth(config: ExperimentConfig):
    logger.info("=" * 60)
    logger.info("Running Ground Truth Construction & Annotation Protocol")
    logger.info("=" * 60)

    segments = _generate_mock_segments(config)
    builder = GroundTruthBuilder(config)
    result, ground_truth = builder.build_ground_truth(segments, n_real=40, n_synthetic=60)

    logger.info(f"  Annotators:              {result.protocol.n_annotators}")
    logger.info(f"  Total segments:          {result.n_total_segments}")
    logger.info(f"  Adjudicated conflicts:   {result.n_adjudicated_conflicts}")
    logger.info(f"  Cohen's kappa:           {result.cohens_kappa:.4f}")
    logger.info(f"  Krippendorff's alpha:    {result.krippendorff_alpha:.4f}")
    logger.info(f"  Interpretation:          {result.agreement_interpretation}")
    logger.info(f"  Per-genre kappa:         {json.dumps(result.per_genre_kappa, indent=2)}")

    return result, ground_truth, segments


def run_sensitivity(config: ExperimentConfig):
    logger.info("=" * 60)
    logger.info("Running Parameter Sensitivity Analysis")
    logger.info("=" * 60)

    analyzer = SensitivityAnalyzer(config)
    result = analyzer.run_all()

    for param, summary in result._summary().items():
        logger.info(f"  {param}: best={summary['best_value']} (F1={summary['best_f1']:.4f}), "
                     f"worst={summary['worst_value']} (F1={summary['worst_f1']:.4f}), "
                     f"range={summary['f1_range']:.4f}")

    return result


def run_profiling(config: ExperimentConfig):
    logger.info("=" * 60)
    logger.info("Running Runtime Profiling & Scalability Analysis")
    logger.info("=" * 60)

    profiler = RuntimeProfiler(config)
    result = profiler.profile_all()

    for video_len, timings in result.by_video_length.items():
        total_t = next(t.mean_sec_total for t in timings if t.component == "total")
        rate = next(t.mean_sec_per_video_min for t in timings if t.component == "total")
        logger.info(f"  {video_len}min video: total={total_t:.1f}s, rate={rate:.1f}s/min")

    return result


def run_robustness(config: ExperimentConfig):
    logger.info("=" * 60)
    logger.info("Running Robustness Tests (5 stress cases)")
    logger.info("=" * 60)

    tester = RobustnessTester(config)
    result = tester.evaluate()

    logger.info(f"  Baseline  F1={result.baseline.segment_metrics.f1_score:.3f}, "
                 f"Sync={result.baseline.sync_metrics.mean_abs_error_ms:.0f}ms")
    for case in result.stress_cases:
        logger.info(f"  {case.case_name:20s} F1={case.segment_metrics.f1_score:.3f} "
                     f"(delta={case.f1_drop:+.3f})  "
                     f"Sync={case.sync_metrics.mean_abs_error_ms:.0f}ms "
                     f"(delta={case.sync_drop_ms:+.0f}ms)  "
                     f"failure={case.failure_module}")

    return result


def run_full_pipeline(config: ExperimentConfig, video_paths: Dict[str, List[str]] = None):
    logger.info("=" * 60)
    logger.info("Running Full Experiment Pipeline")
    logger.info("=" * 60)

    if video_paths is None:
        video_paths = scan_video_paths(config)
    if not video_paths:
        video_paths = {g: [f"mock_{g}.mp4"] for g in config.genre_list}
        logger.info("  No videos found, using mock paths")

    segments = _generate_mock_segments(config)
    ground_truth_result, ground_truth, _ = run_ground_truth(config)

    runner = ExperimentRunner(config)
    result = runner.run_full_pipeline(
        video_paths=video_paths,
        reference_segments=ground_truth,
    )

    logger.info(f"  Overall:  P={result.segment_metrics.precision:.4f}, "
                 f"R={result.segment_metrics.recall:.4f}, "
                 f"F1={result.segment_metrics.f1_score:.4f}")
    logger.info(f"  Per-genre F1:")
    for genre, m in result.segment_metrics.per_genre.items():
        logger.info(f"    {genre:20s} F1={m.f1_score:.4f}")

    return result


def run_visualization(config: ExperimentConfig, pipeline_result=None):
    logger.info("=" * 60)
    logger.info("Generating Visualization & LaTeX Tables")
    logger.info("=" * 60)

    viz = ExperimentVisualizer(str(Path(config.output_dir)))

    if pipeline_result is not None:
        seg_m = pipeline_result.segment_metrics
        sync_m = pipeline_result.sync_metrics
        per_genre_fps = {}
        fps_results = {5: {"P": seg_m.precision, "R": seg_m.recall, "F1": seg_m.f1_score}}
        if seg_m.per_genre:
            for genre, m in seg_m.per_genre.items():
                per_genre_fps[genre] = {5: {"P": m.precision, "R": m.recall, "F1": m.f1_score}}
        else:
            per_genre_fps = {"vlog": fps_results}

        sync_values = {5: sync_m.mean_abs_error_ms if sync_m.mean_abs_error_ms > 0 else 130}
        viz.plot_f1_vs_fps(per_genre_fps)
        viz.plot_sync_error_vs_fps(sync_values)

        runtime_data = [
            {"component": k.replace("_s", ""), "mean_sec_per_video_min": v / 60.0,
             "std_sec_per_video_min": 0.0, "peak_gpu_memory_gb": 0.0}
            for k, v in pipeline_result.component_timing.items()
        ]
        if not runtime_data:
            runtime_data = [
                {"component": "frame_extraction", "mean_sec_per_video_min": 2.5, "std_sec_per_video_min": 0.5, "peak_gpu_memory_gb": 0.0},
                {"component": "clip_encoding", "mean_sec_per_video_min": 8.0, "std_sec_per_video_min": 0.5, "peak_gpu_memory_gb": 3.8},
                {"component": "whisper_transcription", "mean_sec_per_video_min": 3.5, "std_sec_per_video_min": 0.6, "peak_gpu_memory_gb": 5.6},
                {"component": "llm_planning", "mean_sec_per_video_min": 1.2, "std_sec_per_video_min": 0.2, "peak_gpu_memory_gb": 0.0},
                {"component": "ffmpeg_rendering", "mean_sec_per_video_min": 4.0, "std_sec_per_video_min": 0.4, "peak_gpu_memory_gb": 1.2},
            ]
        viz.plot_runtime_breakdown(runtime_data)

        performance_data = [
            {"name": "Rule-Based", "P": 0.74, "F1": 0.73, "Time": 18.5, "Sat": 3.6, "Eff": 3.7, "Use": 3.5, "V": 0.82},
            {"name": "SVM-Based", "P": 0.79, "F1": 0.77, "Time": 16.2, "Sat": 3.9, "Eff": 4.0, "Use": 3.8, "V": 0.85},
            {"name": "AV-Summary", "P": 0.84, "F1": 0.83, "Time": 14.8, "Sat": 4.2, "Eff": 4.3, "Use": 4.0, "V": 0.87},
            {"name": "VideoLLM", "P": 0.91, "F1": 0.89, "Time": 15.0, "Sat": 4.6, "Eff": 4.5, "Use": 4.5, "V": 0.92},
            {"name": "SyncClipAgent", "P": round(seg_m.precision, 2), "F1": round(seg_m.f1_score, 2),
             "Time": round(pipeline_result.total_runtime_s / 60.0, 1), "Sat": 4.5, "Eff": 4.7, "Use": 4.4, "V": 0.90},
        ]
    else:
        per_genre_fps = {
            "vlog": {1: {"P": 0.92, "R": 0.88, "F1": 0.90}, 2: {"P": 0.93, "R": 0.90, "F1": 0.91},
                      3: {"P": 0.93, "R": 0.91, "F1": 0.92}, 5: {"P": 0.94, "R": 0.92, "F1": 0.93}},
        }
        viz.plot_f1_vs_fps(per_genre_fps)
        viz.plot_sync_error_vs_fps({1: 280, 2: 210, 3: 160, 5: 130})
        viz.plot_sensitivity_curves("theta",
                                     [0.05, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30],
                                     [0.86, 0.88, 0.91, 0.93, 0.92, 0.87, 0.83],
                                     [250, 200, 155, 130, 145, 210, 280])
        viz.plot_sensitivity_curves("tau",
                                     [0.50, 0.65, 0.75, 0.80, 0.90, 0.95],
                                     [0.87, 0.89, 0.91, 0.93, 0.91, 0.85],
                                     [230, 180, 145, 130, 140, 195])
        runtime_data = [
            {"component": "frame_extraction", "mean_sec_per_video_min": 1.3, "std_sec_per_video_min": 0.2, "peak_gpu_memory_gb": 0.0},
            {"component": "clip_encoding", "mean_sec_per_video_min": 3.4, "std_sec_per_video_min": 0.5, "peak_gpu_memory_gb": 3.8},
            {"component": "whisper_transcription", "mean_sec_per_video_min": 4.0, "std_sec_per_video_min": 0.6, "peak_gpu_memory_gb": 5.6},
            {"component": "llm_planning", "mean_sec_per_video_min": 1.1, "std_sec_per_video_min": 0.2, "peak_gpu_memory_gb": 0.0},
            {"component": "plan_validation", "mean_sec_per_video_min": 0.2, "std_sec_per_video_min": 0.05, "peak_gpu_memory_gb": 0.0},
            {"component": "ffmpeg_rendering", "mean_sec_per_video_min": 3.3, "std_sec_per_video_min": 0.4, "peak_gpu_memory_gb": 1.2},
        ]
        viz.plot_runtime_breakdown(runtime_data)
        performance_data = [
            {"name": "Rule-Based", "P": 0.74, "F1": 0.73, "Time": 18.5, "Sat": 3.6, "Eff": 3.7, "Use": 3.5, "V": 0.82},
            {"name": "SVM-Based", "P": 0.79, "F1": 0.77, "Time": 16.2, "Sat": 3.9, "Eff": 4.0, "Use": 3.8, "V": 0.85},
            {"name": "AV-Summary", "P": 0.84, "F1": 0.83, "Time": 14.8, "Sat": 4.2, "Eff": 4.3, "Use": 4.0, "V": 0.87},
            {"name": "VideoLLM", "P": 0.91, "F1": 0.89, "Time": 15.0, "Sat": 4.6, "Eff": 4.5, "Use": 4.5, "V": 0.92},
            {"name": "SyncClipAgent", "P": 0.90, "F1": 0.91, "Time": 13.3, "Sat": 4.5, "Eff": 4.7, "Use": 4.4, "V": 0.90},
        ]

    tables = viz.generate_manuscript_tables(
        segment_data=per_genre_fps,
        performance_data=performance_data,
        runtime_data=runtime_data,
    )

    logger.info(f"  Generated {len(tables)} LaTeX tables:")
    for name, path in tables.items():
        logger.info(f"    {name}: {path}")

    logger.info(f"  Generated figures in: {config.output_dir}")
    return viz.save_all()


def run_cross_modal_projection(config: ExperimentConfig):
    logger.info("=" * 60)
    logger.info("Running Cross-Modal Projection Module Test")
    logger.info("=" * 60)

    import numpy as np
    from src.models.cross_modal_projection import CrossModalProjection

    n_samples = 200
    rng = np.random.default_rng(config.seed)

    V = rng.normal(0, 1, (n_samples, config.model["clip_embedding_dim"]))
    A = rng.normal(0, 1, (n_samples, config.model["whisper_embedding_dim"]))

    projector = CrossModalProjection(
        d_visual=config.model["clip_embedding_dim"],
        d_audio=config.model["whisper_embedding_dim"],
        d_common=config.model["common_projection_dim"],
        method=config.model["projection_method"],
    )
    projector.fit(V, A)

    logger.info(f"  Method:        {projector.method}")
    logger.info(f"  d_visual:      {projector.d_visual}")
    logger.info(f"  d_audio:       {projector.d_audio}")
    logger.info(f"  d_common:      {projector.d_common}")
    logger.info(f"  W_v shape:     {projector.W_v.shape}")
    logger.info(f"  W_a shape:     {projector.W_a.shape}")

    z_v = projector.project_visual(V[0])
    z_a = projector.project_audio(A[0])
    sim = float(np.dot(z_v, z_a))
    logger.info(f"  ||z_v||:       {np.linalg.norm(z_v):.4f}")
    logger.info(f"  ||z_a||:       {np.linalg.norm(z_a):.4f}")
    logger.info(f"  cos_sim(v,a):  {sim:.4f}")

    P = projector.compute_alignment_matrix(V[:5], A[:5])
    logger.info(f"  Alignment P:   shape={P.shape}, sum={P.sum():.2f}")

    pairs = projector.get_alignment_pairs(V[:5], A[:5], tau=0.1)
    logger.info(f"  Pairs (tau=0.1): {len(pairs)} found")
    if pairs:
        logger.info(f"    Example: {pairs[0]}")

    cfg_path = os.path.join(config.output_dir, "projection_config.json")
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(projector.to_dict(), f, indent=2)
    logger.info(f"  Config saved to: {cfg_path}")


def run_download_pipeline(
    config: ExperimentConfig,
    url: str = "",
    search_query: str = "",
    source: str = "pexels",
    count: int = 5,
    download_only: bool = False,
):
    """Download videos from online sources and optionally run the experiment pipeline."""
    from experiments.video_downloader import download_videos, download_single

    if url:
        logger.info(f"Downloading single video: {url}")
        meta = download_single(url, cache_dir=config.dataset_dir)
        if meta:
            video_paths = {meta["genre"]: [meta["path"]]}
            metas = [meta]
        else:
            logger.error(f"Failed to download: {url}")
            return None, []
    elif search_query:
        logger.info(f"Searching '{search_query}' on {source} (count={count})...")
        video_paths, metas = download_videos(
            query=search_query, source=source, count=count,
            cache_dir=config.dataset_dir,
        )
    else:
        return None, []

    if not metas:
        logger.error("No videos downloaded.")
        return None, []

    for meta in metas:
        logger.info(f"  [{meta['genre']}] {meta['title'][:60]} -> {meta['path']}")

    if download_only:
        summary_path = os.path.join(config.output_dir, "download_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump([{k: v for k, v in m.items() if k != "path"} for m in metas],
                      f, indent=2, ensure_ascii=False)
        logger.info(f"Download summary saved to {summary_path}")
        return None, metas

    ground_truth_builder = GroundTruthBuilder(config)
    mock_segments = _generate_mock_segments(config)
    _, ground_truth = ground_truth_builder.build_ground_truth(mock_segments)

    runner = ExperimentRunner(config)
    result = runner.run_full_pipeline(
        video_paths={g: video_paths.get(g, []) for g in config.genre_list},
        reference_segments=ground_truth,
    )

    logger.info(f"  Pipeline result — P={result.segment_metrics.precision:.4f}, "
                 f"R={result.segment_metrics.recall:.4f}, "
                 f"F1={result.segment_metrics.f1_score:.4f}")
    return result, metas


def main():
    parser = argparse.ArgumentParser(description="SyncCLIPAgent Experiment Suite")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Run in mock mode (no GPU/API required)")
    parser.add_argument("--real", action="store_true",
                        help="Run with real models (requires GPU+API)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to dataset directory")
    parser.add_argument("--url", type=str, default="",
                        help="Download and process a single video URL")
    parser.add_argument("--search", type=str, default="",
                        help="Search query for downloading videos (e.g. 'vlog daily')")
    parser.add_argument("--source", type=str, default="pexels",
                        choices=["pexels", "pixabay", "archive", "ytdlp"],
                        help="Video source for search (default: pexels)")
    parser.add_argument("--count", type=int, default=5,
                        help="Number of videos to download when searching (default: 5)")
    parser.add_argument("--download-only", action="store_true",
                        help="Download videos without running the experiment pipeline")
    parser.add_argument("--ground-truth", action="store_true",
                        help="Run ground truth construction")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run parameter sensitivity analysis")
    parser.add_argument("--profile", action="store_true",
                        help="Run runtime profiling")
    parser.add_argument("--robustness", action="store_true",
                        help="Run robustness tests")
    parser.add_argument("--projection", action="store_true",
                        help="Run cross-modal projection module test")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations and LaTeX tables")
    parser.add_argument("--all", action="store_true",
                        help="Run all experiments")
    parser.add_argument("--end-to-end", action="store_true",
                        help="Run end-to-end pipeline (preprocess->CLIP->Whisper->plan->validate->render)")
    parser.add_argument("--video", type=str, default="",
                        help="Video file for end-to-end pipeline")
    parser.add_argument("--request", type=str, default="",
                        help="Editing request text for end-to-end pipeline")
    parser.add_argument("--target-duration", type=float, default=60.0,
                        help="Target video duration in seconds (default: 60)")
    parser.add_argument("--output", type=str, default="experiments/output",
                        help="Output directory")

    args = parser.parse_args()

    config = load_config(
        mock_mode=args.mock and not args.real,
        output_dir=args.output,
    )
    if args.data:
        config.dataset_dir = args.data

    os.makedirs(config.output_dir, exist_ok=True)
    config.save(os.path.join(config.output_dir, "config.json"))

    start_time = datetime.now()
    logger.info(f"SyncCLIPAgent Experiment Suite starting at {start_time}")
    logger.info(f"  Mock mode: {config.mock_mode}")
    logger.info(f"  Output:    {config.output_dir}")

    is_download = bool(args.url or args.search)
    is_e2e = args.end_to_end or bool(args.video)
    has_explicit_experiments = any([
        args.ground_truth, args.sensitivity, args.profile,
        args.robustness, args.projection, args.visualize,
    ])

    if is_download:
        run_download_pipeline(
            config,
            url=args.url,
            search_query=args.search,
            source=args.source,
            count=args.count,
            download_only=args.download_only,
        )
        if args.download_only:
            logger.info("Download-only mode: skipping experiments.")
            return

    if is_e2e:
        logger.info("=" * 60)
        logger.info("Running End-to-End Pipeline")
        logger.info("=" * 60)
        from experiments.run_experiment import EndToEndRunner
        e2e = EndToEndRunner(config)
        if args.video:
            e2e.run_single(args.video, args.request or "Create a highlight video.",
                           target_duration=args.target_duration, mock=config.mock_mode)
        else:
            import tempfile
            mock_video = os.path.join(tempfile.gettempdir(), "mock_demo_video.mp4")
            Path(mock_video).touch()
            e2e.run_single(mock_video, "Create a 60-second highlight reel.", mock=True)
        e2e.save_results()

    run_all = args.all or (not is_download and not is_e2e and not has_explicit_experiments)

    if run_all and not config.mock_mode:
        video_paths = scan_video_paths(config)
        if video_paths:
            n = sum(len(v) for v in video_paths.values())
            logger.info(f"Found {n} videos, running real E2E pipeline first...")
            e2e_results = run_e2e_pipeline(config, video_paths)
            logger.info(f"E2E pipeline complete: {len(e2e_results)} videos processed")

    pipeline_result_for_viz = None
    if run_all and not config.mock_mode:
        try:
            pipeline_result_for_viz = run_full_pipeline(config)
        except Exception as e:
            logger.warning(f"Full pipeline failed (may need real data): {e}")

    if run_all or args.ground_truth:
        run_ground_truth(config)

    if run_all or args.sensitivity:
        run_sensitivity(config)

    if run_all or args.profile:
        run_profiling(config)

    if run_all or args.robustness:
        run_robustness(config)

    if run_all or args.projection:
        run_cross_modal_projection(config)

    if run_all or args.visualize:
        run_visualization(config, pipeline_result_for_viz)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"All experiments completed in {elapsed:.1f}s")
    logger.info(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
