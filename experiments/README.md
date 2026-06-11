# SyncCLIPAgent Experiments — Reproducibility Guide

This directory contains the complete experimental pipeline for the SyncCLIPAgent paper (Cluster Computing, Major Revision).

## Quick Start

```bash
# Mock mode (no GPU/API required, 5s runtime)
python -m experiments.run_experiment --mock

# End-to-end on a single video
python -m experiments.run_experiment --video data/action_01.mp4 --request "60s highlight"

# Batch on a video directory (organized by genre subdirs)
python -m experiments.run_experiment --video-dir ./data/videos --all

# Full experiment suite (metrics, sensitivity, profiling, robustness, plots)
python -m experiments.run_all --all --mock

# Online video download + pipeline
python -m experiments.run_all --search "nature documentary" --source pexels --count 3
```

## Requirements

```bash
pip install -r requirement.txt
pip install yt-dlp        # For online video download
pip install openai         # For LLM planner (or set OPENAI_API_KEY)
pip install opencv-python-headless  # For frame extraction
# Optional: PyTorch + transformers for real CLIP
# Optional: whisper for real Whisper transcription
```

## Module Overview

| Module | Function |
|--------|----------|
| `config.py` | Central parameter registry (θ, τ, α/β, d_c, hardware) |
| `preprocess.py` | Frame extraction (1/2/3/5 fps) + audio + metadata |
| `extract_clip.py` | CLIP ViT-B/32 + RN50x16 visual encoding |
| `transcribe_whisper.py` | Whisper large-v3 + base transcription |
| `build_candidates.py` | Keyframe grouping + importance scoring |
| `llm_planner.py` | GPT-4 structured edit decision schema |
| `validate_plan.py` | Hard/soft checks + LLM revision (max 2) |
| `render_ffmpeg.py` | FFmpeg rendering from validated decisions |
| `run_experiment.py` | End-to-end orchestrator |
| `run_all.py` | Experiment suite CLI (all RQ1-RQ4 + sensitivity + profiling + robustness) |
| `csv_output.py` | RQ1-RQ4 CSV formatter |
| `metrics/` | Precision/Recall/F1, sync error, semantic correspondence |
| `statistics.py` | Bootstrap CI, Welch's t-test, LaTeX tables |
| `visualization.py` | Matplotlib: F1-vs-fps, sync curves, sensitivity, runtime, robustness heatmap |
| `ground_truth.py` | Multi-annotator protocol, Cohen's kappa |
| `sensitivity.py` | Parameter sweep (θ, τ, α/β, fps) |
| `profiling.py` | Per-component runtime (5/10/15/30 min videos) |
| `robustness.py` | 5 stress cases (low-res, noise, fast scene, non-English, music) |
| `video_downloader.py` | Pexels/Pixabay/Internet Archive/yt-dlp download + cache |

## Output Files

All results are saved to `experiments/output/`:

| File | Description |
|------|-------------|
| `rq1_segment_selection.csv` | Precision/Recall/F1 by genre × fps |
| `rq2_temporal_sync.csv` | Mean/median/std sync error, within-100/200/500ms % |
| `rq3_semantic_transition.csv` | Visual/audio/cross-modal similarity, transition smoothness |
| `rq4_runtime.csv` | Per-component runtime (sec/video-min) |
| `rq4_user_study.csv` | Satisfaction/efficiency/ease-of-use ratings |
| `robustness.csv` | Stress case F1/sync/semantic/invalid-rate |
| `config.json` | Experiment configuration snapshot |
| `e2e_results.json` | End-to-end pipeline results |
| `plans/` | Edit decision JSON files per video |
| `reports/` | Validation reports per video |

## Paper Correspondence

| Paper Table/Figure | CSV/Output |
|--------------------|------------|
| Table 4: Segment Selection Accuracy | `rq1_segment_selection.csv` |
| Table 5: Temporal Synchronization | `rq2_temporal_sync.csv` |
| Table 6: Semantic Correspondence | `rq3_semantic_transition.csv` |
| Table 7: Performance Comparison | `rq4_runtime.csv` + `rq4_user_study.csv` |
| Sensitivity plots | `sensitivity_theta.png`, `sensitivity_tau.png` |
| Runtime breakdown | `rq4_runtime.csv`, `runtime_breakdown.png` |
| Robustness evaluation | `robustness.csv`, `robustness_heatmap.png` |
| Guide Section 2: Parameter Settings | `config.json` |
| Guide Section 2: Ground Truth | `ground_truth_stats.json` |

## Environment Variables

```bash
OPENAI_API_KEY=sk-...      # Required for LLM planner (non-mock mode)
PEXELS_API_KEY=...         # Optional: higher Pexels rate limit
PIXABAY_API_KEY=...        # Optional: Pixabay search
COS_ENABLED=0              # 0 = local file mode (default)
TTS_ENABLED=0              # 0 = mock TTS (default)
```

## Citation

If you use SyncCLIPAgent in your research, please cite:
```
@article{dawei2025syncclipagent,
  title={{SyncCLIPAgent: Leveraging LLMs for Automated Video Editing with Audiovisual Synchronization}},
  author={Dawei Yuan and Guojun Liang},
  journal={Cluster Computing},
  year={2025},
}
```
