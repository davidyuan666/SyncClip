# SyncCLIPAgent

**Leveraging LLMs for Automated Video Editing with Audiovisual Synchronization**

[arXiv / Publisher URL](https://github.com/davidyuan666/SyncClip) &nbsp;|&nbsp; Submitted to *Cluster Computing*

---

## Overview

SyncCLIPAgent is a zero-shot orchestration pipeline that chains pretrained components — **CLIP** (visual encoding), **Whisper** (speech transcription), an **LLM** (constrained planning), and **FFmpeg** (deterministic rendering) — to produce timestamped, machine-executable video editing plans from raw video and a natural-language instruction. The LLM does not generate video content; it receives structured evidence and returns a validated, schema-constrained edit decision.

On 22 vlog-style clips, the pipeline achieves a **100% validation pass rate** (22/22 rendered successfully), with a mean end-to-end processing time of 136.1 s per video.

---

## Architecture

```
Raw Video + User Request
        │
        ▼
┌─────────────────────────┐
│ 1. Preprocessing        │   Frame sampling (SSIM keyframe detection)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 2. Evidence Extraction  │   CLIP ViT-B/32 (visual) + Whisper large-v3 (audio)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 3. LLM Planning         │   Structured edit decision (segments, transitions, sync)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 4. Validation + Render  │   Schema checks → FFmpeg composition
└─────────────────────────┘
```

---

## Installation

```bash
# 1. Clone and create .env
cp .env.template .env
# Edit .env: set OPENAI_API_KEY (or DEEPSEEK_API_KEY)

# 2. Install system deps + Python packages
make install
```

**Requirements:** Python 3.10–3.12, FFmpeg, CUDA-capable GPU recommended.

---

## Quick Start

### Single-video experiment

```bash
python -m experiments.run_experiment --video data/my_video.mp4 --request "Create a 60-second highlight video"
```

### Full experiment suite (mock mode, no GPU/API)

```bash
python -m experiments.run_all --all --mock
```

### Production server

```bash
make start          # FastAPI on port 6001
make stop           # Shutdown
make logs           # View live logs
```

---

## Experiments & Baselines

### Run baselines (against the same dataset)

```bash
python -m baselines.rule_based      --data experiments/data/vlog/
python -m baselines.svm_based       --data experiments/data/vlog/
python -m baselines.clip4clip_based --data experiments/data/vlog/
python -m baselines.pglsum_based    --data experiments/data/vlog/
```

| Baseline | Approach |
|---|---|
| **Rule-Based** | Frame-difference shot boundary + audio energy, greedy selection |
| **SVM-Based** | Visual/audio features trained with LinearSVC pseudo-labels |
| **CLIP4Clip** | CLIP-based text-visual retrieval for segment ranking |
| **PGL-SUM** | Self-attention video summarization on CLIP frame embeddings |

### Evaluation metrics

- **RQ1** — Segment selection precision, recall, F1 (vs. heuristic ground truth)
- **RQ2** — Temporal sync error (ms), within-200ms rate
- **RQ3** — Semantic correspondence (cross-modal similarity)
- **RQ4** — Runtime per component, pilot quality ratings

Results are saved under `experiments/output/` as CSV files and plots.

---

## Project Structure

```
SyncClip/
├── app.py                 # FastAPI server entry point
├── Makefile               # Build, install, start/stop helpers
├── pyproject.toml         # Python dependencies (pdm)
├── experiments/           # Research evaluation suite
│   ├── run_all.py         # Full 6-stage experiment orchestrator
│   ├── run_experiment.py  # Single-video end-to-end pipeline
│   ├── preprocess.py      # Frame extraction + SSIM keyframe detection
│   ├── extract_clip.py    # CLIP ViT-B/32 visual encoding
│   ├── transcribe_whisper.py  # Whisper large-v3 transcription
│   ├── build_candidates.py    # Segment grouping + importance scoring
│   ├── llm_planner.py     # LLM structured edit decision generation
│   ├── validate_plan.py   # Schema + timing constraint validation
│   ├── render_ffmpeg.py   # FFmpeg command generation and rendering
│   └── ...
├── baselines/             # Comparison methods
│   ├── rule_based.py
│   ├── svm_based.py
│   ├── clip4clip_based.py
│   └── pglsum_based.py
├── src/
│   ├── agents/            # Production pipeline agents (13 total)
│   ├── models/            # Cross-modal projection, BLIP, ViT
│   └── utils/             # LLM client, cloud storage, video utilities
└── scripts/               # Human evaluation, data organization
```

---

## Citation

```bibtex
@article{yuan2025syncclipagent,
  title   = {SyncCLIPAgent: Leveraging LLMs for Automated Video Editing
             with Audiovisual Synchronization},
  author  = {Yuan, Dawei and Liang, Guojun},
  journal = {Cluster Computing},
  year    = {2025},
  note    = {Under review},
  url     = {https://github.com/davidyuan666/SyncClip}
}
```

## Contact

yuandawei@gdust.edu.cn
