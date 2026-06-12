#!/bin/bash
set -e

echo "=== SyncCLIPAgent Baseline Runner ==="
echo ""

# 1. Pull latest code
echo "[1/4] Pulling latest code..."
git pull

# 2. Install dependencies
echo "[2/4] Installing dependencies..."
pip install openai>=1.0.0 openai-whisper>=20231117 ffmpeg-python>=0.2.0 "transformers>=4.44,<4.45"

# 3. Run CLIP4Clip baseline
echo "[3/4] Running CLIP4Clip baseline..."
python -m baselines.clip4clip_based

# 4. Run PGL-SUM baseline
echo "[4/4] Running PGL-SUM baseline..."
python -m baselines.pglsum_based

echo ""
echo "=== Done ==="
echo "Results: experiments/output/baselines/clip4clip/"
echo "Results: experiments/output/baselines/pglsum/"
