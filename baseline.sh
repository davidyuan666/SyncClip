#!/bin/bash
set -e

echo "=== SyncCLIPAgent Baseline Runner ==="
echo ""

# 1. Pull latest code
echo "[1/6] Pulling latest code..."
git pull

# 2. Install dependencies
echo "[2/6] Installing dependencies..."
pip install openai>=1.0.0 openai-whisper>=20231117 ffmpeg-python>=0.2.0 "transformers>=4.44,<4.45"

# 3. Run Rule-Based baseline
echo "[3/6] Running Rule-Based baseline..."
python -m baselines.rule_based

# 4. Run SVM-Based baseline
echo "[4/6] Running SVM-Based baseline..."
python -m baselines.svm_based

# 5. Run CLIP4Clip baseline
echo "[5/6] Running CLIP4Clip baseline..."
python -m baselines.clip4clip_based

# 6. Run PGL-SUM baseline
echo "[6/6] Running PGL-SUM baseline..."
python -m baselines.pglsum_based

echo ""
echo "=== Done ==="
echo "Results:"
echo "  experiments/output/baselines/rule_based/"
echo "  experiments/output/baselines/svm_based/"
echo "  experiments/output/baselines/clip4clip/"
echo "  experiments/output/baselines/pglsum/"
