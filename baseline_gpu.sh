#!/bin/bash
set -e

export HF_ENDPOINT=https://hf-mirror.com

NOW=$(date '+%Y-%m-%d %H:%M:%S')

echo "=== SyncCLIPAgent Baseline Runner (GPU) ==="
echo "$NOW"
echo ""

# ----------------------------------------------------------
# 1. Pull latest code
# ----------------------------------------------------------
echo "[1/5] Pulling latest code..."
START=$SECONDS
git pull
ELAPSED=$(($SECONDS - START))
echo "  ✓ ${ELAPSED}s"
echo ""

# ----------------------------------------------------------
# 2. Install system dependencies
# ----------------------------------------------------------
echo "[2/5] Installing system dependencies..."
START=$SECONDS
if command -v ffmpeg &>/dev/null; then
    echo "  ffmpeg already installed, skip"
else
    apt-get update -qq && apt-get install -y -qq ffmpeg
fi
ELAPSED=$(($SECONDS - START))
echo "  ✓ ${ELAPSED}s"
echo ""

# ----------------------------------------------------------
# 3. Install Python dependencies
# ----------------------------------------------------------
echo "[3/5] Installing Python dependencies..."
START=$SECONDS
pip install "openai>=1.0.0" "openai-whisper>=20231117" "ffmpeg-python>=0.2.0" "transformers>=4.44,<4.45"
ELAPSED=$(($SECONDS - START))
echo "  ✓ ${ELAPSED}s"
echo ""

# ----------------------------------------------------------
# Helper: read summary.json
# ----------------------------------------------------------
read_summary() {
    local path="$1"
    if [ -f "$path" ]; then
        python3 -c "
import json
with open('$path') as f:
    s = json.load(f)
print(f\"{s['avg_precision']:.2f} {s['avg_recall']:.2f} {s['avg_f1']:.2f} {s['n_videos']}\")
" 2>/dev/null || echo "N/A N/A N/A N/A"
    else
        echo "N/A N/A N/A N/A"
    fi
}

# ----------------------------------------------------------
# 4. CLIP4Clip
# ----------------------------------------------------------
echo "[4/5] CLIP4Clip"
START=$SECONDS
export PYTHONUNBUFFERED=1
python -m baselines.clip4clip_based
ELAPSED=$(($SECONDS - START))
CLIP_TIME="${ELAPSED}s"
CLIP_SUMMARY=$(read_summary "experiments/output/baselines/clip4clip/summary.json")
CLIP_P=$(echo "$CLIP_SUMMARY" | awk '{print $1}')
CLIP_R=$(echo "$CLIP_SUMMARY" | awk '{print $2}')
CLIP_F1=$(echo "$CLIP_SUMMARY" | awk '{print $3}')
CLIP_N=$(echo "$CLIP_SUMMARY" | awk '{print $4}')
if [ "$CLIP_P" = "N/A" ]; then
    echo "  ✗ FAIL"
else
    echo "  ✓ ${ELAPSED}s | ${CLIP_N}/22 videos | P=${CLIP_P} R=${CLIP_R} F1=${CLIP_F1}"
fi
echo ""

# ----------------------------------------------------------
# 5. PGL-SUM
# ----------------------------------------------------------
echo "[5/5] PGL-SUM"
START=$SECONDS
export PYTHONUNBUFFERED=1
python -m baselines.pglsum_based
ELAPSED=$(($SECONDS - START))
PGL_TIME="${ELAPSED}s"
PGL_SUMMARY=$(read_summary "experiments/output/baselines/pglsum/summary.json")
PGL_P=$(echo "$PGL_SUMMARY" | awk '{print $1}')
PGL_R=$(echo "$PGL_SUMMARY" | awk '{print $2}')
PGL_F1=$(echo "$PGL_SUMMARY" | awk '{print $3}')
PGL_N=$(echo "$PGL_SUMMARY" | awk '{print $4}')
if [ "$PGL_P" = "N/A" ]; then
    echo "  ✗ FAIL"
else
    echo "  ✓ ${ELAPSED}s | ${PGL_N}/22 videos | P=${PGL_P} R=${PGL_R} F1=${PGL_F1}"
fi
echo ""

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo "========================================"
printf "  %-14s %8s %8s %8s %8s\n" "Baseline" "P" "R" "F1" "Time"
echo "  --------------------------------------"
printf "  %-14s %8s %8s %8s %8s\n" \
    "CLIP4Clip" "$CLIP_P" "$CLIP_R" "$CLIP_F1" "$CLIP_TIME"
printf "  %-14s %8s %8s %8s %8s\n" \
    "PGL-SUM" "$PGL_P" "$PGL_R" "$PGL_F1" "$PGL_TIME"
echo "========================================"
echo ""
echo "Results directories:"
echo "  experiments/output/baselines/clip4clip/"
echo "  experiments/output/baselines/pglsum/"
