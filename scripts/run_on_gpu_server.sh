#!/bin/bash
# SyncCLIPAgent — Remote GPU Server Experiment Runner
# 
# Usage:
#   bash scripts/run_on_gpu_server.sh              # Run all experiments (mock)
#   bash scripts/run_on_gpu_server.sh --real        # Run with real models (needs GPU+API)
#   bash scripts/run_on_gpu_server.sh --data /data/videos  # Custom data dir
#
# Pre-requisites:
#   1. Python 3.10+ with CUDA-enabled PyTorch
#   2. ffmpeg + ffprobe in PATH
#   3. Set env vars: OPENAI_API_KEY (or DEEPSEEK_API_KEY)
#   4. pip install -r requirement.txt (or pdm install)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ——— Auto-detect GPU hardware ———
echo "=== Hardware Detection ==="
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "CPU-only")
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
GPU_MEM_GB=$(echo "scale=1; $GPU_MEM / 1024" | bc 2>/dev/null || echo "0")
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' 2>/dev/null || echo "N/A")

echo "  GPU: $GPU_NAME (${GPU_MEM_GB}GB)"
echo "  RAM: ${RAM_GB}GB"
echo "  CPU: $CPU_MODEL"
echo "  CUDA: $CUDA_VER"

export SYNCCLIP_GPU="$GPU_NAME"
export SYNCCLIP_GPU_MEMORY_GB="$GPU_MEM_GB"
export SYNCCLIP_RAM_GB="$RAM_GB"
export SYNCCLIP_CPU="$CPU_MODEL"
export SYNCCLIP_CUDA_VERSION="$CUDA_VER"

# ——— Data directory ———
DATA_DIR="${SYNCCLIP_DATA_DIR:-$PROJECT_DIR/experiments/data}"
OUTPUT_DIR="${SYNCCLIP_OUTPUT_DIR:-$PROJECT_DIR/experiments/output}"

echo ""
echo "=== Paths ==="
echo "  Data dir:   $DATA_DIR"
echo "  Output dir: $OUTPUT_DIR"

# ——— API Key Check ———
if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo ""
    echo "WARNING: No OPENAI_API_KEY or DEEPSEEK_API_KEY set."
    echo "  LLM planning will fall back to mock mode."
    echo "  Set one of them to use real LLM:"
    echo "    export OPENAI_API_KEY=sk-..."
    echo "    export DEEPSEEK_API_KEY=sk-..."
fi

# ——— Parse arguments ———
REAL_FLAG=""
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --real) REAL_FLAG="--real" ;;
        --data) shift; EXTRA_ARGS+=("--data" "$1"); shift ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

# ——— Run ———
echo ""
echo "=== Starting SyncCLIPAgent Experiments ==="
echo "Date: $(date)"

if [ -n "$REAL_FLAG" ]; then
    echo "Mode: REAL (GPU + API required)"
else
    echo "Mode: Mock (testing pipeline without GPU)"
fi

python -m experiments.run_all \
    $REAL_FLAG \
    --all \
    --output "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Done ==="
echo "Results saved to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR/"
