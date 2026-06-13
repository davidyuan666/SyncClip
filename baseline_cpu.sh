#!/bin/bash
set -e

NOW=$(date '+%Y-%m-%d %H:%M:%S')

echo "=== SyncCLIPAgent Baseline Runner (CPU) ==="
echo "$NOW"
echo ""

# ----------------------------------------------------------
# 1. Pull latest code
# ----------------------------------------------------------
echo "[1/4] Pulling latest code..."
START=$SECONDS
git pull
ELAPSED=$(($SECONDS - START))
echo "  ✓ ${ELAPSED}s"
echo ""

# ----------------------------------------------------------
# 2. Install dependencies
# ----------------------------------------------------------
echo "[2/4] Installing Python dependencies..."
START=$SECONDS
pip install "scipy>=1.10.0" "scikit-learn>=1.0.0" "ffmpeg-python>=0.2.0"
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
# 3. Rule-Based
# ----------------------------------------------------------
echo "[3/4] Rule-Based"
START=$SECONDS
export PYTHONUNBUFFERED=1
python -m baselines.rule_based
ELAPSED=$(($SECONDS - START))
RULE_TIME="${ELAPSED}s"
RULE_SUMMARY=$(read_summary "experiments/output/baselines/rule_based/summary.json")
RULE_P=$(echo "$RULE_SUMMARY" | awk '{print $1}')
RULE_R=$(echo "$RULE_SUMMARY" | awk '{print $2}')
RULE_F1=$(echo "$RULE_SUMMARY" | awk '{print $3}')
RULE_N=$(echo "$RULE_SUMMARY" | awk '{print $4}')
if [ "$RULE_P" = "N/A" ]; then
    echo "  ✗ FAIL"
else
    echo "  ✓ ${ELAPSED}s | ${RULE_N}/22 videos | P=${RULE_P} R=${RULE_R} F1=${RULE_F1}"
fi
echo ""

# ----------------------------------------------------------
# 4. SVM-Based
# ----------------------------------------------------------
echo "[4/4] SVM-Based"
START=$SECONDS
export PYTHONUNBUFFERED=1
python -m baselines.svm_based
ELAPSED=$(($SECONDS - START))
SVM_TIME="${ELAPSED}s"
SVM_SUMMARY=$(read_summary "experiments/output/baselines/svm_based/summary.json")
SVM_P=$(echo "$SVM_SUMMARY" | awk '{print $1}')
SVM_R=$(echo "$SVM_SUMMARY" | awk '{print $2}')
SVM_F1=$(echo "$SVM_SUMMARY" | awk '{print $3}')
SVM_N=$(echo "$SVM_SUMMARY" | awk '{print $4}')
if [ "$SVM_P" = "N/A" ]; then
    echo "  ✗ FAIL"
else
    echo "  ✓ ${ELAPSED}s | ${SVM_N}/22 videos | P=${SVM_P} R=${SVM_R} F1=${SVM_F1}"
fi
echo ""

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo "========================================"
printf "  %-14s %8s %8s %8s %8s\n" "Baseline" "P" "R" "F1" "Time"
echo "  --------------------------------------"
printf "  %-14s %8s %8s %8s %8s\n" \
    "Rule-Based" "$RULE_P" "$RULE_R" "$RULE_F1" "$RULE_TIME"
printf "  %-14s %8s %8s %8s %8s\n" \
    "SVM-Based" "$SVM_P" "$SVM_R" "$SVM_F1" "$SVM_TIME"
echo "========================================"
echo ""
echo "Results directories:"
echo "  experiments/output/baselines/rule_based/"
echo "  experiments/output/baselines/svm_based/"
