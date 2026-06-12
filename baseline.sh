#!/bin/bash
set -e

NOW=$(date '+%Y-%m-%d %H:%M:%S')

echo "=== SyncCLIPAgent Baseline Runner ==="
echo "$NOW"
echo ""

# ----------------------------------------------------------
# 1. Pull latest code
# ----------------------------------------------------------
echo "[1/6] Pulling latest code..."
START=$SECONDS
git pull
ELAPSED=$(($SECONDS - START))
echo "  ✓ ${ELAPSED}s"
echo ""

# ----------------------------------------------------------
# 2. Install dependencies
# ----------------------------------------------------------
echo "[2/6] Installing dependencies..."
START=$SECONDS
pip install "openai>=1.0.0" "openai-whisper>=20231117" "ffmpeg-python>=0.2.0" "transformers>=4.44,<4.45"
ELAPSED=$(($SECONDS - START))
echo "  ✓ ${ELAPSED}s"
echo ""

# ----------------------------------------------------------
# 3. Rule-Based
# ----------------------------------------------------------
echo "[3/6] Rule-Based"
START=$SECONDS
python -m baselines.rule_based 2>&1 | while IFS= read -r line; do
    if echo "$line" | grep -q '^\[Rule-Based\]'; then
        echo "  $line"
    fi
done
ELAPSED=$(($SECONDS - START))
RULE_TIME="${ELAPSED}s"
RULE_SUMMARY=$(python -c "
import json
with open('experiments/output/baselines/rule_based/summary.json') as f:
    s = json.load(f)
print(f\"{s['avg_precision']:.2f} {s['avg_recall']:.2f} {s['avg_f1']:.2f} {s['n_videos']}\")
" 2>/dev/null || echo "FAIL FAIL FAIL FAIL")
echo "  ✓ ${ELAPSED}s | videos=$(echo $RULE_SUMMARY | awk '{print $4}') | P=$(echo $RULE_SUMMARY | awk '{print $1}') R=$(echo $RULE_SUMMARY | awk '{print $2}') F1=$(echo $RULE_SUMMARY | awk '{print $3}')"
echo ""

# ----------------------------------------------------------
# 4. SVM-Based
# ----------------------------------------------------------
echo "[4/6] SVM-Based"
START=$SECONDS
python -m baselines.svm_based 2>&1 | while IFS= read -r line; do
    if echo "$line" | grep -q '^\[SVM-Based\]'; then
        echo "  $line"
    fi
done
ELAPSED=$(($SECONDS - START))
SVM_TIME="${ELAPSED}s"
SVM_SUMMARY=$(python -c "
import json
with open('experiments/output/baselines/svm_based/summary.json') as f:
    s = json.load(f)
print(f\"{s['avg_precision']:.2f} {s['avg_recall']:.2f} {s['avg_f1']:.2f} {s['n_videos']}\")
" 2>/dev/null || echo "FAIL FAIL FAIL FAIL")
echo "  ✓ ${ELAPSED}s | videos=$(echo $SVM_SUMMARY | awk '{print $4}') | P=$(echo $SVM_SUMMARY | awk '{print $1}') R=$(echo $SVM_SUMMARY | awk '{print $2}') F1=$(echo $SVM_SUMMARY | awk '{print $3}')"
echo ""

# ----------------------------------------------------------
# 5. CLIP4Clip
# ----------------------------------------------------------
echo "[5/6] CLIP4Clip"
START=$SECONDS
python -m baselines.clip4clip_based 2>&1 | while IFS= read -r line; do
    if echo "$line" | grep -q '^\['; then
        echo "  $line"
    fi
done
ELAPSED=$(($SECONDS - START))
CLIP_TIME="${ELAPSED}s"
CLIP_SUMMARY=$(python -c "
import json
with open('experiments/output/baselines/clip4clip/summary.json') as f:
    s = json.load(f)
print(f\"{s['avg_precision']:.2f} {s['avg_recall']:.2f} {s['avg_f1']:.2f} {s['n_videos']}\")
" 2>/dev/null || echo "FAIL FAIL FAIL FAIL")
echo "  ✓ ${ELAPSED}s | videos=$(echo $CLIP_SUMMARY | awk '{print $4}') | P=$(echo $CLIP_SUMMARY | awk '{print $1}') R=$(echo $CLIP_SUMMARY | awk '{print $2}') F1=$(echo $CLIP_SUMMARY | awk '{print $3}')"
echo ""

# ----------------------------------------------------------
# 6. PGL-SUM
# ----------------------------------------------------------
echo "[6/6] PGL-SUM"
START=$SECONDS
python -m baselines.pglsum_based 2>&1 | while IFS= read -r line; do
    if echo "$line" | grep -q '^\['; then
        echo "  $line"
    fi
done
ELAPSED=$(($SECONDS - START))
PGL_TIME="${ELAPSED}s"
PGL_SUMMARY=$(python -c "
import json
with open('experiments/output/baselines/pglsum/summary.json') as f:
    s = json.load(f)
print(f\"{s['avg_precision']:.2f} {s['avg_recall']:.2f} {s['avg_f1']:.2f} {s['n_videos']}\")
" 2>/dev/null || echo "FAIL FAIL FAIL FAIL")
echo "  ✓ ${ELAPSED}s | videos=$(echo $PGL_SUMMARY | awk '{print $4}') | P=$(echo $PGL_SUMMARY | awk '{print $1}') R=$(echo $PGL_SUMMARY | awk '{print $2}') F1=$(echo $PGL_SUMMARY | awk '{print $3}')"
echo ""

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo "========================================"
printf "%-16s %8s %8s %8s %8s\n" "Baseline" "P" "R" "F1" "Time"
echo "----------------------------------------"
printf "%-16s %8.2f %8.2f %8.2f %8s\n" \
    "Rule-Based" \
    $(echo $RULE_SUMMARY | awk '{print $1}') \
    $(echo $RULE_SUMMARY | awk '{print $2}') \
    $(echo $RULE_SUMMARY | awk '{print $3}') \
    "$RULE_TIME"
printf "%-16s %8.2f %8.2f %8.2f %8s\n" \
    "SVM-Based" \
    $(echo $SVM_SUMMARY | awk '{print $1}') \
    $(echo $SVM_SUMMARY | awk '{print $2}') \
    $(echo $SVM_SUMMARY | awk '{print $3}') \
    "$SVM_TIME"
printf "%-16s %8.2f %8.2f %8.2f %8s\n" \
    "CLIP4Clip" \
    $(echo $CLIP_SUMMARY | awk '{print $1}') \
    $(echo $CLIP_SUMMARY | awk '{print $2}') \
    $(echo $CLIP_SUMMARY | awk '{print $3}') \
    "$CLIP_TIME"
printf "%-16s %8.2f %8.2f %8.2f %8s\n" \
    "PGL-SUM" \
    $(echo $PGL_SUMMARY | awk '{print $1}') \
    $(echo $PGL_SUMMARY | awk '{print $2}') \
    $(echo $PGL_SUMMARY | awk '{print $3}') \
    "$PGL_TIME"
echo "========================================"
echo ""
echo "Results directories:"
echo "  experiments/output/baselines/rule_based/"
echo "  experiments/output/baselines/svm_based/"
echo "  experiments/output/baselines/clip4clip/"
echo "  experiments/output/baselines/pglsum/"
