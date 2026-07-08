#!/bin/bash
# LatentSync vs MuseTalk sync + sharpness eval driver.
# LSE-C/D via the standard SyncNet (Chung&Zisserman) eval shipped with LatentSync.
set -e
cd /home/z/work/LatentSync
export PATH=/home/z/bin:$PATH
PY=/home/z/work/LatentSync/.venv-ls/bin/python

LATENTSYNC_OUT=${1:-/home/z/work/LatentSync/zen-dub-latentsync.mp4}
MUSETALK_OUT=${2:-/home/z/work/zen-dub-run/zen-dub-fullnative.mp4}
SRC_VIDEO=${3:-/home/z/work/zen-dub-run/fn_work/video.mp4}

echo "############################################################"
echo "# SYNC EVAL (SyncNet LSE-C / LSE-D)"
echo "############################################################"
echo ""
echo "===== LatentSync output: $LATENTSYNC_OUT ====="
$PY -m eval.eval_sync_conf --video_path "$LATENTSYNC_OUT" || echo "LatentSync eval FAILED"
echo ""
echo "===== MuseTalk output: $MUSETALK_OUT ====="
$PY -m eval.eval_sync_conf --video_path "$MUSETALK_OUT" || echo "MuseTalk eval FAILED"
echo ""
echo "############################################################"
echo "# MOUTH SHARPNESS (Laplacian variance, mouth crop)"
echo "############################################################"
echo ""
echo "----- LatentSync vs source -----"
$PY eval/mouth_sharpness.py --gen "$LATENTSYNC_OUT" --src "$SRC_VIDEO" || echo "sharpness FAILED"
echo ""
echo "----- MuseTalk vs source -----"
$PY eval/mouth_sharpness.py --gen "$MUSETALK_OUT" --src "$SRC_VIDEO" || echo "sharpness FAILED"
