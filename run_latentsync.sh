#!/bin/bash
# LatentSync 1.6 inference on the sun video + dub audio, with FPS timing.
set -e
cd /home/z/work/LatentSync
export PATH=/home/z/bin:$PATH
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/z/work/LatentSync/.venv-ls/bin/python

VIDEO=${1:-/home/z/work/zen-dub-run/fn_work/video.mp4}
AUDIO=${2:-/home/z/work/zen-dub-run/fn_work/tts_en.wav}
OUT=${3:-/home/z/work/LatentSync/zen-dub-latentsync.mp4}

echo "video=$VIDEO"
echo "audio=$AUDIO"
echo "out=$OUT"
ffmpeg -version >/dev/null 2>&1 && echo "ffmpeg OK" || echo "ffmpeg MISSING"

START=$(date +%s.%N)
$PY -m scripts.inference \
  --unet_config_path "configs/unet/stage2_512.yaml" \
  --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
  --inference_steps 20 \
  --guidance_scale 1.5 \
  --enable_deepcache \
  --video_path "$VIDEO" \
  --audio_path "$AUDIO" \
  --video_out_path "$OUT"
END=$(date +%s.%N)

ELAPSED=$(echo "$END - $START" | bc)
echo "INFERENCE_WALL_SECONDS=$ELAPSED"
# frame count of output
FRAMES=$(ffmpeg -nostdin -i "$OUT" -map 0:v:0 -c copy -f null - 2>&1 | grep -oE "frame=[ ]*[0-9]+" | tail -1 | grep -oE "[0-9]+")
echo "OUTPUT_FRAMES=$FRAMES"
echo "FPS=$(echo "scale=3; $FRAMES / $ELAPSED" | bc)"
ls -la "$OUT"
