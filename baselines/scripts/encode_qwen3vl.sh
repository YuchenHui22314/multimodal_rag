#!/bin/bash
# encode_qwen3vl.sh
#
# Launch script for encode_qwen3vl.py (Qwen3-VL-Embedding-2B encoding pipeline).
#
# Usage:
#   sh scripts/encode_qwen3vl.sh --mode notes --cuda 0,1,2,3
#   sh scripts/encode_qwen3vl.sh --mode queries --split search_test --cuda 0,1,2,3
#   sh scripts/encode_qwen3vl.sh --mode notes --cuda 0 --no_int8 --dim 0   # fp16, full 2048 dims
#
# Key options:
#   --mode      : notes | queries
#   --split     : dataset split for query encoding (default: search_test)
#   --cuda      : comma-separated GPU indices to use
#   --no_int8   : disable int8 inference (use fp16)
#   --dim       : Matryoshka output dim (default 768; 0 = full 2048)
#   --batch_size: per-GPU batch size (default 8; reduce if OOM)

set -e

# ---- defaults ----
MODE=""
SPLIT="search_test"
CUDA_DEVICES=""
EXTRA_ARGS=""

# ---- parse args ----
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode)       MODE="$2"; shift ;;
        --split)      SPLIT="$2"; shift ;;
        --cuda)       CUDA_DEVICES="$2"; shift ;;
        # pass-through any other args to the Python script
        *)            EXTRA_ARGS="$EXTRA_ARGS $1" ;;
    esac
    shift
done

if [[ -z "$MODE" ]]; then
    echo "Error: --mode (notes or queries) is required."
    exit 1
fi

# ---- GPU selection ----
if [[ -z "$CUDA_DEVICES" ]]; then
    CUDA_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
fi
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
NUM_GPUS=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)
echo "Using GPUs: $CUDA_DEVICES  (nproc_per_node=$NUM_GPUS)"

# ---- log file ----
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_DIR=/data/rech/huiyuche/TREC_iKAT_2024/logs
mkdir -p $LOG_DIR
LOG_FILE=${LOG_DIR}/encode_qwen3vl_${MODE}_${TIMESTAMP}.log
echo "Log: $LOG_FILE"

# ---- find free port ----
PORT=29581
while lsof -i:"$PORT" > /dev/null 2>&1; do
    echo "Port $PORT in use, trying next..."
    PORT=$((PORT + 1))
done
echo "Using port: $PORT"

# ---- run ----
cd "$(dirname "$0")/.."  # cd to baselines/

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    src/encode_qwen3vl.py \
    --mode "$MODE" \
    --split "$SPLIT" \
    --log_path "$LOG_FILE" \
    $EXTRA_ARGS \
    2>&1 | tee -a "$LOG_FILE"

echo "=== encode_qwen3vl.sh done ==="
