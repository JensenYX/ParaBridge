#!/bin/bash
# ============================================================================
# ParaBridge training (LoRA, on-policy self-distillation)
#
# Trains a single SLM as both teacher (with paralinguistic scaffold) and
# student (without scaffold). Token-level JSD on student rollouts pulls the
# scaffold-free distribution toward the scaffolded distribution, so the
# scaffold can be dropped at inference.
#
# Default config matches the cv+cp / 1k-sample setting used in the paper.
# Edit the variables below to point to your model checkpoint and training
# jsonl. All paths are read from the environment if set, otherwise default to
# the in-repo sample data so a smoke run works out of the box.
# ============================================================================

set -e
export no_proxy=localhost,127.0.0.1

# --------------------------------------------------------------------------
# 0. User-configurable paths
# --------------------------------------------------------------------------
# Repository root (this file lives in <repo>/parabridge/, so go one up).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# MODEL: directory holding a Qwen3-Omni-thinking checkpoint. Download from:
#   https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking
# and either set MODEL=/path/to/checkpoint or place the snapshot at
# ./models/Qwen3-Omni-30B-A3B-Thinking.
MODEL="${MODEL:-${REPO_ROOT}/models/Qwen3-Omni-30B-A3B-Thinking}"

# DATA: training jsonl. The shipped sample at data/sample_train.jsonl uses
# paths relative to the repo root and is resolved to absolute paths below
# before being handed to ms-swift.
DATA_REL="${DATA_REL:-data/sample_train.jsonl}"
DATA_ABS="${REPO_ROOT}/${DATA_REL%.jsonl}.resolved.jsonl"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/parabridge_cv_cp_1000}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/logs}"
mkdir -p "${OUTPUT_DIR}" "${LOGDIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VLLM_LOG="${LOGDIR}/vllm_${TIMESTAMP}.log"
TRAIN_LOG="${LOGDIR}/train_${TIMESTAMP}.log"

# --------------------------------------------------------------------------
# 1. Activate environment (assumes conda env named ms-swift, see README)
# --------------------------------------------------------------------------
if [ -n "${CONDA_EXE}" ] && [ -z "${PARABRIDGE_SKIP_CONDA}" ]; then
    CONDA_BASE="$(${CONDA_EXE} info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-ms-swift}"
fi

# --------------------------------------------------------------------------
# 2. Resolve audio paths in the jsonl (relative -> absolute)
# --------------------------------------------------------------------------
python parabridge/prepare_data.py \
    --input  "${REPO_ROOT}/${DATA_REL}" \
    --output "${DATA_ABS}" \
    --root   "${REPO_ROOT}"

# --------------------------------------------------------------------------
# 3. Free any stale rollout/rlhf processes from a previous failed run
# --------------------------------------------------------------------------
echo "=== Cleaning up leftover processes ==="
kill -9 $(ps aux | grep -E "swift rollout|rollout.py|swift rlhf|rlhf.py" \
          | grep -v grep | awk '{print $2}') 2>/dev/null || true
sleep 3

# --------------------------------------------------------------------------
# 4. Pick a free TCP port for the vLLM <-> trainer weight-sync group
#    (61000-64999 is outside the typical Linux ephemeral range and avoids
#    clashes with outbound TCP clients on the same host).
# --------------------------------------------------------------------------
GROUP_PORT=$(python3 -c "
import socket, random
for _ in range(200):
    port = random.randint(61000, 64999)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            s.bind(('0.0.0.0', port))
        print(port)
        break
    except OSError:
        continue
")
if [ -z "${GROUP_PORT}" ]; then
    echo 'ERROR: could not find a free port in 61000-64999' >&2
    exit 1
fi
echo "Using group port: ${GROUP_PORT}"

# --------------------------------------------------------------------------
# 5. GPU layout: 1 GPU for the rollout server, the rest for the trainer
# --------------------------------------------------------------------------
ROLLOUT_GPU="${ROLLOUT_GPU:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
VLLM_PORT="${VLLM_PORT:-8000}"

echo "============================================"
echo "ParaBridge training - $(date)"
echo "Model:  ${MODEL}"
echo "Data:   ${DATA_ABS}"
echo "Output: ${OUTPUT_DIR}"
echo "Log:    ${TRAIN_LOG}"
echo "============================================"

# --------------------------------------------------------------------------
# 6. Start vLLM rollout server (serves student rollouts to the trainer)
# --------------------------------------------------------------------------
echo "=== Starting vLLM rollout server ==="
CUDA_VISIBLE_DEVICES=${ROLLOUT_GPU} nohup swift rollout \
    --model "${MODEL}" \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 4096 \
    --vllm_limit_mm_per_prompt '{"audio":1}' \
    --port ${VLLM_PORT} \
    --vllm_engine_kwargs '{"load_format":"auto"}' \
    > "${VLLM_LOG}" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 120); do
    grep -q 'Uvicorn running' "${VLLM_LOG}" 2>/dev/null && \
        echo "Server ready after ${i}x3s" && break
    if [ $i -eq 120 ]; then
        echo 'Timeout waiting for vLLM server'
        tail -20 "${VLLM_LOG}"
        kill $SERVER_PID
        exit 1
    fi
    sleep 3
done
grep 'Model loading took' "${VLLM_LOG}" || true

# --------------------------------------------------------------------------
# 7. Run the ParaBridge trainer
#    Key arguments:
#      --rlhf_type gkd                 base trainer family (token-level KD)
#      --model / --teacher_model       same checkpoint for student & teacher
#      --tuner_type lora               LoRA on all linear projections
#      --beta 0.5 --temperature 1.2    JSD mixing coefficient & temperature
#      --lmbda 1.0                     fully on-policy: every step is a fresh
#                                      student rollout
#      --gkd_rollout_batch_size 20     queries per rollout call to vLLM
#      --use_vllm true                 student rollouts served by vLLM
# --------------------------------------------------------------------------
echo "=== Starting ParaBridge trainer ==="
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
PYTORCH_ALLOC_CONF=expandable_segments:True \
swift rlhf \
    --rlhf_type gkd \
    --model "${MODEL}" --teacher_model "${MODEL}" \
    --tuner_type lora --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 \
    --target_modules all-linear \
    --freeze_vit true --freeze_aligner true \
    --dataset "${DATA_ABS}" \
    --lmbda 1.0 --beta 0.5 --temperature 1.2 --sft_alpha 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_ratio 0.1 \
    --logging_steps 1 --save_steps 50 --save_only_model true --save_total_limit 3 \
    --max_length 4096 --max_completion_length 2048 \
    --gradient_checkpointing true --deepspeed zero3 --attn_impl flash_attn \
    --output_dir "${OUTPUT_DIR}" \
    --use_vllm true --vllm_mode server \
    --vllm_server_host 127.0.0.1 --vllm_server_port ${VLLM_PORT} \
    --vllm_server_group_port ${GROUP_PORT} \
    --vllm_max_model_len 4096 \
    --gkd_rollout_batch_size 20 \
    --move_model_batches 8 \
    --report_to swanlab \
    2>&1 | tee "${TRAIN_LOG}"

# --------------------------------------------------------------------------
# 8. Stop the rollout server
# --------------------------------------------------------------------------
echo "=== Cleanup ==="
pkill -9 -P $SERVER_PID 2>/dev/null || true
kill -9 $SERVER_PID 2>/dev/null || true
kill -9 $(ps aux | grep -E "swift rollout|rollout.py" | grep -v grep | awk '{print $2}') 2>/dev/null || true

echo
echo "============================================"
echo "Training complete - $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "Log:    ${TRAIN_LOG}"
echo "============================================"
