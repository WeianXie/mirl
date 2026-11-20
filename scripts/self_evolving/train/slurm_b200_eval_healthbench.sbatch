#!/bin/bash
#SBATCH --job-name=hb_pro_eval
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --exclude=b0017
#
# One-shot OFFICIAL HealthBench Professional eval of a model (all 525 tasks),
# graded with OpenAI's reference simple-evals rubric grader — the leaderboard-
# comparable number, as opposed to the in-loop val score of the training run.
#
# Port of dvdai's aicr/eval_healthbench.sbatch to weianxie's layout. Grader is
# gpt-5.4_2026-03-05 at low effort via TRAPI (dvdai regraded every Table-2 row with
# it in Aug 2026 after gpt-5.3-chat was retired). The model under test is served
# with this SIF's vLLM using the same B200 flash-attn v2 pin as
# slurm_b200_selfimprove.sbatch's chat server (known-good on this image).
#
#   MODEL_ID   HF id or local HF-format dir (required)
#   LABEL      short label for the output dir (required)
#   THINKING   "off" -> --disable-thinking (Qwen family); default leaves thinking on
#   MAXTOK     completion budget (default 2048, dvdai's sweep setting; raise for
#              thinking-on runs or the answer can be cut off)
#   GRADER     TRAPI grader deployment (default gpt-5.4_2026-03-05)
#
# A verl FSDP checkpoint must be merged to HF format first, e.g.:
#   apptainer exec --bind /work:/work --bind /scratch:/scratch $SIF python -m verl.model_merger merge --backend fsdp \
#     --local_dir $CKPT/global_step_60/actor --target_dir $CKPT/global_step_60/hf_merged
#
#   MODEL_ID=Qwen/Qwen3.5-9B LABEL=qwen35_9b_base sbatch .../slurm_b200_eval_healthbench.sbatch
#   MODEL_ID=$CKPT/global_step_60/hf_merged LABEL=hbpro_9b_step60 sbatch ...

set -euo pipefail

WORK=/work/mit/ppliang_mit/weianxie
REPO=$WORK/mirl
SIF=${SIF:-$WORK/sif/verl-selfevolving-cu130-vllm0.22.1.sif}
HF_CACHE=${HF_CACHE:-$WORK/hf_cache}
MODEL_ID=${MODEL_ID:?set MODEL_ID}
LABEL=${LABEL:?set LABEL}
MAXTOK=${MAXTOK:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
GRADER=${GRADER:-gpt-5.4_2026-03-05}
TRAPI_BASE=${TRAPI_BASE:-http://point.dd.works:18890/v1}
TRAPI_KEY_FILE=${TRAPI_KEY_FILE:-$WORK/.trapi_key}
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=$WORK/self_evolving_logs/healthbench_pro_eval/$LABEL
mkdir -p "$OUT" "$HF_CACHE"

[ -s "$TRAPI_KEY_FILE" ] || { echo "FATAL: TRAPI key file missing/empty: $TRAPI_KEY_FILE" >&2; exit 1; }
TRAPI_KEY=$(<"$TRAPI_KEY_FILE")
THINK_FLAG=""
[[ "${THINKING:-}" == "off" ]] && THINK_FLAG="--disable-thinking"

unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
export HF_HOME=$HF_CACHE HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
# The SIF lacks two pure-python deps of the eval script; install once into a
# user site on /work shared by all eval jobs.
export PYTHONUSERBASE=$WORK/pyuser
mkdir -p "$PYTHONUSERBASE"
# Compile caches on node-local tmp, NOT $HOME (100 GiB home quota; concurrent
# torch-compile caches blew it on 2026-08-11 and killed vLLM startup).
JOBCACHE=/tmp/hbev_cache_${SLURM_JOB_ID}
mkdir -p "$JOBCACHE"
export XDG_CACHE_HOME=$JOBCACHE TORCHINDUCTOR_CACHE_DIR=$JOBCACHE/inductor VLLM_CACHE_ROOT=$JOBCACHE/vllm
RUN="apptainer exec --nv --writable-tmpfs --bind $WORK:$WORK --bind /scratch:/scratch --bind /tmp:/tmp $SIF"

$RUN python -c 'import blobfile, jinja2' 2>/dev/null \
    || $RUN python -m pip install --user --no-warn-script-location blobfile jinja2

# Pre-download weights so download time never eats the health-wait budget.
[[ -e "$MODEL_ID" ]] || $RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')"

$RUN vllm serve "$MODEL_ID" --trust-remote-code --dtype bfloat16 \
    --host 127.0.0.1 --port "$PORT" --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    --attention-config '{"flash_attn_version": 2}' \
    > "$OUT/vllm_serve.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT INT TERM
start=$SECONDS
until curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died during startup:" >&2; tail -50 "$OUT/vllm_serve.log" >&2; exit 1; }
    (( SECONDS - start > 3600 )) && { echo "vLLM never became healthy" >&2; exit 1; }
    sleep 15
done
echo "vLLM healthy on :$PORT after $((SECONDS - start))s"

cd "$REPO"
$RUN python scripts/self_evolving/eval/healthbench_professional_eval.py \
    --model-base "http://127.0.0.1:$PORT/v1" --model-name "$MODEL_ID" \
    --model-key EMPTY --model-provider vllm \
    --temperature 0.0 --max-tokens "$MAXTOK" $THINK_FLAG \
    --grader trapi --grader-base "$TRAPI_BASE" --grader-key "$TRAPI_KEY" \
    --grader-model "$GRADER" --grader-effort low \
    --concurrency 16 \
    --simple-evals-dir "$WORK/simple_evals_src" \
    --output-dir "$OUT" \
    --no-wandb
