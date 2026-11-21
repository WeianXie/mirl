#!/bin/bash
#SBATCH --job-name=hb_pro_proposer
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=hb_pro_proposer_%j.out
#
# Proposer-only HealthBench Pro novelty experiment: control vs Best-of-N, no trainer, no GPU.
#
# Runs the same gen server as slurm_b200_healthbench.sbatch (same flags), one per arm, side
# by side so both see the same TRAPI load. In rubric mode the proposer is open-loop (trainer
# /report feedback only evicts already-served tasks, /evolve is off), so what it generates
# does not depend on the trainer. Without a trainer nothing drains the pool, so
# --max_pool_size is the stop: workers back off once TARGET tasks are accepted. The job
# then stops both servers and writes the comparison report.
#
# Measures: candidate/task novelty vs history, similarity to the val prompts, intra-batch
# similarity, category mix + entropy, cluster coverage, validator funnel.
# Does NOT measure: anything needing rollouts (task scores, solvability, accuracy growth).
#
#   sbatch scripts/self_evolving/train/slurm_cpu_healthbench_proposer.sbatch
#   TARGET=960 BON_N=4 sbatch scripts/self_evolving/train/slurm_cpu_healthbench_proposer.sbatch
set -euo pipefail

WORK=/work/mit/ppliang_mit/weianxie
REPO=$WORK/mirl
SIF=${SIF:-$WORK/sif/verl-selfevolving-cu130-vllm0.22.1.sif}
DATA_DIR=${DATA_DIR:-$WORK/self_evolving_datasets/healthbench_pro}
LOG_ROOT=${LOG_DIR:-$WORK/self_evolving_logs/healthbench}
VAL=$DATA_DIR/healthbench_pro_val.parquet
TRAPI_BASE=${TRAPI_BASE:-http://point.dd.works:18890/v1}
TRAPI_KEY_FILE=${TRAPI_KEY_FILE:-$WORK/.trapi_key}
JUDGE=${JUDGE:-gpt-chat-latest_2026-05-28}
EMBED_BASE=${EMBED_BASE:-http://mib.media.mit.edu:18001/v1}
MILVUS_URI=${MILVUS_URI:-http://mib.media.mit.edu:19531}
# 60 trainer steps x train_batch_size 32 = what one full arm would consume.
TARGET=${TARGET:-1920}
BON_N=${BON_N:-4}
TAG=${TAG:-}   # e.g. TAG=_r2 for a repeat; log dirs must not already hold gen data
# arm name : NOVELTY_N : port
ARMS=("control:1:8051" "bon${BON_N}:${BON_N}:8053")
EXP_PREFIX=hbpro_proposer_only
REPORT_DIR=$LOG_ROOT/report_${EXP_PREFIX}${TAG}

[ -f "$VAL" ] || { echo "FATAL: val set missing: $VAL" >&2; exit 1; }
[ -s "$TRAPI_KEY_FILE" ] || { echo "FATAL: TRAPI key file missing/empty: $TRAPI_KEY_FILE" >&2; exit 1; }
TRAPI_KEY=$(<"$TRAPI_KEY_FILE")
unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
export HF_HOME=$WORK/hf_cache
export CHAT_PROVIDER=trapi
export TRAPI_NO_THINK_EFFORT=""   # gpt-chat-latest rejects reasoning_effort="none"
RUN_CPU="apptainer exec --writable-tmpfs --bind $WORK:$WORK $SIF"

probe_trapi() {
    curl -s -m 30 -H "Authorization: Bearer $TRAPI_KEY" -H "Content-Type: application/json" \
        -d "{\"model\":\"$JUDGE\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_completion_tokens\":16}" \
        "$TRAPI_BASE/chat/completions" | grep -q '"choices"'
}
start=$SECONDS
until probe_trapi; do
    (( SECONDS - start > ${TRAPI_WAIT_S:-600} )) && { echo "FATAL: TRAPI '$JUDGE' not answering at $TRAPI_BASE" >&2; exit 1; }
    sleep 30
done
echo "TRAPI judge $JUDGE healthy"
curl -sf -m 15 "$EMBED_BASE/models" >/dev/null || { echo "FATAL: embed server unreachable at $EMBED_BASE" >&2; exit 1; }

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

wait_healthy() {  # $1=url $2=pid $3=name $4=timeout_s $5=log
    local t0=$SECONDS
    until curl -sf -m 5 "$1" > /dev/null; do
        kill -0 "$2" 2>/dev/null || { echo "FATAL: $3 died during startup:" >&2; tail -40 "$5" >&2; exit 1; }
        (( SECONDS - t0 > $4 )) && { echo "FATAL: $3 not healthy after $4 s" >&2; exit 1; }
        sleep 5
    done
}

cd "$REPO"
declare -A DIR PID
for spec in "${ARMS[@]}"; do
    IFS=: read -r arm n port <<< "$spec"
    d=$LOG_ROOT/${EXP_PREFIX}_${arm}${TAG}
    if compgen -G "$d/gen/server_*.jsonl" > /dev/null; then
        echo "FATAL: $d/gen already has data; set TAG=... or move it aside (report would mix runs)" >&2; exit 1
    fi
    mkdir -p "$d/gen" "$d/prompts"
    DIR[$arm]=$d
    snap="$d/code_${SLURM_JOB_ID:-local}"
    { git rev-parse HEAD; git status --short; } > "$snap.txt" 2>&1 || true
    git diff HEAD > "$snap.diff" 2>/dev/null || true
    git ls-files -o --exclude-standard -z | tar --null -czf "$snap.untracked.tgz" -T - 2>/dev/null || true
    if curl -sf -m 5 "localhost:$port/healthz" >/dev/null 2>&1; then
        echo "FATAL: port $port already serving (orphan?)" >&2; exit 1
    fi
    # Same flags as slurm_b200_healthbench.sbatch (RETRIEVAL=0) except --max_pool_size.
    GEN_SERVER_API_KEY="$TRAPI_KEY" $RUN_CPU python scripts/self_evolving/generation_server.py \
        --rubric_mode --prompt_dir "$d/prompts" \
        --api_base "$TRAPI_BASE" --model_name "$JUDGE" \
        --embed_api_base "$EMBED_BASE" --embed_model Qwen/Qwen3-VL-Embedding-2B \
        --milvus_uri "$MILVUS_URI" --milvus_token root:Milvus \
        --milvus_collection medical_knowledge_v2 --milvus_top_k 8 \
        --retrieve_top_k "${RETRIEVE_TOP_K:-5}" --retrieve_total "${RETRIEVE_TOTAL:-16}" \
        --n_queries "${N_QUERIES:-6}" --questions_per_query 1 \
        --novelty_best_of_n "$n" --novelty_ref_parquet "$VAL" \
        --accuracy_window 64 --max_pool_size "$TARGET" \
        --workers "${GEN_WORKERS:-8}" --log_dir "$d/gen" \
        --host 127.0.0.1 --port "$port" \
        > "$d/gen_server_${SLURM_JOB_ID:-local}.log" 2>&1 &
    PIDS+=($!); PID[$arm]=$!
    echo "$arm: NOVELTY_N=$n port=$port pid=${PID[$arm]} log_dir=$d"
done
for spec in "${ARMS[@]}"; do
    IFS=: read -r arm n port <<< "$spec"
    wait_healthy "http://localhost:$port/healthz" "${PID[$arm]}" "gen server ($arm)" 900 \
        "${DIR[$arm]}/gen_server_${SLURM_JOB_ID:-local}.log"
    echo "gen server ($arm) up on :$port"
done

# ---- wait until each arm has TARGET accepted tasks (pool full = workers idle) -----
accepted() { cat "$1"/gen/server_accepted_*.jsonl 2>/dev/null | wc -l || true; }
# Plain counter: ${#DONE[@]} on an empty assoc array is "unbound" under set -u.
declare -A DONE=()
ndone=0
while (( ndone < ${#ARMS[@]} )); do
    line="$(date '+%F %T')"
    for spec in "${ARMS[@]}"; do
        IFS=: read -r arm n port <<< "$spec"
        if [ -n "${DONE[$arm]:-}" ]; then line+=" $arm=${DONE[$arm]}"; continue; fi
        a=$(accepted "${DIR[$arm]}")
        line+=" $arm=$a/$TARGET"
        if ! kill -0 "${PID[$arm]}" 2>/dev/null; then
            echo "$arm gen server exited early at $a accepted" >&2; tail -40 "${DIR[$arm]}/gen_server_${SLURM_JOB_ID:-local}.log" >&2 || true
            DONE[$arm]=died; ndone=$((ndone + 1))
        elif (( a >= TARGET )); then
            DONE[$arm]=done; ndone=$((ndone + 1))
        fi
    done
    echo "$line"
    if (( ndone < ${#ARMS[@]} )); then sleep 60; fi
done
# Grace for in-flight iterations to flush their candidate rows.
sleep 120
cleanup
wait 2>/dev/null || true
echo "both arms stopped"

# ---- comparison report ------------------------------------------------------------
RUN_ARGS=()
for spec in "${ARMS[@]}"; do IFS=: read -r arm n port <<< "$spec"; RUN_ARGS+=(--run "$arm=${DIR[$arm]}"); done
$RUN_CPU python -I scripts/self_evolving/report_novelty.py "${RUN_ARGS[@]}" --out_dir "$REPORT_DIR"
echo "report: $REPORT_DIR/report.html"
