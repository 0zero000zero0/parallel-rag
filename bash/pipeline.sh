#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# -------------------------------
# Config (can be overridden by env)
# -------------------------------
RETRIEVER_PORT="${RETRIEVER_PORT:-9100}"
RETRIEVER_GPU_ID="${RETRIEVER_GPU_ID:-0,1}"
RETRIEVER_NUM="${RETRIEVER_NUM:-1}"

VLLM_MAIN_PORT="${VLLM_MAIN_PORT:-8000}"
VLLM_AUX_PORT="${VLLM_AUX_PORT:-8001}"
VLLM_API_KEY="${VLLM_API_KEY:-TEST}"
ENABLE_AUX_VLLM="${ENABLE_AUX_VLLM:-1}"

RUN_SCRIPT="${RUN_SCRIPT:-bash/run_adaptive_parallel_o1_all.sh}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-900}"

LOG_DIR="${LOG_DIR:-logs/pipeline}"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

PIDS=()

start_bg() {
  local name="$1"
  local cmd="$2"
  local log_file="${LOG_DIR}/${TS}_${name}.log"

  echo "[Pipeline] Starting ${name} ..."
  echo "[Pipeline] Command: ${cmd}"
  bash -lc "${cmd}" >"${log_file}" 2>&1 &
  local pid=$!
  PIDS+=("${pid}")
  echo "[Pipeline] ${name} pid=${pid}, log=${log_file}"
}

port_open() {
  local host="$1"
  local port="$2"
  (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1
}

wait_port() {
  local name="$1"
  local host="$2"
  local port="$3"
  local timeout_sec="$4"
  local start_ts now elapsed

  start_ts=$(date +%s)
  while true; do
    if port_open "${host}" "${port}"; then
      echo "[Pipeline] ${name} is ready at ${host}:${port}"
      return 0
    fi

    now=$(date +%s)
    elapsed=$((now - start_ts))
    if [[ ${elapsed} -ge ${timeout_sec} ]]; then
      echo "[Pipeline] ERROR: ${name} not ready after ${timeout_sec}s (${host}:${port})" >&2
      return 1
    fi

    sleep 2
  done
}

cleanup() {
  local code=$?
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "[Pipeline] Cleaning up server processes ..."
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill "${pid}" >/dev/null 2>&1 || true
      fi
    done
    # Give processes a moment to exit gracefully.
    sleep 2
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill -9 "${pid}" >/dev/null 2>&1 || true
      fi
    done
  fi
  if [[ ${code} -eq 0 ]]; then
    echo "[Pipeline] Finished successfully."
  else
    echo "[Pipeline] Exited with error code ${code}." >&2
  fi
}
trap cleanup EXIT INT TERM

echo "[Pipeline] Repo root: ${REPO_ROOT}"
echo "[Pipeline] Run script: ${RUN_SCRIPT}"

start_bg \
  "retrieval_server" \
  "python retrieval-server/app.py --config_file_path retrieval-server/retriever_config.yaml --num_retriever ${RETRIEVER_NUM} --port ${RETRIEVER_PORT} --gpu_id '${RETRIEVER_GPU_ID}'"

start_bg \
  "vllm_main_32b" \
  "CUDA_VISIBLE_DEVICES='0,1,2,3' vllm serve Qwen3-32B --port ${VLLM_MAIN_PORT} -tp 4 -dp 1 --gpu-memory-utilization 0.8 --api-key '${VLLM_API_KEY}'"

if [[ "${ENABLE_AUX_VLLM}" == "1" ]]; then
  start_bg \
    "vllm_aux_4b" \
    "CUDA_VISIBLE_DEVICES='4,5' vllm serve Qwen3-4B --port ${VLLM_AUX_PORT} -tp 1 -dp 2 --gpu-memory-utilization 0.85 --api-key '${VLLM_API_KEY}'"
fi

echo "[Pipeline] Waiting for services to be ready ..."
wait_port "retrieval_server" "127.0.0.1" "${RETRIEVER_PORT}" "${WAIT_TIMEOUT_SEC}"
wait_port "vllm_main_32b" "127.0.0.1" "${VLLM_MAIN_PORT}" "${WAIT_TIMEOUT_SEC}"
if [[ "${ENABLE_AUX_VLLM}" == "1" ]]; then
  wait_port "vllm_aux_4b" "127.0.0.1" "${VLLM_AUX_PORT}" "${WAIT_TIMEOUT_SEC}"
fi

echo "[Pipeline] All services are up. Running ${RUN_SCRIPT} ..."
bash "${RUN_SCRIPT}"

echo "[Pipeline] Run script completed."
