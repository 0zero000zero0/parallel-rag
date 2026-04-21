#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="/home/zdw2200170271/llm/datasets/FlashRAG_datasets/bamboogle/test.jsonl"
RETRIEVER_BASE_URL="http://localhost:9100"
OPENAI_BASE_URL="http://localhost:8001/"
OPENAI_API_KEY="TEST"
MODEL="Qwen3-14B"

DATASET_NAME="$(basename "$(dirname "$INPUT_FILE")")"
DATASET_OUTPUT_ROOT="outputs/${DATASET_NAME}"

mkdir -p "$DATASET_OUTPUT_ROOT"

MAX_INDEX="$(find "$DATASET_OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1 || true)"
if [[ -z "$MAX_INDEX" ]]; then
  NEXT_INDEX=0
else
  NEXT_INDEX=$((MAX_INDEX + 1))
fi

RESULT_DIR="${DATASET_OUTPUT_ROOT}/${NEXT_INDEX}"
RESULT_FILE="${RESULT_DIR}/output.jsonl"

echo "[Pipeline] Dataset: ${DATASET_NAME}"
echo "[Pipeline] Expected output dir: ${RESULT_DIR}"

echo "[Pipeline] Running run_parallel_rag.py ..."
python run_parallel_rag.py \
  --input_file "$INPUT_FILE" \
  --batch_size 512 \
  --retriever_base_url "$RETRIEVER_BASE_URL" \
  --retriever_top_k 5 \
  --openai_base_url "$OPENAI_BASE_URL" \
  --openai_api_key "$OPENAI_API_KEY" \
  --model "$MODEL" \
  --parallel_query_count 5 \
  --docs_per_query 5 \
  --query_max_tokens 1024 \
  --query_temperature 0.8 \
  --query_top_p 0.9 \
  --refine_max_tokens 1024 \
  --refine_temperature 0.8 \
  --refine_top_p 0.9 \
  --synthesize_max_tokens 1024 \
  --synthesize_temperature 0.8 \
  --synthesize_top_p 0.9 \
  --max_iterations 5 \
  --debug False

if [[ ! -f "$RESULT_FILE" ]]; then
  echo "[Pipeline] ERROR: main.py finished but result file not found: $RESULT_FILE" >&2
  exit 1
fi

echo "[Pipeline] Running evaluate.py ..."
python evaluate.py \
  --result_file "$RESULT_FILE" \
  --dataset_name "$DATASET_NAME" \
  --dataset_path "$INPUT_FILE" \
  --debug False

echo "[Pipeline] Done."
echo "[Pipeline] Result file: $RESULT_FILE"
echo "[Pipeline] Metrics file: ${RESULT_DIR}/metrics.json"
