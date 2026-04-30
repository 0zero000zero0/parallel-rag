#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/home/zdw2200170271/llm/datasets/FlashRAG_datasets"
RETRIEVER_BASE_URL="http://127.0.0.1:9100"

NAVIGATOR_AGENT_OPENAI_BASE_URL="http://127.0.0.1:8000/"
NAVIGATOR_AGENT_OPENAI_API_KEY="TEST"
NAVIGATOR_AGENT_MODEL="Qwen3-32B"
NAVIGATOR_AGENT_MODEL_PATH="/home/zdw2200170271/llm/models/Qwen3-32B"

GLOBAL_REFINE_AGENT_OPENAI_BASE_URL="http://127.0.0.1:8000/"
GLOBAL_REFINE_AGENT_OPENAI_API_KEY="TEST"
GLOBAL_REFINE_AGENT_MODEL="Qwen3-32B"
GLOBAL_REFINE_AGENT_MODEL_PATH="/home/zdw2200170271/llm/models/Qwen3-32B"

NUM_SAMPLES=1024
DATASETS=(
  # bamboogle
  # 2wikimultihopqa
  # hotpotqa
  # musique
  # gpqa
  nq
  popqa
  triviaqa
  ambigqa
)

for DATASET in "${DATASETS[@]}"; do
  INPUT_FILE="${DATA_ROOT}/${DATASET}/test.jsonl"

  if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Skip ${DATASET}: input file not found -> $INPUT_FILE"
    continue
  fi

  DATASET_OUTPUT_ROOT="./outputs/parallel-rag/${NAVIGATOR_AGENT_MODEL}/${DATASET}"
  mkdir -p "$DATASET_OUTPUT_ROOT"

  MAX_INDEX="$(find "$DATASET_OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1 || true)"
  if [[ -z "$MAX_INDEX" ]]; then
    NEXT_INDEX=0
  else
    NEXT_INDEX=$((MAX_INDEX + 1))
  fi

  RESULT_DIR="${DATASET_OUTPUT_ROOT}/${NEXT_INDEX}"
  RESULT_FILE="${RESULT_DIR}/${DATASET}.jsonl"

  echo "=================================================="
  echo "Processing Dataset: ${DATASET}"
  echo "Output Dir: ${RESULT_DIR}"

  python run_parallel_rag.py \
    --input_file "$INPUT_FILE" \
    --result_file "$RESULT_FILE" \
    --batch_size 512 \
    --retriever_base_url "$RETRIEVER_BASE_URL" \
    --retriever_top_k 5 \
    --navigator_agent_openai_base_url "$NAVIGATOR_AGENT_OPENAI_BASE_URL" \
    --navigator_agent_openai_api_key "$NAVIGATOR_AGENT_OPENAI_API_KEY" \
    --navigator_agent_model "$NAVIGATOR_AGENT_MODEL" \
    --navigator_agent_model_path "$NAVIGATOR_AGENT_MODEL_PATH" \
    --navigator_agent_max_tokens 1024 \
    --navigator_agent_temperature 0.6 \
    --navigator_agent_top_p 0.9 \
    --navigator_agent_use_chat_template \
    --global_refine_agent_openai_base_url "$GLOBAL_REFINE_AGENT_OPENAI_BASE_URL" \
    --global_refine_agent_openai_api_key "$GLOBAL_REFINE_AGENT_OPENAI_API_KEY" \
    --global_refine_agent_model "$GLOBAL_REFINE_AGENT_MODEL" \
    --global_refine_agent_model_path "$GLOBAL_REFINE_AGENT_MODEL_PATH" \
    --global_refine_agent_max_tokens 1024 \
    --global_refine_agent_temperature 0.6 \
    --global_refine_agent_top_p 0.9 \
    --global_refine_agent_use_chat_template \
    --synthesize_max_tokens 1024 \
    --synthesize_temperature 0.6 \
    --synthesize_top_p 0.9 \
    --max_iterations 5 \
    --num_samples "$NUM_SAMPLES"

  python evaluate.py \
    --result_file "$RESULT_FILE" \
    --dataset_name "$DATASET" \
    --dataset_path "$INPUT_FILE"

  echo "Finished Dataset: ${DATASET}"
  echo "Result File: ${RESULT_FILE}"
  echo "Metrics File: ${RESULT_DIR}/metrics.json"
done

echo "All datasets done."
