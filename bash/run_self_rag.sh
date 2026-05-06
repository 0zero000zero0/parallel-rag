#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/home/zdw2200170271/llm/datasets/FlashRAG_datasets"
RETRIEVER_BASE_URL="http://127.0.0.1:9100"

OPENAI_BASE_URL="http://127.0.0.1:8000/"
OPENAI_API_KEY="TEST"

MODEL="Qwen3-32B"
NUM_SAMPLES=10
DATASET="bamboogle"
INPUT_FILE="${DATA_ROOT}/${DATASET}/test.jsonl"
RESULT_FILE="./output.jsonl"

python run_self_rag.py \
  --input_file "$INPUT_FILE" \
  --result_file "$RESULT_FILE" \
  --batch_size 10 \
  --retriever_base_url "$RETRIEVER_BASE_URL" \
  --retriever_top_k 5 \
  --openai_base_url "$OPENAI_BASE_URL" \
  --openai_api_key "$OPENAI_API_KEY" \
  --model "$MODEL" \
  --generation_max_tokens 1024 \
  --generation_temperature 0.8 \
  --generation_top_p 0.9 \
  --model_path /home/zdw2200170271/llm/models/Qwen3-32B \
  --max_search_limit 5 \
  --max_iterations 10 \
  --num_samples "$NUM_SAMPLES" \
  --use_chat_template

python evaluate.py \
  --result_file "$RESULT_FILE" \
  --dataset_name "$DATASET" \
  --dataset_path "$INPUT_FILE"
