#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="$HOME/llm/datasets/FlashRAG_datasets"
OPENAI_BASE_URL="http://127.0.0.1:9101/"
OPENAI_API_KEY="TEST"

MODEL="Qwen3-32B"
NUM_SAMPLES=10
dataset="bamboogle"
input_file="${DATA_ROOT}/${dataset}/test.jsonl"
result_file="./output.jsonl"

python run/run_naive_generation.py \
  --input_file "$input_file" \
  --result_file "$result_file" \
  --openai_base_url "$OPENAI_BASE_URL" \
  --openai_api_key "$OPENAI_API_KEY" \
  --model "$MODEL" \
  --generation_max_tokens 1024 \
  --generation_temperature 0.8 \
  --generation_top_p 0.9 \
  --num_samples "$NUM_SAMPLES" \
  --model_path ~/llm/models/Qwen3-32B \
  --batch_size 10 \
  --use_chat_template

python src/evaluate.py \
  --result_file "$result_file" \
  --dataset_name "$dataset" \
  --dataset_path "$input_file"