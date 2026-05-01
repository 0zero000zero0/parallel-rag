#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/home/zdw2200170271/llm/datasets/FlashRAG_datasets"
RETRIEVER_BASE_URL="http://localhost:9100"

# local
# OPENAI_BASE_URL="http://localhost:8000/"
# OPENAI_API_KEY="TEST"

# siliconflow
OPENAI_BASE_URL="https://api.siliconflow.cn"
OPENAI_API_KEY="sk-mezpojwjhugpvysantmdbftvwqmjkqtxaxlhwvmdqmegirhl"
MODEL="Qwen3-32B"
NUM_SAMPLES=1024
DATASET="bamboogle"

input_file="${DATA_ROOT}/${DATASET}/test.jsonl"

python run_search_o1.py \
  --input_file "$input_file" \
  --batch_size 512 \
  --retriever_base_url "$RETRIEVER_BASE_URL" \
  --retriever_top_k 5 \
  --openai_base_url "$OPENAI_BASE_URL" \
  --openai_api_key "$OPENAI_API_KEY" \
  --model "$MODEL" \
  --max_search_limit 5 \
  --search_max_tokens 512 \
  --search_temperature 1.0 \
  --search_top_p 0.9 \
  --refine_max_tokens 1024 \
  --refine_temperature 1.0 \
  --refine_top_p 0.9 \
  --max_iterations 5 \
  --num_samples "$NUM_SAMPLES"
