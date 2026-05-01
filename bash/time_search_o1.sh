#!/usr/bin/env bash

DATA_ROOT="/home/zdw2200170271/llm/datasets/FlashRAG_datasets"
RETRIEVER_BASE_URL="http://127.0.0.1:9100"
# local
OPENAI_BASE_URL="http://127.0.0.1:8000/"
OPENAI_API_KEY="TEST"

MODEL="Qwen3-32B"
NUM_SAMPLES=1024
dataset=bamboogle
input_file="${DATA_ROOT}/${dataset}/test.jsonl"
output_dir="time-test"

echo "testing running time on ${dataset} dataset"

python run_search_o1.py \
  --input_file "$input_file" \
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
  --num_samples "$NUM_SAMPLES" \
  --model_path /home/zdw2200170271/llm/models/Qwen3-32B \
  --batch_size 512 \
  --output_top_dir "$output_dir"
#   --use_chat_template \

echo "finished testing running-time on ${dataset}, all files are saved in ${output_dir}"
