#!/usr/bin/env bash

DATA_ROOT="/home/zdw2200170271/llm/datasets/FlashRAG_datasets"
RETRIEVER_BASE_URL="http://127.0.0.1:9100"
# local
OPENAI_BASE_URL="http://127.0.0.1:8000/"
OPENAI_API_KEY="TEST"

MODEL="Qwen3-32B"
NUM_SAMPLES=10
dataset=bamboogle
input_file="${DATA_ROOT}/${dataset}/test.jsonl"

result_file="./output.jsonl"
python run_parallel_o1.py \
  --input_file "$input_file" \
  --retriever_base_url "$RETRIEVER_BASE_URL" \
  --retriever_top_k 5 \
  --openai_base_url "$OPENAI_BASE_URL" \
  --openai_api_key "$OPENAI_API_KEY" \
  --model "$MODEL" \
  --docs_per_query 5 \
  --navigator_agent_max_tokens 1024 \
  --navigator_agent_temperature 0.8 \
  --navigator_agent_top_p 0.9 \
  --path_agent_max_tokens 512 \
  --path_agent_temperature 0.8 \
  --path_agent_top_p 0.9 \
  --refine_max_tokens 1024 \
  --refine_temperature 0.8 \
  --refine_top_p 0.9 \
  --max_iterations 5 \
  --num_samples "$NUM_SAMPLES" \
  --model_path /home/zdw2200170271/llm/models/Qwen3-32B \
  --batch_size 10 \
  --use_chat_template
  # --debug


python evaluate.py \
  --result_file "$result_file" \
  --dataset_name "$dataset" \
  --dataset_path "$input_file"
