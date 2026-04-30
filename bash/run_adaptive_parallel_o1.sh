#!/usr/bin/env bash

DATA_ROOT="/home/zdw2200170271/llm/datasets/FlashRAG_datasets"
RETRIEVER_BASE_URL="http://127.0.0.1:9100"
# local
NAVIGATOR_AGENT_OPENAI_BASE_URL="http://127.0.0.1:8000/"
NAVIGATOR_AGENT_OPENAI_API_KEY="TEST"

GLOBAL_REFINE_AGENT_OPENAI_BASE_URL="http://127.0.0.1:8000/"
GLOBAL_REFINE_AGENT_OPENAI_API_KEY="TEST"

MODEL="Qwen3-32B"
NUM_SAMPLES=10
dataset=bamboogle
input_file="${DATA_ROOT}/${dataset}/test.jsonl"

result_file="./output.jsonl"
python run_adaptive_parallel_o1.py \
  --input_file "$input_file" \
  --retriever_base_url "$RETRIEVER_BASE_URL" \
  --retriever_top_k 5 \
  --navigator_agent_openai_base_url "$NAVIGATOR_AGENT_OPENAI_BASE_URL" \
  --navigator_agent_openai_api_key "$NAVIGATOR_AGENT_OPENAI_API_KEY" \
  --navigator_agent_model "$MODEL" \
  --docs_per_query 5 \
  --navigator_agent_max_tokens 1024 \
  --navigator_agent_temperature 0.8 \
  --navigator_agent_top_p 0.9 \
  --navigator_agent_model_path /home/zdw2200170271/llm/models/Qwen3-32B \
  --navigator_agent_use_chat_template \
  --navigator_agent_enable_thinking \
  --path_agent_max_tokens 512 \
  --path_agent_temperature 0.8 \
  --path_agent_top_p 0.9 \
  --global_refine_agent_openai_base_url "$GLOBAL_REFINE_AGENT_OPENAI_BASE_URL" \
  --global_refine_agent_openai_api_key "$GLOBAL_REFINE_AGENT_OPENAI_API_KEY" \
  --global_refine_agent_model "$MODEL" \
  --global_refine_agent_model_path /home/zdw2200170271/llm/models/Qwen3-32B \
  --global_refine_agent_max_tokens 1024 \
  --global_refine_agent_temperature 0.8 \
  --global_refine_agent_top_p 0.9 \
  --global_refine_agent_use_chat_template \
  --global_refine_agent_enable_thinking \
  --max_iterations 5 \
  --num_samples "$NUM_SAMPLES" \
  --path_agent_model "$MODEL" \
  --path_agent_model_path /home/zdw2200170271/llm/models/Qwen3-32B \
  --batch_size 10 \
  --path_agent_use_chat_template \
  --path_agent_enable_thinking
  # --debug


python evaluate.py \
  --result_file "$result_file" \
  --dataset_name "$dataset" \
  --dataset_path "$input_file"
