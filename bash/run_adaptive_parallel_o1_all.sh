#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/home/zdw2200170271/llm/datasets/FlashRAG_datasets"
RETRIEVER_BASE_URL="http://127.0.0.1:9100"
# local
NAVIGATOR_AGENT_OPENAI_BASE_URL="http://127.0.0.1:8000/"
NAVIGATOR_AGENT_OPENAI_API_KEY="TEST"
NAVIGATOR_AGENT_MODEL="Qwen3-32B"
NAVIGATOR_AGENT_MODEL_PATH="/home/zdw2200170271/llm/models/Qwen3-32B"

GLOBAL_REFINE_AGENT_OPENAI_BASE_URL="http://127.0.0.1:8000/"
GLOBAL_REFINE_AGENT_OPENAI_API_KEY="TEST"
GLOBAL_REFINE_AGENT_MODEL="Qwen3-32B"
GLOBAL_REFINE_AGENT_MODEL_PATH="/home/zdw2200170271/llm/models/Qwen3-32B"

PATH_AGENT_OPENAI_BASE_URL="http://127.0.0.1:8000/"
PATH_AGENT_OPENAI_API_KEY="TEST"
PATH_AGENT_MODEL="Qwen3-32B"
PATH_AGENT_MODEL_PATH="/home/zdw2200170271/llm/models/Qwen3-32B"

TEMPERATURE=0.8
TOP_P=0.8

NUM_SAMPLES=1024
DATASETS=(
  # bamboogle
  # 2wikimultihopqa
  # hotpotqa
  # musique
  # gpqa

  # popqa
  nq
  # triviaqa
  # ambigqa
)

for dataset in "${DATASETS[@]}"; do
  input_file="${DATA_ROOT}/${dataset}/test.jsonl"

  if [[ ! -f "$input_file" ]]; then
    echo " Skip ${dataset}: input file not found -> $input_file"
    continue
  fi

  dataset_output_root="./outputs/adaptive-parallel-o1/${NAVIGATOR_AGENT_MODEL}/${dataset}"
  mkdir -p "$dataset_output_root"

  max_index="$(find "$dataset_output_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1 || true)"
  if [[ -z "$max_index" ]]; then
    next_index=0
  else
    next_index=$((max_index + 1))
  fi

  echo "=================================================="
  echo "Result Index for ${dataset}: ${next_index}"

  result_dir="${dataset_output_root}/${next_index}"
  result_file="${result_dir}/${dataset}.jsonl"

  echo "Processing Dataset: ${dataset}"
  echo "Output Dir: ${result_dir}"

  python run_adaptive_parallel_o1.py \
    --input_file "$input_file" \
    --batch_size 512 \
    --retriever_base_url "$RETRIEVER_BASE_URL" \
    --retriever_top_k 5 \
    --navigator_agent_openai_base_url "$NAVIGATOR_AGENT_OPENAI_BASE_URL" \
    --navigator_agent_openai_api_key "$NAVIGATOR_AGENT_OPENAI_API_KEY" \
    --navigator_agent_model "$NAVIGATOR_AGENT_MODEL" \
    --navigator_agent_model_path "$NAVIGATOR_AGENT_MODEL_PATH" \
    --navigator_agent_max_tokens 1024 \
    --navigator_agent_temperature ${TEMPERATURE} \
    --navigator_agent_top_p ${TOP_P} \
    --navigator_agent_use_chat_template \
    --navigator_agent_enable_thinking \
    --global_refine_agent_openai_base_url "$GLOBAL_REFINE_AGENT_OPENAI_BASE_URL" \
    --global_refine_agent_openai_api_key "$GLOBAL_REFINE_AGENT_OPENAI_API_KEY" \
    --global_refine_agent_model "$GLOBAL_REFINE_AGENT_MODEL" \
    --global_refine_agent_model_path "$GLOBAL_REFINE_AGENT_MODEL_PATH" \
    --global_refine_agent_max_tokens 1024 \
    --global_refine_agent_temperature ${TEMPERATURE} \
    --global_refine_agent_top_p ${TOP_P} \
    --global_refine_agent_use_chat_template \
    --global_refine_agent_enable_thinking \
    --path_agent_openai_base_url "$PATH_AGENT_OPENAI_BASE_URL" \
    --path_agent_openai_api_key "$PATH_AGENT_OPENAI_API_KEY" \
    --path_agent_model "$PATH_AGENT_MODEL" \
    --path_agent_model_path "$PATH_AGENT_MODEL_PATH" \
    --path_agent_max_tokens 1024 \
    --path_agent_temperature ${TEMPERATURE} \
    --path_agent_top_p ${TOP_P} \
    --path_agent_use_chat_template \
    --path_agent_enable_thinking \
    --max_iterations 5 \
    --num_samples "$NUM_SAMPLES"

  python evaluate.py \
    --result_file "$result_file" \
    --dataset_name "$dataset" \
    --dataset_path "$input_file"

  echo "Finished Dataset: ${dataset}"
  echo "Result File: ${result_file}"
  echo "Metrics File: ${result_dir}/metrics.json"
done

echo "All datasets done."

python gather_metric.py \
 --method adaptive-parallel-o1\
 --model "${NAVIGATOR_AGENT_MODEL}" \
 --outputs_root outputs