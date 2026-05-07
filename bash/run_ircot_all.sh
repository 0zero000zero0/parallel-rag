#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/mnt1/zhangdingwen/llm/datasets"
RETRIEVER_BASE_URL="http://127.0.0.1:9000"
OPENAI_BASE_URL="http://127.0.0.1:9101/"
OPENAI_API_KEY="TEST"

MODEL="Qwen3-32B"
NUM_SAMPLES=1024
DATASETS=(
  bamboogle
  2wikimultihopqa
  hotpotqa
  musique
  nq
  popqa
  triviaqa
  ambigqa
  gpqa
  # gaia
)

for dataset in "${DATASETS[@]}"; do
  input_file="${DATA_ROOT}/${dataset}/test.jsonl"

  if [[ ! -f "$input_file" ]]; then
    echo " Skip ${dataset}: input file not found -> $input_file"
    continue
  fi

  dataset_output_root="./outputs/ircot/${MODEL}/${dataset}"
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

  python run/run_ircot.py \
    --input_file "$input_file" \
    --batch_size 512 \
    --retriever_base_url "$RETRIEVER_BASE_URL" \
    --retriever_top_k 5 \
    --openai_base_url "$OPENAI_BASE_URL" \
    --openai_api_key "$OPENAI_API_KEY" \
    --model "$MODEL" \
    --generation_max_tokens 1024 \
    --generation_temperature 0.8 \
    --generation_top_p 0.9 \
    --model_path /mnt1/zhangdingwen/llm/models/Qwen3-32B \
    --max_search_limit 5 \
    --max_iterations 10 \
    --num_samples "$NUM_SAMPLES" \
    --use_chat_template

  if [[ ! -f "$result_file" ]]; then
    echo "ERROR: result file not found after run/run_ircot.py: $result_file" >&2
    exit 1
  fi

  python src/evaluate.py \
    --result_file "$result_file" \
    --dataset_name "$dataset" \
    --dataset_path "$input_file"

  echo "Finished Dataset: ${dataset}"
  echo "Result File: ${result_file}"
  echo "Metrics File: ${result_dir}/metrics.json"
done

echo "All datasets done."

python src/gather_metric.py \
 --method ircot \
 --model Qwen3-32B \
 --outputs_root outputs
