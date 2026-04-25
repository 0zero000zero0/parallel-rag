#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/home/zdw2200170271/llm/datasets/FlashRAG_datasets"
RETRIEVER_BASE_URL="http://127.0.0.1:9100"
OPENAI_BASE_URL="http://127.0.0.1:8000/"
OPENAI_API_KEY="TEST"

MODEL="Qwen3-32B"
NUM_SAMPLES=1024
DATASETS=(
  # bamboogle
  # 2wikimultihopqa
  # hotpotqa
  # musique
  # nq
  # popqa
  # triviaqa
  # ambigqa
  # gpqa
  gaia
)

for dataset in "${DATASETS[@]}"; do
  input_file="${DATA_ROOT}/${dataset}/test.jsonl"

  if [[ ! -f "$input_file" ]]; then
    echo "[All] Skip ${dataset}: input file not found -> $input_file"
    continue
  fi

  dataset_output_root="./outputs/naive-rag/${MODEL}/${dataset}"
  mkdir -p "$dataset_output_root"

  max_index="$(find "$dataset_output_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1 || true)"
  if [[ -z "$max_index" ]]; then
    next_index=0
  else
    next_index=$((max_index + 1))
  fi

  result_dir="${dataset_output_root}/${next_index}"
  result_file="${result_dir}/${dataset}.jsonl"

  echo "[All]=================================================="
  echo "[All] Dataset: ${dataset}"
  echo "[All] Expected output dir: ${result_dir}"

  python run_naive_rag.py \
    --input_file "$input_file" \
    --batch_size 512 \
    --retriever_base_url "$RETRIEVER_BASE_URL" \
    --retriever_top_k 5 \
    --openai_base_url "$OPENAI_BASE_URL" \
    --openai_api_key "$OPENAI_API_KEY" \
    --model "$MODEL" \
    --docs_per_query 5 \
    --generation_max_tokens 1024 \
    --generation_temperature 0.8 \
    --generation_top_p 0.9 \
    --model_path /home/zdw2200170271/llm/models/Qwen3-32B \
    --num_samples "$NUM_SAMPLES" \
    --use_chat_template

  if [[ ! -f "$result_file" ]]; then
    echo "[All] ERROR: result file not found after run_naive_rag.py: $result_file" >&2
    exit 1
  fi

  python evaluate.py \
    --result_file "$result_file" \
    --dataset_name "$dataset" \
    --dataset_path "$input_file"

  echo "[All] Finished ${dataset}"
  echo "[All] Result: ${result_file}"
  echo "[All] Metrics: ${result_dir}/metrics.json"
done

echo "[All] All datasets done."