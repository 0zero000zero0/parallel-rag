#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/home/zdw2200170271/llm/datasets/FlashRAG_datasets"
RETRIEVER_BASE_URL="http://localhost:9100"
OPENAI_BASE_URL="http://localhost:8001/"
OPENAI_API_KEY="TEST"
MODEL="Qwen3-14B"
NUM_SAMPLES=512
DATASETS=(
  # bamboogle
  # 2wikimultihopqa
  # hotpotqa
  # musique
  # nq
  # popqa
  triviaqa
  ambigqa
)

for dataset in "${DATASETS[@]}"; do
  input_file="${DATA_ROOT}/${dataset}/test.jsonl"

  if [[ ! -f "$input_file" ]]; then
    echo "[All] Skip ${dataset}: input file not found -> $input_file"
    continue
  fi

  dataset_output_root="outputs/parallel_rag/${MODEL}/${dataset}"
  mkdir -p "$dataset_output_root"

  max_index="$(find "$dataset_output_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1 || true)"
  if [[ -z "$max_index" ]]; then
    next_index=0
  else
    next_index=$((max_index + 1))
  fi

  result_dir="${dataset_output_root}/${next_index}"
  result_file="${result_dir}/output.jsonl"

  echo "[All]=================================================="
  echo "[All] Dataset: ${dataset}"
  echo "[All] Expected output dir: ${result_dir}"

  python run_parallel_rag.py \
    --input_file "$input_file" \
    --batch_size 512 \
    --retriever_base_url "$RETRIEVER_BASE_URL" \
    --retriever_top_k 5 \
    --openai_base_url "$OPENAI_BASE_URL" \
    --openai_api_key "$OPENAI_API_KEY" \
    --model "$MODEL" \
    --parallel_query_count 5 \
    --docs_per_query 5 \
    --query_max_tokens 1024 \
    --query_temperature 0.8 \
    --query_top_p 0.9 \
    --refine_max_tokens 1024 \
    --refine_temperature 0.8 \
    --refine_top_p 0.9 \
    --synthesize_max_tokens 1024 \
    --synthesize_temperature 0.8 \
    --synthesize_top_p 0.9 \
    --max_iterations 5 \
    --debug False
    # --num_samples "$NUM_SAMPLES"

  if [[ ! -f "$result_file" ]]; then
    echo "[All] ERROR: result file not found after run_parallel_rag.py: $result_file" >&2
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
