result_file="outputs/adaptive-parallel-o1/Qwen3-32B/nq/4/nq.jsonl"
dataset="nq"
DATA_ROOT="/home/zdw2200170271/llm/datasets/FlashRAG_datasets"

input_file="${DATA_ROOT}/${dataset}/test.jsonl"

python evaluate.py \
    --result_file "$result_file" \
    --dataset_name "$dataset" \
    --dataset_path "$input_file"
    # \
    # --debug
