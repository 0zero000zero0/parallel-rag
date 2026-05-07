CUDA_VISIBLE_DEVICES="2,3,5,6" vllm serve Qwen3-32B --port 9101 -tp 4 -dp 1 --gpu-memory-utilization 0.8 --api-key "TEST"

CUDA_VISIBLE_DEVICES="4,5" vllm serve Qwen3-4B --port 8001 -tp 1 -dp 2 --gpu-memory-utilization 0.85 --api-key "TEST"