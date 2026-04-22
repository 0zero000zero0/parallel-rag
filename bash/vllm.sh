CUDA_VISIBLE_DEVICES="0,2,4,5" vllm serve Qwen3-32B --port 8000 --tensor-parallel-size 4 --gpu-memory-utilization 0.8 --api-key "TEST" --enforce-eager

CUDA_VISIBLE_DEVICES="6,7" vllm serve Qwen3-14B --port 8001 --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --api-key "TEST"