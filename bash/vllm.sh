CUDA_VISIBLE_DEVICES="0,1,2,4" vllm serve Qwen3-32B --port 8000 --tensor-parallel-size 4 --gpu-memory-utilization 0.8 --api-key "TEST"

CUDA_VISIBLE_DEVICES="6,7" vllm serve Qwen3-14B --port 8001 --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --api-key "TEST"