bs=64
in_len=1024
out_len=1024
total_len=$((in_len + out_len))
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
for bs in 128 96 64;do
python3 -u vllm/benchmarks/benchmark_throughput.py \
    --model ${model} \
    --device cuda \
    --backend vllm \
    --num-prompts ${bs} \
    --input_len ${in_len} \
    --output_len ${out_len} \
    --max_model_len ${total_len} 2>&1 | tee benchmark_logs/vllm_offline_heapq_bf16_bs${bs}_i${in_len}_o${out_len}_fwd_lat.log
done
