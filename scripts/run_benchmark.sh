#!/bin/bash
gpu_utils=0.9
bs=64
num_prompts=1024
tp_parrallel=1
log_name="online-${gpu_utils}util-TPparallel${tp_parrallel}"

model="/model/meta-llama/Llama-2-7b-chat-hf"
model_name="Llama-2-7b"

VLLM_USE_V1=1 \
python -m vllm.entrypoints.openai.api_server \
    --port 18080 \
    --model ${model} \
    --tensor-parallel-size ${tp_parrallel} \
    --max-num-seqs ${bs} \
    --disable-log-requests \
    --dtype float16 \
    --block-size 16 \
    --enforce-eager --no-enable-prefix-caching \
    --gpu_memory_utilization ${gpu_utils} \
    --trust_remote_code 2>&1 | tee benchmark_logs/${log_name}_serving.log &
pid=$(($!-1))

until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Started server process" benchmark_logs/${log_name}_serving.log; then
        break
    fi
    sleep 5s
done
sleep 10s
echo ${pid}


start_time=$(date +%s)
echo "Start to benchmark"
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model ${model} \
    --max-concurrency ${bs} \
    --dataset-name sharegpt \
    --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts ${num_prompts} \
    --port 18080  2>&1 | tee benchmark_logs/${log_name}_run.log
end_time=$(date +%s)
echo "Time elapsed: $((end_time - start_time))s"

sleep 10

kill ${pid}
