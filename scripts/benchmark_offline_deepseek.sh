#!/bin/bash
total_prompts=384
bs=192 # batch_size is dynamic, this setting throttles the max batch size
in_len=1024 # input length is dynamic, this tell warmup the max input len
out_len=1024 # if not set fixed_out_len, the output_len will be dynamic
total_len=$((in_len + out_len))
tp_parallel=8
 
dataset="random"

VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len * bs / 128))
VLLM_DECODE_BLOCK_BUCKET_MAX=$((total_len * bs / 128 + 128))
model="/data/models/DeepSeek-R1-static/"
tokenizer="/data/models/DeepSeek-R1-static/"
# model="/data/models/DeepSeek-R1/"
# tokenizer="/data/models/DeepSeek-R1/"
model_name="DeepSeek-R1-static"

#VLLM_PROFILER_ENABLED=true \
VLLM_DMOE_DYNAMIC_SCALE=1 \
HABANA_VISIBLE_DEVICES="ALL" \
VLLM_MOE_N_SLICE=1 \
VLLM_TEST_ENABLE_EP=1 \
VLLM_MLA_DISABLE_REQUANTIZATION=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_MAX=16 \
VLLM_PROMPT_SEQ_BUCKET_MIN=${in_len} \
VLLM_PROMPT_SEQ_BUCKET_MAX=${in_len} \
VLLM_DECODE_BS_BUCKET_MIN=${bs} \
VLLM_DECODE_BS_BUCKET_MAX=${bs} \
VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN} \
VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX} \
python3 benchmarks/benchmark_throughput.py \
    --model ${model} \
    --max-num-seqs ${bs} \
    --disable-log-requests \
    --dtype bfloat16 \
    --use-v2-block-manager \
    --backend vllm \
    --num-prompts ${total_prompts} \
    --tensor-parallel-size  ${tp_parallel} \
    --max_model_len 4096 \
    --input-len ${in_len} \
    --output-len ${out_len} \
    --trust-remote-code \
    --distributed_executor_backend mp \
    --kv_cache_dtype fp8_inc \
    --gpu-memory-util 0.85 2>&1 | tee benchmark_logs_upstream/offline-throughput-${model_name}-bs${bs}-in${in_len}-out${out_len}-tp${tp_parallel}.log