#!/bin/bash
# Check if the number of arguments is zero
if (( $# == 0 )); then
    # Print an error message to standard error (>&2)
    echo "Error: Please provide model_path TP_size" >&2
    # Exit with a non-zero status code to indicate an error
    exit 1
fi
model=$1
gpu_utils=0.9
num_prompts=1024
tp_parrallel=2

branch_name=$(git branch --show-current)
model_name=$(basename "$model")
log_name="online-${branch_name}-${gpu_utils}util-TPparallel${tp_parrallel}-${model_name}"

mkdir -p benchmark_logs

# Define the directory and file path
DIRECTORY="benchmarks"
FILE_PATH="$DIRECTORY/ShareGPT_V3_unfiltered_cleaned_split.json"

# Define the URL for the file to be downloaded
URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

# Check if the file already exists at the specified path
if [ -f "$FILE_PATH" ]; then
    # If the file exists, print a message and exit
    echo "File '$FILE_PATH' already exists. No download needed."
else
    # If the file does not exist, proceed with the download
    echo "File '$FILE_PATH' not found."

    # Download the file using wget
    echo "Downloading file from $URL..."
    wget -O "$FILE_PATH" "$URL"

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Download successful. File saved to '$FILE_PATH'."
    else
        echo "Error during download. Please check the URL or your network connection."
    fi
fi

ZE_AFFINITY_MASK=6,7 \
CCL_ATL_TRANSPORT=ofi \
CCL_ZE_IPC_EXCHANGE=drmfd \
VLLM_USE_V1=1 \
TORCH_LLM_ALLREDUCE=1 \
CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
python -m vllm.entrypoints.openai.api_server \
    --port 18080 \
    --model ${model} \
    --tensor-parallel-size ${tp_parrallel} \
    --disable-log-requests \
    --max-num-batched-tokens=8192 \
    --max-model-len=8192 \
    --dtype float16 \
    --block-size 64 \
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
    --max_concurrency 256 \
    --dataset-name sharegpt \
    --trust_remote_code \
    --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts ${num_prompts} \
    --port 18080  2>&1 | tee benchmark_logs/${log_name}_run.log
end_time=$(date +%s)
echo "Time elapsed: $((end_time - start_time))s"

sleep 10

kill ${pid}
