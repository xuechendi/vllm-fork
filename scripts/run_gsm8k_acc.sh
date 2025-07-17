MODEL_PATH=$1

MODEL_NAME=$(basename $MODEL_PATH)
TP_SIZE=1

VLLM_USE_V1=1 \
lm_eval --model vllm \
  --model_args "pretrained=${MODEL_PATH},tensor_parallel_size=${TP_SIZE},trust_remote_code=true,max_model_len=4096" \
  --tasks gsm8k --num_fewshot "5" \
  --batch_size "auto" --log_samples --output_path gsm8k_acc_${MODEL_NAME}.json
