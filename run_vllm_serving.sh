model="meta-llama/Meta-Llama-3.1-8B-Instruct"
bs=64
in_len=1024
out_len=1024
#model="meta-llama/Meta-Llama-3.1-8B-Instruct"

for bs in 128 96 64; do
        echo "Start to warmup"
        for i in 1;do
                python3 vllm/benchmarks/benchmark_serving.py --backend vllm --model $model --dataset-name sonnet --dataset-path vllm/benchmarks/sonnet.txt --request-rate 512 --num-prompts ${bs} --port 18080 --sonnet-input-len ${in_len} --sonnet-output-len ${out_len} --sonnet-prefix-len 100
        done

        echo "Start to benchmark"
python3 vllm/benchmarks/benchmark_serving.py --backend vllm --model ${model} --dataset-name sonnet --dataset-path vllm/benchmarks/sonnet.txt --request-rate 512 --num-prompts ${bs} --port 18080 --sonnet-input-len ${in_len} --sonnet-output-len ${out_len} --sonnet-prefix-len 100 | tee benchmark_logs/benchmark_serving_Llama-3.1-8B-Instruct_sonnet_bs${bs}_i${in_len}_o${out_len}.txt

        sleep 10
done
kill 1473
