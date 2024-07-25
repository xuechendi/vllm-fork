# DEV prepare
```bash
export port_number="8008"
export HUGGINGFACEHUB_API_TOKEN=xxx

docker run -it --runtime=habana --name="vllm-gaudi-dev-chendi" -p $port_number:80 -v `pwd`:/root/vllm -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host -e HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN} vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.2:latest /bin/bash

# put model to attached volume
docker run -it --runtime=habana --name="vllm-gaudi-dev-chendi" -p $port_number:80 -v `pwd`:/root/vllm -v /scratch-2/models/:/root/models -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host -e HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN} -e HF_HOME="/root/models" vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.2:latest /bin/bash

# sshd
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    service ssh restart

# install habana
pip install --upgrade-strategy eager optimum[habana]
cd /root/vllm
pip install -v ./

```

# DEV serving
```bash
export PT_HPU_LAZY_ACC_PAR_MODE=0
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export model_name="meta-llama/Meta-Llama-3-70b"
export parallel_number=2
export block_size=128
export max_num_seqs=256
export max_seq_len_to_capture=2048
HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN} OMPI_MCA_btl_vader_single_copy_mechanism=none HABANA_VISIBLE_DEVICES=all VLLM_CPU_KVCACHE_SPACE=40 python3 -m vllm.entrypoints.openai.api_server --enforce-eager --model $model_name --tensor-parallel-size $parallel_number --host 0.0.0.0 --port 80 --block-size $block_size --max-num-seqs  $max_num_seqs --max-seq_len-to-capture $max_seq_len_to_capture
```

# DEV Benchmark
```bash

```

===


ENV no_proxy=localhost,127.0.0.1
```

# Build docker
```bash
cd ${vllm}
docker build -f Dockerfile.hpu -t vllm:hpu --shm-size=128g .
```

# launch docker
```bash
export model_name="meta-llama/Meta-Llama-3-70b"
export port_number="8008"
export num_hpu=8
bash ./launch_vllm_service.sh ${port_number} ${model_name} hpu ${num_hpu}
```

# Test
```bash
python test.py
```

```bash
curl http://127.0.0.1:80/v1/completions   -H "Content-Type: application/json"   -d '{
  "model": "meta-llama/Meta-Llama-3-70b",
  "prompt": "What is Deep Learning?",
  "max_tokens": 32,
  "temperature": 0
  }'
```

# Benchmark
```bash

```
