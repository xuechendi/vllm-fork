## 1. Gaudi Cluster
Login JumpNode:    jumpid@10.239.44.190 passwd: intel@123
Then:
```
ssh -i ~/.ssh/id_ed25519_idc -J guest@146.152.224.71 sdp@100.83.111.250
```
Gaudi Instance
 
aise-cluster-03-0 sdp@100.83.111.250

 Driver version 1.16
aise-cluster-03-1 sdp@100.83.111.228

 Don't use, for Carson's team CICD
~~aise-cluster-03-2 sdp@100.83.111.234~~

 Driver version 1.17
aise-cluster-03-3 sdp@100.83.111.238


## 2. Commit id cf6952d, Driver 1.16

### 2.1 CMD:

```

docker run -it --runtime=habana --rm --name="benchmark" -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.2:latest

git clone https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
git checkout cf6952d
pip install -e .  
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
python /vllm-fork/benchmarks/benchmark_throughput.py --model lmsys/vicuna-7b-v1.5 --enforce-eager --dtype bfloat16  --device hpu --dataset /vllm-fork/ShareGPT_V3_unfiltered_cleaned_split.json
```

### 2.2 perfromance config:
 https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#troubleshooting-tweaking-hpu-graphs:
- *block_size*(default 16, we set 128 now) can greatly affect performance(3x).
- *max-seq-len-to-capture* did not show a significant impact on performance.
- *--enforce-eager* Removing this parameter when block_size is default(16) can improve performance( +13%) but adding it after 
> block_size is set 128 can also improve performance(+12.7% ).With HPU Graphs disabled, you are trading latency and throughput at lower batches for potentially higher throughput on higher batches. You can do that by adding --enforce-eager flag to server (for online inference), or by passing enforce_eager=True argument to LLM constructor (for offline inference).

### 2.3 Best Result:

```
#benchmark_throughput.py 
max-seq-len-to-capture=2048 
block_size=128
```
CMD:  
```
python /vllm-fork/benchmarks/benchmark_throughput.py --model  meta-llama/Llama-2-7b-chat-hf --dtype bfloat16  --enforce-eager --device hpu --dataset /vllm-fork/ShareGPT_V3_unfiltered_cleaned_split.json
```
Throughput: 1195.31 tokens/s


## 3. Branch habana_next, Driver 1.17

Please use aise-cluster-03-3 sdp@100.83.111.238



### 3.1 Config
edit benchmark_throughput.py
```
--- a/benchmarks/benchmark_throughput.py
+++ b/benchmarks/benchmark_throughput.py
@@ -100,6 +100,11 @@ def run_vllm(
         download_dir=download_dir,
         enable_chunked_prefill=enable_chunked_prefill,
         max_num_batched_tokens=max_num_batched_tokens,
+        block_size=128,
+        max_num_seqs=128,
+        num_lookahead_slots=1,
+        use_v2_block_manager=True,
+        enable_delayed_sampling=True,
     )

     # Add the requests to the engine.
@@ -109,10 +114,11 @@ def run_vllm(
         prompts.append(prompt)
         sampling_params.append(
             SamplingParams(
-                n=n,
-                temperature=0.0 if use_beam_search else 1.0,
-                top_p=1.0,
-                use_beam_search=use_beam_search,
+                # n=n,
+                # temperature=0.0 if use_beam_search else 1.0,
+                # top_p=1.0,
+                # use_beam_search=use_beam_search,
+                temperature=0.0,
                 ignore_eos=True,
                 max_tokens=output_len,
             ))
```

### 3.2 CMD:
```

docker run -it --runtime=habana --rm --name="benchmark" -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host hlc/jizhang

git clone https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
git checkout habana_next
pip install -e .  
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
VLLM_GRAPH_RESERVED_MEM=0.2 VLLM_GRAPH_PROMPT_RATIO=0.8 VLLM_DECODE_BS_BUCKET_MIN=1 VLLM_DECODE_BLOCK_BUCKET_STEP=64 VLLM_DECODE_BLOCK_BUCKET_MIN=64  python benchmark_throughput.py --model meta-llama/Llama-2-7b-chat-hf --device hpu --seed 2024 --backend vllm --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --dtype bfloat16
```

### 3.3 perfromance config(From Chen, Xinyu): 
```
native
2249.29tokens/s

flat pagedattention
2700.48 tokens/s

num_lookahead_slots=1, use_v2_block_manager, (prv)enable_delayed_sampling
2958.24 tokens/s

Tune HPU Graph Memory.
3035.64 tokens/s

VLLM_GRAPH_RESERVED_MEM=0.2 used to determine how much free memory after loading weights should be used for HPU-graphs. default 0.4

VLLM_GRAPH_PROMPT_RATIO=0.8 split free memory(98% free memory after loading weights) for prompt graph and decoding graph.default 0.5

Tune Decode Bucket.
4848.84 tokens/s

(bug, commit fix by poland team)VLLM_DECODE_BS_BUCKET_MIN=1. default 32

(User tuned)VLLM_DECODE_BLOCK_BUCKET_STEP=64. default 128

(User tuned)VLLM_DECODE_BLOCK_BUCKET_MIN=64. default 128

VLLM_PROMPT/DECODE_BS/BLOCK_BUCKET_MIN/STEP/MAX. 2x2x3 space
```

### 3.4 Best Result:

```
#benchmark_throughput.py 
max_num_seqs=128,
block_size=128,
num_lookahead_slots=1,
use_v2_block_manager=True,
enable_delayed_sampling=True,
```
CMD:  
```
VLLM_GRAPH_RESERVED_MEM=0.2 VLLM_GRAPH_PROMPT_RATIO=0.8 VLLM_DECODE_BS_BUCKET_MIN=1 VLLM_DECODE_BLOCK_BUCKET_STEP=64 VLLM_DECODE_BLOCK_BUCKET_MIN=64  python benchmark_throughput.py --model meta-llama/Llama-2-7b-chat-hf --device hpu --seed 2024 --backend vllm --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --dtype bfloat16
```
Throughput: 4802.51.31 tokens/s






### 4. Driver Upgrade Gaudi(from Keping)

Normal upgrade: https://wiki.ith.intel.com/pages/viewpage.action?spaceKey=AppliedML&title=Set+up+Gaudi2+SW+Env

#### Step 1: Install habanalabs-* packages manually.

First download habanalabs-* deb files and sync them to gaudi machine. Use `sudo apt install deb_file_path` to install these package.


```
https://artifactory-kfs.habana-labs.com/artifactory/ubuntu-deb-dev-local/pool/jammy/testing/h/habanalabs-firmware/amd64/habanalabs-firmware-1.17.0-335.amd64.deb

https://artifactory-kfs.habana-labs.com/artifactory/ubuntu-deb-dev-local/pool/jammy/testing/h/habanalabs-dkms/all/habanalabs-dkms-1.17.0-335.all.deb

https://artifactory-kfs.habana-labs.com/artifactory/ubuntu-deb-dev-local/pool/jammy/testing/h/habanalabs-rdma-core/all/habanalabs-rdma-core-1.17.0-335_all.deb

https://artifactory-kfs.habana-labs.com/artifactory/ubuntu-deb-dev-local/pool/jammy/testing/h/habanalabs-thunk/all/habanalabs-thunk-1.17.0-335_all.deb

https://artifactory-kfs.habana-labs.com/artifactory/ubuntu-deb-dev-local/pool/jammy/testing/h/habanalabs-graph/amd64/habanalabs-graph-1.17.0-335_amd64.deb

https://artifactory-kfs.habana-labs.com/artifactory/ubuntu-deb-dev-local/pool/jammy/testing/h/habanalabs-firmware-tools/amd64/habanalabs-firmware-tools_1.17.0-335_amd64.deb

https://artifactory-kfs.habana-labs.com/artifactory/ubuntu-deb-dev-local/pool/jammy/testing/h/habanalabs-qual/amd64/habanalabs-qual-1.17.0.335_amd64.deb
```


Run `hl-smi` to view the version after this installation:



And run `apt list --installed | grep habana` to verify the system SW:

$ apt list --installed | grep habana

habanalabs-dkms/now 1.17.0-335 all [installed,local]

habanalabs-firmware-tools/now 1.17.0-335 amd64 [installed,local]

habanalabs-firmware/now 1.17.0-335 amd64 [installed,local]

habanalabs-graph/now 1.17.0-335 amd64 [installed,local]

habanalabs-qual/now 1.17.0-335 amd64 [installed,local]

habanalabs-rdma-core/now 1.17.0-335 all [installed,local]

habanalabs-thunk/now 1.17.0-335 all [installed,local]

 

#### Step 2: Upgrade driver.

Download habanalabs-installer.sh, comment out the codes for installing habanalabs-* (Line 1161-1171), and then execute `./habanalabs-installer.sh install --type base`.



```
# # install habanalabs packages

# if [ -z "${args[--skip-install-firmware]}" ]; then

#     header "Install the firmware"

#     installPackages "${HABANALABS_PRE_REQUIRED}"

# fi

# header "Install the driver"

# installPackages "${HABANALABS_DRIVER}"

# header "Install the basic packages"

# installPackages "${HABANALABS_REQUIRED}"

# header "Install the optional packages"

# installPackages "${HABANALABS_OPTIONAL}"
```


Run `hl-smi` to view the driver version after this operation:



 

#### Step 3: Upgrade container runtime.

Also manually download the corresponding container deb file, sync it to gaudi machine, and then install it.

Run ` apt list --installed | grep habana` to verify it:

$ apt list --installed | grep habana

habanalabs-container-runtime/now 1.17.0-335 amd64 [installed,local]

habanalabs-dkms/now 1.17.0-335 all [installed,local]

habanalabs-firmware-tools/now 1.17.0-335 amd64 [installed,local]

habanalabs-firmware/now 1.17.0-335 amd64 [installed,local]

habanalabs-graph/now 1.17.0-335 amd64 [installed,local]

habanalabs-qual/now 1.17.0-335 amd64 [installed,local]

habanalabs-rdma-core/now 1.17.0-335 all [installed,local]

habanalabs-thunk/now 1.17.0-335 all [installed,local]

 

Step 4: Sync docker image and validate the environment.

There is a 1.17 version docker image on jump node(jumpid@10.239.44.190:/home/jumpid/hlc.tar). Please copy it to gaudi node and run `docker load -i hlc.tar`. Then you will get an image,

hlc/jizhang           latest       de6764b32f3e   4 hours ago         6.59GB

You can build new image or do some tests based on it.

