#!/bin/bash
#HABANA_PROFILE_WRITE_HLTV=1 HABANA_PROFILE=1 
#VLLM_PT_PROFILE=prompt_2_1024_t \
#HABANA_PROF_CONFIG=profile_api_trace_analyzer.json \
VLLM_TP_OVERLAP_PER_BATCH=true \
python scripts/run_example_TP.py --tp_size 1 --model /data/models/models/Llama-2-7b-chat-hf/