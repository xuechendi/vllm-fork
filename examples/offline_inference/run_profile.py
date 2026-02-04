#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Profile multiple models using vLLM's profiling capabilities.

This script runs profiling on a list of models and saves the profiling
results to separate directories for each model.
"""

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CUDAGraphMode


# List of models to profile
MODELS_default = [
    # "meta-llama/Llama-3.1-8B",
    # #"meta-llama/Llama-4-Scout-17B-16E-Instruct",
    # "deepseek-ai/DeepSeek-V2-Lite",
    # "Pinaster/DeepSeek-V3.2-5layer",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "openai/gpt-oss-20b",
    "google/gemma-3-4b-it",
]

MODELS_fp8 = [
    "Qwen/Qwen3-0.6B-FP8",
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
]

MODELS_fp8_dynamic = [
    "RedHatAI/Qwen3-8B-FP8-dynamic",
]

MODELS=MODELS_fp8
#MODELS=MODELS_fp8_dynamic

# Sample prompts for profiling
PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Default sampling parameters
SAMPLING_PARAMS = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=5)

# Base directory for profiling results
PROFILE_BASE_DIR = Path("./vllm_profiles")


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in directory names."""
    return model_name.replace("/", "_").replace(" ", "_")


def run_profile(
    model: str,
    profile_dir: Path,
    sampling_params: SamplingParams,
    prompts: list[str],
    enforce_eager: bool = False,
    use_cuda_graph: bool = True,
) -> Dict[str, Any]:
    """
    Run profiling for a single model.
    
    Args:
        model: Model name/identifier
        profile_dir: Directory to save profiling results
        sampling_params: Sampling parameters for generation
        prompts: List of prompts to use for profiling
        enforce_eager: Whether to enforce eager execution
        use_cuda_graph: Whether to use CUDA graphs (only applies when enforce_eager=False)
        
    Returns:
        Dictionary with profiling results and metadata
    """
    result = {
        "model": model,
        "enforce_eager": enforce_eager,
        "use_cuda_graph": use_cuda_graph,
        "status": "unknown",
        "profile_dir": str(profile_dir),
        "start_time": None,
        "end_time": None,
        "duration": None,
        "error": None,
        "outputs": [],
    }
    
    print(f"\n{'='*80}")
    print(f"Profiling model: {model}")
    print(f"Configuration: enforce_eager={enforce_eager}, use_cuda_graph={use_cuda_graph}")
    print(f"Profile directory: {profile_dir}")
    print(f"{'='*80}")
    
    start_time = datetime.now()
    result["start_time"] = start_time.isoformat()
    
    try:
        # Create profile directory
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare LLM arguments
        llm_kwargs = {
            "model": model,
            "tensor_parallel_size": 1,  # Adjust as needed
            "profiler_config": {
                "profiler": "torch",
                "torch_profiler_dir": str(profile_dir),
                "torch_profiler_record_shapes": True,
                "torch_profiler_with_stack": True,
            },
            "enforce_eager": enforce_eager,
            "max_model_len": 4096,
            "kv_cache_dtype": "fp8",


        }
        
        # Configure CUDA graph if enforce_eager is False
        if not enforce_eager:
            if use_cuda_graph:
                # Use default CUDA graph (FULL mode)
                llm_kwargs["compilation_config"] = CompilationConfig(
                    cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
                )
            else:
                # Disable CUDA graph explicitly
                llm_kwargs["compilation_config"] = CompilationConfig(
                    cudagraph_mode=CUDAGraphMode.NONE,
                )
        
        # Create LLM with profiling enabled
        print(f"Initializing LLM with profiling...")
        llm = LLM(**llm_kwargs)
        
        print(f"Starting profiler...")
        llm.start_profile()
        
        # Generate texts from the prompts
        print(f"Generating with {len(prompts)} prompts...")
        outputs = llm.generate(prompts, sampling_params)
        
        print(f"Stopping profiler...")
        llm.stop_profile()
        
        # Store outputs
        result["outputs"] = [
            {
                "prompt": output.prompt,
                "generated_text": output.outputs[0].text,
                "token_ids": len(output.outputs[0].token_ids),
            }
            for output in outputs
        ]
        
        # Print the outputs
        print("-" * 50)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}")
            print(f"Generated text: {generated_text!r}")
            print(f"Tokens generated: {len(output.outputs[0].token_ids)}")
            print("-" * 50)
        
        result["status"] = "success"
        print(f"✓ Profiling completed successfully for {model}")
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"✗ Profiling failed for {model}")
        print(f"Error: {str(e)}")
        print(f"Traceback:")
        traceback.print_exc()
    
    finally:
        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["duration"] = (end_time - start_time).total_seconds()
        
        # Add a buffer to wait for profiler in the background process
        # (in case MP is on) to finish writing profiling output.
        print(f"Waiting for profiler to finish writing output...")
        time.sleep(10)
    
    return result


def main():
    """Main function to run profiling for all models."""
    print("="*80)
    print("vLLM Model Profiling Script")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Profiling {len(MODELS)} models with 3 configurations each")
    print(f"Profile results will be saved to: {PROFILE_BASE_DIR.absolute()}")
    print("="*80)
    
    # Create base profile directory
    PROFILE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define test configurations
    test_configs = [
        {"enforce_eager": True, "use_cuda_graph": False, "name": "eager"},
        {"enforce_eager": False, "use_cuda_graph": True, "name": "cuda_graph"},
        {"enforce_eager": False, "use_cuda_graph": False, "name": "no_cuda_graph"},
    ]
    
    results = []
    total_tests = len(MODELS) * len(test_configs)
    successful = 0
    failed = 0
    
    for model in MODELS:
        model_safe = sanitize_model_name(model)
        
        for config in test_configs:
            # Create configuration-specific profile directory
            # Format: {model_name}_{config_name}
            config_name = config["name"]
            folder_name = f"{model_safe}_{config_name}"
            profile_dir = PROFILE_BASE_DIR / folder_name
            
            # Run profiling
            result = run_profile(
                model=model,
                profile_dir=profile_dir,
                sampling_params=SAMPLING_PARAMS,
                prompts=PROMPTS,
                enforce_eager=config["enforce_eager"],
                use_cuda_graph=config["use_cuda_graph"],
            )
            
            results.append(result)
            
            if result["status"] == "success":
                successful += 1
            else:
                failed += 1
    
    # Print summary
    print("\n" + "="*80)
    print("PROFILING SUMMARY")
    print("="*80)
    print(f"Total tests: {total_tests} ({len(MODELS)} models × {len(test_configs)} configurations)")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("\nDetailed results:")
    print("-" * 100)
    
    # Group results by model
    for model in MODELS:
        model_safe = sanitize_model_name(model)
        model_results = [r for r in results if r["model"] == model]
        
        print(f"\nModel: {model}")
        for result in model_results:
            status_icon = "✓" if result["status"] == "success" else "✗"
            duration_str = f"{result['duration']:.2f}s" if result["duration"] else "N/A"
            config_str = f"eager={result['enforce_eager']}, cuda_graph={result['use_cuda_graph']}"
            print(f"  {status_icon} [{config_str:<30}] Duration: {duration_str:<10} Profile: {result['profile_dir']}")
            if result["error"]:
                print(f"      Error: {result['error']}")
    
    print("\n" + "="*80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All profile results saved to: {PROFILE_BASE_DIR.absolute()}")
    print("="*80)
    
    # Exit with error code if any profiles failed
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
