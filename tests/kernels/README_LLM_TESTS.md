# vLLM LLM Test Suite - Quick Reference

**Date:** 2026-05-07  
**Purpose:** Verify correctness of all operations in vLLM LLM inference  
**Coverage:** 28 operations (16 Triton kernels + 12 extern operations)

---

## 📚 Documentation Structure

| File | Purpose | Details |
|------|---------|---------|
| **LLM_COMPLETE_TEST_GUIDE.md** | 📖 Main documentation | Complete guide covering all operations, running instructions, troubleshooting |
| **LLM_TEST_SUITE_SUMMARY.md** | 📊 Quick reference | High-level stats, test coverage, how-to |
| **README_LLM_TESTS.md** | ⚡ This file | Ultra-quick reference |

**👉 Start with:** `LLM_COMPLETE_TEST_GUIDE.md`

---

## 🚀 Quick Start

### Run All Tests (242 tests, ~5 min)

```bash
cd /workspace/vllm
.venv/bin/python -m pytest \
    tests/kernels/test_triton_kernels.py \
    tests/kernels/test_llm_extern_ops.py \
    -v
```

### Run Specific Categories

```bash
# Triton kernels only (150 tests)
.venv/bin/python -m pytest tests/kernels/test_triton_kernels.py -v

# Extern operations only (92 tests)
.venv/bin/python -m pytest tests/kernels/test_llm_extern_ops.py -v

# RMS normalization only (24 tests)
.venv/bin/python -m pytest \
    tests/kernels/test_triton_kernels.py::TestRMSNormKernels -v

# GEMM operations only (48 tests)
.venv/bin/python -m pytest \
    tests/kernels/test_llm_extern_ops.py::TestGEMMOperations -v

# Flash attention only (8 tests)
.venv/bin/python -m pytest \
    tests/kernels/test_llm_extern_ops.py::TestFlashAttention -v

# MoE operations only (20 tests)
.venv/bin/python -m pytest \
    tests/kernels/test_llm_extern_ops.py::TestMoEOperations -v
```

---

## 📋 Test Coverage

### Test Files

| File | Operations | Tests | Coverage |
|------|-----------|-------|----------|
| `test_triton_kernels.py` | 16 Triton kernels | 150 | RMS norm, SiLU, FP8, reductions |
| `test_llm_extern_ops.py` | 12 extern ops | 92 | GEMM, attention, MoE, sampling |
| **Total** | **28** | **242** | **100%** |

### Operation Categories

| Category | Operations | XPU Time | Test Classes |
|----------|-----------|----------|--------------|
| RMS Normalization | 6 kernels | 25-50ms | `TestRMSNormKernels` |
| FP8 Quantization | 4 kernels | 29-50ms | `TestFP8Kernels` |
| Activation Functions | 2 kernels | 19-27ms | `TestActivationKernels` |
| Reductions | 2 kernels | 3-5ms | `TestReductionKernels` |
| Pointwise Fusions | 4 kernels | 0.2-5ms | `TestComboKernels` |
| **GEMM Operations** | **3 variants** | **671ms-1.2s** | **`TestGEMMOperations`** |
| **Flash Attention** | **1 op** | **61-208ms** | **`TestFlashAttention`** |
| **MoE Operations** | **4 ops** | **55ms** | **`TestMoEOperations`** |
| Sampling | 1 op | 6-7ms | `TestSamplingOperations` |
| Cache Operations | 1 op | 2-3ms | `TestCacheOperations` |

---

## 🔧 Test Modes

All tests run in **both modes**:

| Mode | Description | Purpose |
|------|-------------|---------|
| **eager** | PyTorch reference | Validate reference implementation |
| **compiled** | torch.compile + kernels | Validate kernel correctness |

Run specific mode:

```bash
# Eager only
.venv/bin/python -m pytest tests/kernels/ -v -k "eager"

# Compiled only
.venv/bin/python -m pytest tests/kernels/ -v -k "compiled"
```

---

## 🎯 Models Tested

| Model | Config | Operations Tested |
|-------|--------|-------------------|
| **Llama-3.3-70B-Instruct** | TP=4, FP8 W8A16 | FP8 kernels + all shared ops |
| **Qwen3-32B** | TP=4, BF16 | All shared ops |
| **Qwen3-30B-A3B** | TP=4, EP, MoE | MoE ops + all shared ops |

---

## 🐛 Troubleshooting

**Tests skipped?**

```bash
# Check GPU availability
python -c "import torch; print(f'XPU: {torch.xpu.is_available()}, CUDA: {torch.cuda.is_available()}')"
```

**Kernel not compiling?**

```bash
# Enable verbose logging
export TORCH_LOGS="+dynamo,+inductor"
.venv/bin/python -m pytest tests/kernels/test_triton_kernels.py -v -k "compiled" -s
```

**See full guide:** `LLM_COMPLETE_TEST_GUIDE.md` → Troubleshooting section

---

## 📖 Additional Documentation

- **Complete Guide:** `LLM_COMPLETE_TEST_GUIDE.md` (1,000+ lines)
- **Summary:** `LLM_TEST_SUITE_SUMMARY.md` (700 lines)
- **Triton Details:** `TRITON_KERNEL_TESTS.md` (500 lines)
- **Profiling Analysis:** `../vllm_profile/triton_kernel_analysis.md`

---

## ✅ Status

- **Test Coverage:** 100% of profiled operations
- **Total Tests:** 242 (121 eager + 121 compiled)
- **Test Files:** 2 (test_triton_kernels.py, test_llm_extern_ops.py)
- **Documentation:** Complete
- **Status:** ✅ Ready to run

**Created:** 2026-05-07
