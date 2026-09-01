# radiance-vllm-r9700

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Hardware Target: 2x AMD Radeon AI PRO R9700](https://img.shields.io/badge/Target-2x%20AMD%20Radeon%20AI%20PRO%20R9700%20(gfx1201)-crimson.svg)]()
[![ROCm: 7.14.0 / 7.2.4](https://img.shields.io/badge/ROCm-7.14.0%20%2F%207.2.4-blue.svg)]()
[![vLLM: 0.28.0](https://img.shields.io/badge/vLLM-0.28.0-orange.svg)]()
[![PyTorch: 2.12.1+rocm7.14](https://img.shields.io/badge/PyTorch-2.12.1%2Brocm7.14-red.svg)]()

An optimized, production-grade vLLM inference container specifically engineered and tuned for **Dual AMD Radeon AI PRO R9700 GPUs (`gfx1201 / RDNA4`)** running in Tensor Parallel (`TP=2`).

This repository provides full source, Dockerfiles, custom HIP/C++ kernels (`libr4d`), and logic patches for **vLLM 0.28.0**, **PyTorch 2.12.1**, **Triton 3.7.1**, and **AITER 0.1.20**.

---

## 🎯 Target Hardware & System Requirements

This project is tailored specifically for dual-card RDNA4 workstations/servers:
- **GPUs**: **2x AMD Radeon AI PRO R9700** (`gfx1201`, 32 GiB VRAM per card, 64 GiB total).
- **Interconnect**: PCIe Gen5 direct peer-to-peer (P2P enabled, ~28 GB/s bidirectional).
- **Host OS**: Linux with the standard `amdgpu` kernel driver exposing `/dev/kfd` and `/dev/dri`.
- **Runtime**: Docker or Podman (host requires no Python, no ROCm userspace, no PyTorch).

---

## ⚡ Core Architecture & Optimizations

- **P2P One-Shot All-Reduce (`ar_oneshot_2rank_exact`)**:
  - Direct PCIe P2P push/reduce mechanism bypassing RCCL for decode and prefill communications.
  - Achieves zero GPU sync overhead in CUDA/HIP graphs.
- **W4A8 fp8-WMMA GEMM Kernel**:
  - Hand-written 16×16×16 fp8 matrix core kernel running at **325 TFLOP/s** on `gfx1201`.
  - Seamlessly integrates with Quark MXFP4 models via `RadianceMxfp4W4A8LinearKernel`.
- **Fused Gated Delta Net (GDN) Chunk-Scan**:
  - Accelerated linear attention ($S^T = K \cdot Q^T$) for hybrid architectures (Qwen3.5, Ornith).
- **Aligned Mamba SSM Prefix Caching**:
  - Snapshots recurrent state alongside paged attention blocks, enabling Automatic Prefix Caching (APC) on hybrid architectures for ~3.6x TTFT reductions on shared agent system prompts.
- **FP8 KV Cache Pool**:
  - Delivers **2.12 Million tokens** of KV cache capacity across both cards with zero degradation.

---

## 📊 Serving Benchmark Results

**Model**: `Ornith-1.5-35B-A3B-FP8` (35B MoE, 256 fine-grained experts, 8 active, 131k context)  
**Configuration**: Dual AMD Radeon AI PRO R9700 (TP=2) · FP8 KV Cache · Chunked Prefill (4096 tokens)  
**Environment**: ROCm 7.14 / vLLM 0.28.0 / PyTorch 2.12.1 / Triton 3.7.1  

| Concurrency | Total Throughput (tok/s) | Request Rate | TTFT (p50) | TTFT (p95) | TPOT (p50) | TPOT (p95) | E2E Latency (p50) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Stream** | **65.1 tok/s** | 0.50 req/s | 158.8 ms | 1,046.2 ms | **12.1 ms** | 12.6 ms | 1.70 s |
| **2 Streams** | **126.1 tok/s** | 0.98 req/s | 325.4 ms | 365.6 ms | **13.3 ms** | 14.0 ms | 2.05 s |
| **4 Streams** | **187.4 tok/s** | 1.45 req/s | 366.3 ms | 1,258.7 ms | **15.3 ms** | 16.3 ms | 3.14 s |
| **8 Streams** | **352.7 tok/s** | 2.74 req/s | 307.2 ms | 392.2 ms | **20.2 ms** | 21.6 ms | 2.94 s |
| **16 Streams** | **334.9 tok/s** | 2.60 req/s | 3,072.7 ms | 4,111.2 ms | **19.8 ms** | 21.3 ms | 5.73 s |

- **Pure Generation Speed**: **12.1 ms / token** (~82.6 tokens/sec single stream).
- **Multi-Stream Scalability**: **352.7 tokens/sec** at 8 concurrent streams.
- **Available KV Capacity**: **2,125,645 tokens** (16.2x concurrency headroom at 131,072 context).
- **Tool Calling & Agentic Execution**: 100% verified with XML tool calls and reasoning tags.

---

## 🚀 Quickstart

### 1. Clone & Build Container
```bash
git clone https://github.com/drwolfen/radiance-vllm-r9700.git
cd radiance-vllm-r9700

# Build release image
docker build -t radiance-vllm:0.10.0 --build-arg RADIANCE_VERSION=0.10.0 .
```

### 2. Launch Production Server
```bash
docker run -d \
  --name vllm-radiance \
  --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host \
  -p 8000:8000 \
  -v /path/to/Ornith-1.5-35B-A3B-FP8:/models/ornith:ro \
  -v ./chat_template_ornith.jinja:/work/chat_template.jinja:ro \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e RADIANCE_USE_R4D=1 \
  -e RADIANCE_USE_R4D_AR=1 \
  -e RADIANCE_USE_R4D_AR_QUANT=1 \
  -e RADIANCE_FUSE_RMS_QUANT=1 \
  radiance-vllm:0.10.0 \
  /models/ornith \
  --served-model-name Ornith-1.5-35B-A3B-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --chat-template /work/chat_template.jinja \
  --language-model-only \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

### 3. Run Verification Tests & Benchmarks
```bash
# Run micro-kernel verification (Gate 4)
python3 tests/test_micro_kernels.py

# Run Quark MXFP4 layer selection check
python3 tests/test_mxfp4_layer.py

# Run extended benchmark suite
python3 tests/vllm_benchmark_suite.py --url http://localhost:8000/v1/chat/completions
```

---

## 📖 Log Messages, Notices & Diagnostics Reference

When starting the container or compiling graphs, you may observe specific log lines. Here is the complete reference explaining each message:

### 1. Topology & Startup Banners
- `┌─[ RADIANCE · GPU TOPOLOGY & BANDWIDTH ]──`
  - **Meaning**: Automated hardware startup sweep executing `rocm-bandwidth-test`. Confirms that both R9700 cards have direct P2P access enabled and reports host-to-device and device-to-device PCIe bandwidth (~28 GB/s).
- `[0] AMD Radeon AI PRO R9700 gfx1201 31.9 GiB ✓ gfx1201`
  - **Meaning**: Hardware arch check passed. Both GPUs successfully enumerated via direct AMDSMI initialization.

### 2. Kernel Acceleration Hooks
- `[radiance] preshuffle weight-shuffle-at-load hook installed`
  - **Meaning**: Intercepts model load to permute FP8 weights into RDNA4 matrix fragment order in host RAM, avoiding runtime layout transpositions.
- `[radiance] fast-reduce hook armed (RADIANCE_USE_R4D_AR=1, all_reduce wrap)`
  - **Meaning**: Replaces stock RCCL all-reduce with direct PCIe P2P one-shot reduction (`ar_oneshot_2rank_exact`).
- `[radiance.gdn] gdn_chunk_scan ENABLED (head_k 128, head_v 128, chunk 64)`
  - **Meaning**: Activates fused chunked linear attention kernel for hybrid GDN models (Qwen3.5/Ornith).
- `[radiance.gemm] claimed N=... K=... M=...`
  - **Meaning**: Skinny GEMM dispatcher claimed an intermediate matrix shape for execution on native RDNA4 assembly.

### 3. Compilation & Tuning Notices
- `[aiter] start build [module_quant] / [module_moe_asm]`
  - **Meaning**: AITER compiling device-specific JIT kernel modules on first model warmup. Normal one-time cost (~30s).
- `Compiling a graph for compile range (1, 4096) takes ... s`
  - **Meaning**: Torch Inductor compiling optimized Triton graph for batch prefill. Artifacts are cached in `/root/.cache/vllm/torch_compile_cache`.
- `INFO: No available shared memory broadcast block found in 60 seconds.`
  - **Meaning**: Informational polling notice from the API server while worker processes are busy compiling heavy Torch Inductor kernels during cold boot. Once compilation completes, workers resume communication immediately.
- `Setting attention block size to 2096 tokens to ensure attention page size >= mamba page size`
  - **Meaning**: Mamba alignment pass reconciling the linear-attention recurrent state with paged attention block boundaries for bit-identical prefix caching.

### 4. Deprecations & Benign Notices
- `UserWarning: tl.make_block_ptr is deprecated.`
  - **Meaning**: Upstream Triton 3.7 notification recommending `TensorDescriptor`. Handled internally; has no impact on performance or precision.
- `AllReduce fusion pass is disabled.`
  - **Meaning**: AITER's generic CUDA all-reduce fusion pass is skipped in favor of Radiance's native P2P all-reduce kernel.

---

## 🛠️ Verification Gates Architecture

Every layer of the container build is validated through automated test scripts:
- **Gate 1 (`tests/test_hip_toolchain.sh`)**: Direct HIP C++ compilation and kernel execution on `gfx1201`.
- **Gate 2 (`tests/test_wheels.py`)**: Validates PyTorch 2.12.1+rocm7.14, Triton 3.7.1, AITER 0.1.20, and dual-GPU detection.
- **Gate 3 (`tests/test_patch_ast.py`)**: AST syntax tree validation of all 2,903 patched Python modules.
- **Gate 4 (`tests/test_micro_kernels.py`)**: Executes 25 `libr4d` micro-kernels and standalone W4A8 fp8-WMMA GEMM.
- **Gate 5 (`tests/test_e2e_inference.sh`)**: Live end-to-end inference verification.

---

## 📜 Acknowledgements & Upstream Attribution

This project is built upon and directly extends the research and engineering of:
- **`ggz14`**: [`radiance-vllm-mxfp4`](https://codeberg.org/ggz14/radiance-vllm-mxfp4) (Native Quark MXFP4 and RDNA4 tuning).
- **`StillDeadcode`**: [`vllm-radiance`](https://codeberg.org/StillDeadcode/vllm-radiance) & [`libr4d`](https://codeberg.org/StillDeadcode/libr4d) (RDNA4 C++/HIP kernel library, P2P all-reduce, GDN linear attention).

All custom patches and hand-written HIP kernels originated in those repositories and have been ported and validated here for modern ROCm and vLLM releases.

## 📄 License
Apache License 2.0. See [LICENSE](LICENSE) for details.
