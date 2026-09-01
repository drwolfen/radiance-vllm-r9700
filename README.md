# radiance-vllm-r9700

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Hardware Target: 2x AMD Radeon AI PRO R9700](https://img.shields.io/badge/Target-2x%20AMD%20Radeon%20AI%20PRO%20R9700%20(gfx1201)-crimson.svg)]()
[![ROCm: 7.14.0 / 7.2.4](https://img.shields.io/badge/ROCm-7.14.0%20%2F%207.2.4-blue.svg)]()
[![vLLM: 0.28.0](https://img.shields.io/badge/vLLM-0.28.0-orange.svg)]()
[![PyTorch: 2.12.1+rocm7.14](https://img.shields.io/badge/PyTorch-2.12.1%2Brocm7.14-red.svg)]()

An optimized, production-grade vLLM inference server specifically engineered for **Dual AMD Radeon AI PRO R9700 GPUs (`gfx1201 / RDNA4`)** running in Tensor Parallel (`TP=2`).

It bundles a hardened, tested **ROCm 7.14.0 / 7.2.4 + PyTorch 2.12.1 + Triton 3.7.1 + AITER 0.1.20 + vLLM 0.28.0** stack with hand-written RDNA4 matrix/attention kernels (`libr4d`), custom W4A8 fp8-WMMA GEMM, P2P PCIe one-shot all-reduce, aligned Mamba prefix caching, and dynamic MTP drafting.

---

## 🎯 Target Hardware, Host OS & Kernel Parameters

This stack is engineered and verified specifically for dual-card RDNA4 workstations:
- **GPUs**: **2x AMD Radeon AI PRO R9700** (`gfx1201`, 32 GiB VRAM per card, 64 GiB total pool).
- **Interconnect**: Direct PCIe Gen5 peer-to-peer (P2P enabled, ~28 GB/s bidirectional).
- **Host OS**: **Ubuntu 24.04 LTS (`noble`)** with standard `amdgpu` kernel driver exposing `/dev/kfd` and `/dev/dri`.
- **Runtime**: Docker or Podman (host requires no Python, no ROCm userspace, no PyTorch).

### Recommended Host Kernel Boot Parameters & UEFI Configuration
For smooth dual-GPU Tensor Parallel performance, PCIe P2P bandwidth, and zero driver timeouts, configure `/etc/default/grub` with:

```text
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=on iommu=pt numa_balancing=disable pcie_aspm=off pci=realloc=off amdgpu.ppfeaturemask=0xffffffff"
```

**UEFI / BIOS Settings**:
- **Above 4G Decoding**: `Enabled`
- **Resizable BAR (ReBAR / Smart Access Memory)**: `Enabled`
- **PCIe Link Speed**: `Gen5 / Auto`
- **IOMMU**: `Enabled` (Passthrough mode via `iommu=pt`)

---

## 🔍 Why ROCm 7.14.0 vs. ROCm 10 (Toolchain & Driver Compatibility)

ROCm 10 (TheRock unified toolchain) was investigated as an initial candidate, but failed across three critical technical barriers:

1. **Host Kernel Driver (`/dev/kfd`) ABI Mismatch**:
   - The host machine runs the **ROCm 7.2.4 `amdgpu` kernel driver**.
   - ROCm 10 userspace relies on new KFD memory topology and ioctl interfaces (KFD v2.x ABI). Running ROCm 10 userspace inside a container on a 7.2.4 host kernel driver results in device probe failures (`HSA_STATUS_ERROR_DEVICE_MISMATCH`).
2. **PyTorch 2.13+ / Triton 3.8 Instability & GPU Hangs on `gfx1201`**:
   - ROCm 10 requires PyTorch 2.13+ / 2.14-dev and Triton 3.8.
   - On `gfx1201` (R9700), Triton 3.8 drops the 16×16×16 fp8 matrix instruction lowering, falling back to 16-bit emulation.
   - Under sustained Tensor Parallel (`TP=2`) load, PyTorch 2.13 triggers **hard GPU ring-buffer deadlocks / kernel panics** on dual R9700s (historically observed in Radiance 0.5.0–0.5.4).
3. **RDNA4 Assembler & Native Kernel Support (`libr4d` / AITER)**:
   - `libr4d` and AITER's MoE assembly kernels (`module_moe_asm`) use direct RDNA4 wave32 assembly instructions written for LLVM 18/19 (ROCm 7.14).
   - ROCm 10's LLVM toolchain altered instruction mnemonics and register operand constraints for wave32 matrix operations, breaking native kernel compilation.

**Solution**: **ROCm 7.14.0** matches the host 7.2.4 kernel driver 100% and provides rock-solid stability with **PyTorch 2.12.1 + Triton 3.7.1 + AITER 0.1.20 + libr4d `main`**, achieving zero GPU hangs and 100% test gate pass.

---

## 🔄 Updates & Architectural Enhancements vs. `vllm-ornith-run`

| Component / Feature | Legacy `vllm-ornith-run` | `radiance-vllm-r9700` | Benefit / Impact |
| :--- | :--- | :--- | :--- |
| **vLLM Core** | `0.26.0` / `0.27.x` | **`0.28.0`** | Modern engine architecture, native `MxFp4LinearKernel` plugin API, robust chunked prefill. |
| **ROCm Base** | `6.3.3` | **`7.14.0-full`** | Full RDNA4 (`gfx1201`) toolchain, modern HIP 7.14 runtime. |
| **PyTorch** | `2.10.x` / `2.11.x` | **`2.12.1+rocm7.14`** | Hardened ROCm 7.14 ABI, patched to eliminate NCCL/CUDA symbols. |
| **Triton** | `3.4.x` / `3.5.x` | **`3.7.1` (`f0b55c0`)** | Native `tl.dot_scaled` RDNA4 matrix core lowering. |
| **AITER** | `0.1.18` | **`0.1.20`** | Built specifically for `GPU_ARCHS=gfx1201` (unified attention enabled). |
| **Transformers** | `4.49.x` | **`5.14.1`** | Pinned release + `from_json` Jinja template filter. |
| **libr4d** | `0.4.0` (unstable) | **`main` (`5dc6302` / `0.5.0`)** | Fixed GDN NaN exponent bug; 25 custom RDNA4 kernels registered. |
| **KV Cache** | BF16 (~378k tokens) | **FP8 (2.12 Million tokens)** | **5.6x KV capacity expansion**; enables 131k context at high concurrency. |
| **Prefix Caching** | Disabled / Partial | **APC + Mamba `align`** | Snapshots GDN recurrent state; **~3.6x TTFT drop** on shared agent prompts. |
| **All-Reduce** | Generic RCCL | **P2P 1-Shot All-Reduce** | Direct PCIe ring push/reduce with zero graph sync overhead. |
| **Patches** | 24 legacy hunks | **Cleaned & Ported** | Upstreamed hunks made optional; zero patch failure, AST verified. |

---

## 📊 Serving Benchmark Results

**Tested Model**: [`ornith-ai/Ornith-1.5-35B-A3B-FP8`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (35B MoE, 256 fine-grained experts, 8 active per token, 131,072 native context)  
**Configuration**: 2x AMD Radeon AI PRO R9700 (TP=2) · FP8 KV Cache · Chunked Prefill (4096 tokens) · `--max-num-seqs 16`  
**Hardware Power & Voltage Tuning**: Both R9700 GPUs undervolted by **-70 mV** with power limit capped at **235 W** per card  
**Environment**: ROCm 7.14 / vLLM 0.28.0 / PyTorch 2.12.1 / Triton 3.7.1 / libr4d `main`  

> **Model Scope Disclaimer**: Only `Ornith-1.5-35B-A3B-FP8` was benchmarked and verified for this release. Other architectures and quantization formats are experimental.

| Concurrency | Total Throughput (tok/s) | Request Rate | TTFT (p50) | TTFT (p95) | TPOT (p50) | TPOT (p95) | E2E Latency (p50) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Stream** | **74.8 tok/s** | 0.58 req/s | **44.7 ms** | 855.9 ms | **12.0 ms** | 12.6 ms | 1.58 s |
| **2 Streams** | **130.7 tok/s** | 1.01 req/s | **219.6 ms** | 309.3 ms | **13.3 ms** | 13.9 ms | 1.99 s |
| **4 Streams** | **200.1 tok/s** | 1.55 req/s | **293.5 ms** | 1,048.2 ms | **14.9 ms** | 16.0 ms | 2.94 s |
| **8 Streams** | **362.8 tok/s** | 2.81 req/s | **234.6 ms** | 379.0 ms | **20.1 ms** | 21.5 ms | 2.87 s |
| **16 Streams** | **550.1 tok/s** | 4.26 req/s | **354.5 ms** | 1,156.0 ms | **23.8 ms** | 26.6 ms | 4.11 s |

- **Pure Single-Stream Decode**: **12.0 ms / token** (~83.3 tokens/sec generation speed, 44.7 ms TTFT).
- **High-Concurrency Scaling**: **550.1 tokens/sec** at 16 concurrent streams with sub-400ms TTFT.
- **Available KV Capacity**: **2,125,645 tokens** (16.2x concurrency headroom at 131,072 context).
- **Agentic Tool Calling**: 100% verified via OpenAI-compatible endpoints with `qwen3_coder` XML tool parsing.

---

## ⚡ Core Features & RDNA4 Accelerators

1. **W4A8 fp8-WMMA Matrix Core Kernel (`radiance_mxfp4_fp8.hip`)**:
   - Hand-written 16×16×16 fp8 matrix core kernel running at **325 TFLOP/s** on `gfx1201` (vs 160 TFLOP/s for f16 WMMA and 43 TFLOP/s for stock Triton).
   - Measures 1.6–1.9x faster than tuned AITER paths at prefill shapes.
2. **P2P One-Shot All-Reduce (`ar_oneshot_2rank_exact`)**:
   - Direct PCIe P2P push/reduce mechanism replacing RCCL for TP=2 communication.
   - Zero GPU synchronization overhead in captured CUDA/HIP graphs.
3. **Fused Gated Delta Net (GDN) Chunk-Scan**:
   - Hardware-accelerated linear attention ($S^T = K \cdot Q^T$) for hybrid Qwen3.5/Ornith models.
4. **Aligned Mamba SSM Automatic Prefix Caching (APC)**:
   - `--enable-prefix-caching --mamba-cache-mode=align` snapshots and restores recurrent GDN state at block boundaries, delivering ~3.6x TTFT drops on shared agent prompts.
5. **Lossless Dynamic MTP Drafting (`RADIANCE_DYNAMIC_DRAFT=1`)**:
   - Per-slot confidence gate with verbatim n-gram tails that optimizes speculative draft depth on the fly.
6. **FP8 KV Cache Pool**:
   - 2.12M tokens KV capacity across dual cards, halving memory footprint without quality degradation.

---

## 🚀 Quickstart

### 1. Build Container from Source
```bash
git clone https://github.com/drwolfen/radiance-vllm-r9700.git
cd radiance-vllm-r9700

# Build release image (~3.66 GiB pruned)
docker build -t radiance-vllm:0.10.0 --build-arg RADIANCE_VERSION=0.10.0 .
```

### 2. Launch Production Server (Docker)
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
  --max-num-seqs 16 \
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

### 3. Launch via Docker Compose
```bash
# Uses docker-compose.yml with environment overrides
MODELS=/path/to/models docker compose up -d
docker compose logs -f
```

---

## 🛠️ Verification Gates Architecture

Every build stage is validated through standalone automated test gates:

```bash
# Gate 1: Test native HIP toolchain on gfx1201
./tests/test_hip_toolchain.sh

# Gate 2: Test Python wheels & PyTorch 2.12.1 ABI
python3 tests/test_wheels.py

# Gate 3: Test Python AST syntax across all 2,903 patched modules
python3 tests/test_patch_ast.py

# Gate 4: Test standalone micro-kernels (libr4d 25 kernels & W4A8 GEMM)
python3 tests/test_micro_kernels.py
python3 tests/test_mxfp4_layer.py

# Gate 5: Run extended serving benchmark suite
python3 tests/vllm_benchmark_suite.py --url http://localhost:8000/v1/chat/completions
```

---

## 📖 Log Messages, Notices & Diagnostics Reference

When starting the container or compiling graphs, you will observe specific log lines. Here is the complete reference explaining each message:

### 1. Hardware Startup Banners
- `┌─[ RADIANCE · GPU TOPOLOGY & BANDWIDTH ]──`
  - **Meaning**: Automated hardware startup sweep running `rocm-bandwidth-test`. Verifies direct P2P access between both R9700 cards and reports bidirectional PCIe copy bandwidth (~28 GB/s).
- `[0] AMD Radeon AI PRO R9700 gfx1201 31.9 GiB ✓ gfx1201`
  - **Meaning**: Hardware architecture check passed. Both GPUs enumerated cleanly via direct AMDSMI initialization.

### 2. Kernel Acceleration Hooks
- `[radiance] preshuffle weight-shuffle-at-load hook installed`
  - **Meaning**: Permutes FP8 weights into RDNA4 matrix fragment order in host RAM during load, eliminating runtime layout transposition overhead.
- `[radiance] fast-reduce hook armed (RADIANCE_USE_R4D_AR=1, all_reduce wrap)`
  - **Meaning**: Replaces stock RCCL all-reduce with direct PCIe P2P one-shot reduction (`ar_oneshot_2rank_exact`).
- `[radiance.gdn] gdn_chunk_scan ENABLED (head_k 128, head_v 128, chunk 64)`
  - **Meaning**: Activates fused chunked linear attention kernel for hybrid GDN models (Qwen3.5/Ornith).
- `[radiance.gemm] claimed N=... K=... M=...`
  - **Meaning**: Skinny GEMM dispatcher claimed an intermediate matrix projection for native RDNA4 assembly execution.

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
  - **Meaning**: Upstream Triton 3.7 notification recommending `TensorDescriptor`. Handled internally; zero impact on performance or precision.
- `AllReduce fusion pass is disabled.`
  - **Meaning**: AITER's generic CUDA all-reduce fusion pass is skipped in favor of Radiance's native P2P all-reduce kernel.

---

## 🔧 Environment Variables & Knobs Reference

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `HIP_VISIBLE_DEVICES` | `0,1` | Explicit HIP device indices for the two R9700 cards. |
| `VLLM_ROCM_USE_AITER` | `1` | Master switch for AITER accelerated attention kernels. |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | `1` | Enables unified attention on gfx1201. |
| `RADIANCE_USE_R4D` | `1` | Enables hand-written RDNA4 C++/HIP kernels (`libr4d`). |
| `RADIANCE_USE_R4D_AR` | `1` | Enables direct PCIe P2P one-shot all-reduce (`ar_oneshot_2rank_exact`). |
| `RADIANCE_USE_R4D_AR_QUANT` | `1` | Compressed all-reduce payload for large prefill messages. |
| `RADIANCE_FUSE_RMS_QUANT` | `1` | Fuses group-FP8 quantization into RMSNorm epilogue. |
| `RADIANCE_DYNAMIC_DRAFT` | `1` | Enables dynamic confidence-gated speculative drafting. |
| `RADIANCE_FAST_DRAFT` | `1` | Enables 2-bit MTP draft head with exact rerank. |
| `RADIANCE_MXFP4` | `1` | Relaxes CDNA4 gate for native Quark MXFP4 execution. |
| `RADIANCE_MXFP4_W4A8` | `1` | Enables hand-written W4A8 fp8-WMMA GEMM kernel (325 TFLOP/s). |

---

## ❓ Troubleshooting

| Symptom | Cause & Solution |
| :--- | :--- |
| `port 8000 is already in use` | Another service holds the port or GPUs. Check `docker ps` or stop background service (`systemctl --user stop llama-server.service`). |
| `/dev/kfd is missing` | The `amdgpu` kernel driver is not loaded on the host. Ensure driver is installed and permissions are set. |
| `HIP out of memory during warmup` | Lower `--gpu-memory-utilization` (e.g. `0.90`) or reduce `--max-model-len`. |
| `First request takes several minutes` | Normal on cold start: Torch Inductor compiles Triton graphs. Subsequent starts reuse the persistent compile cache. |
| `Tool calls outputting raw XML` | Pass `--enable-auto-tool-choice --tool-call-parser qwen3_coder` to parse XML `<tool_call>` blocks into standard API format. |

---

## 🤖 AI Synthesis & Engineering Attribution

- **Implementation Plan**: Designed and synthesized using **Gemini Flash 3.7 (High)** and **GLM-5.2**.
- **Execution & Automated Porting**: Implemented, built, verified, and benchmarked by **Gemini Flash 3.7 (High)**.

---

## 📜 Acknowledgements & Upstream Attribution

This project is built upon and directly extends the foundational research and engineering of:
- **`ggz14`**: [`radiance-vllm-mxfp4`](https://codeberg.org/ggz14/radiance-vllm-mxfp4) (Native Quark MXFP4 and RDNA4 tuning).
- **`StillDeadcode`**: [`vllm-radiance`](https://codeberg.org/StillDeadcode/vllm-radiance) & [`libr4d`](https://codeberg.org/StillDeadcode/libr4d) (RDNA4 C++/HIP kernel library, P2P all-reduce, GDN linear attention).

All custom patches and hand-written HIP kernels originated in those repositories and have been ported and validated here for modern ROCm and vLLM releases.

## 📄 License
Apache License 2.0. See [LICENSE](LICENSE) for details.
