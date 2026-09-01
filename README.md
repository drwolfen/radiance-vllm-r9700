# radiance-vllm · AMD Radeon AI PRO R9700 (gfx1201)

High-performance vLLM 0.28 inference stack for **Dual AMD Radeon AI PRO R9700 (gfx1201 / RDNA4)** powered by **ROCm 7.14 / 7.2.4**, **PyTorch 2.12.1**, **Triton 3.7.x**, **AITER 0.1.20**, and **native `libr4d` C++/HIP kernels**.

---

## ⚡ Key Highlights & Architecture

- **Hardware Target**: Dual AMD Radeon AI PRO R9700 (`gfx1201`, 32 GiB VRAM per card, PCIe P2P direct interconnect).
- **Core Software Stack**:
  - **Base OS**: Ubuntu 24.04 LTS (`rocm/dev-ubuntu-24.04:7.14.0-full`)
  - **vLLM**: `0.28.0` (with 23 ported RDNA4/gfx1201 optimizations)
  - **PyTorch**: `2.12.1+rocm7.14` (`release/2.12` @ `6bbd260`)
  - **Torchvision**: `0.27.1+df56172`
  - **Triton**: ROCm fork `internal/3.7.x` (`f0b55c0`)
  - **AITER**: `v0.1.20` (`GPU_ARCHS=gfx1201`)
  - **Transformers**: `5.14.1`
  - **libr4d**: `main` (`5dc6302` / 25 custom RDNA4 kernels)
- **Custom Hardware Accelerators**:
  - **W4A8 fp8-WMMA GEMM**: Hand-written 16×16×16 fp8 matrix core kernel measuring 325 TFLOP/s on gfx1201.
  - **P2P One-Shot All-Reduce**: Direct PCIe P2P ring all-reduce (`ar_oneshot_2rank_exact`) replacing RCCL for low-latency TP=2 communication.
  - **Fused GDN Chunk-Scan**: Fused linear attention kernel ($S^T = K \cdot Q^T$) for hybrid Qwen3.5/Ornith models.
  - **Automatic Prefix Caching (APC) + Mamba Alignment**: Snapshotting recurrent state with paged attention for ~3.6x TTFT reduction on shared agent prefixes.
  - **FP8 KV Cache**: 2.12M tokens KV capacity across dual cards.

---

## 📊 Serving Benchmark Results

**Workload**: `Ornith-1.5-35B-A3B-FP8` (35B MoE, 256 fine-grained experts, 8 active, 131k context)  
**System**: 2x AMD Radeon AI PRO R9700 (TP=2) · ROCm 7.14 / PyTorch 2.12.1  
**Benchmark Suite**: Multi-concurrency streaming throughput ladder & latency percentiles

| Concurrency | Throughput (tok/s) | Request Rate | TTFT (p50) | TTFT (p95) | TPOT (p50) | TPOT (p95) | E2E Latency (p50) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Stream** | **65.1 tok/s** | 0.50 req/s | 158.8 ms | 1,046.2 ms | 12.1 ms | 12.6 ms | 1.70 s |
| **2 Streams** | **126.1 tok/s** | 0.98 req/s | 325.4 ms | 365.6 ms | 13.3 ms | 14.0 ms | 2.05 s |
| **4 Streams** | **187.4 tok/s** | 1.45 req/s | 366.3 ms | 1,258.7 ms | 15.3 ms | 16.3 ms | 3.14 s |
| **8 Streams** | **352.7 tok/s** | 2.74 req/s | 307.2 ms | 392.2 ms | 20.2 ms | 21.6 ms | 2.94 s |
| **16 Streams** | **334.9 tok/s** | 2.60 req/s | 3,072.7 ms | 4,111.2 ms | 19.8 ms | 21.3 ms | 5.73 s |

- **Peak Concurrent Throughput**: **352.7 tokens/sec** at 8 streams with balanced TPOT (~20ms).
- **Single-Stream Inter-Token Latency (TPOT)**: **12.1 ms / token** (~82.6 tokens/sec pure decode speed).
- **FP8 KV Capacity**: **2,125,645 tokens** (16.2x max concurrency at 131k context).
- **XML Tool Calling**: 100% verified via OpenAI-compatible endpoints (`qwen3_coder` tool parser).

---

## 🚀 Quickstart

### 1. Build Docker Image
```bash
git clone https://github.com/<your-username>/radiance-vllm-rocm10.git
cd radiance-vllm-rocm10

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

### 3. Run Verification Tests
```bash
# Run standalone micro-kernel verification (Gate 4)
python3 tests/test_micro_kernels.py

# Run Quark MXFP4 layer selection verification
python3 tests/test_mxfp4_layer.py

# Run extended benchmark suite
python3 tests/vllm_benchmark_suite.py --url http://localhost:8000/v1/chat/completions
```

---

## 🛠️ Verification Gates

Every build stage is validated through automated test gates:
- **Gate 1: HIP Toolchain**: Native C++ compilation and GPU execution on `gfx1201`.
- **Gate 2: Python Wheels ABI**: Import and matmul verification on PyTorch 2.12.1 / Triton 3.7.1 / AITER 0.1.20.
- **Gate 3: AST Syntax Integrity**: Python AST parsing over 2,900 patched vLLM/transformers files.
- **Gate 4: Micro-Kernels**: Verification of 25 `libr4d` kernels and `radiance_mxfp4_fp8` W4A8 GEMM.
- **Gate 5: End-to-End Live Inference**: Real-time streaming generation and tool calling.

---

## 📜 Acknowledgements & Attribution

This project builds upon and extends the foundational work of:
- **`ggz14`**: [`radiance-vllm-mxfp4`](https://codeberg.org/ggz14/radiance-vllm-mxfp4)
- **`StillDeadcode`**: [`vllm-radiance`](https://codeberg.org/StillDeadcode/vllm-radiance) & [`libr4d`](https://codeberg.org/StillDeadcode/libr4d)

All custom patches and hand-written HIP kernels originated in those repositories and have been updated here for modern ROCm and vLLM releases.

## 📄 License
Apache License 2.0. See [LICENSE](LICENSE) for details.
