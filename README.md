# radiance-vllm-rocm10

An optimized inference stack for **AMD Radeon AI PRO R9700 (gfx1201 / RDNA4)** with **ROCm 10.0 / 7.2.4**, **vLLM 0.28+**, **PyTorch 2.13+**, and **native RDNA4 kernels**.

## Acknowledgements & Upstream Attribution

This repository is built upon and directly extends the work of:
- **`ggz14`**: [`radiance-vllm-mxfp4`](https://codeberg.org/ggz14/radiance-vllm-mxfp4) (Native MXFP4 and RDNA4 tuning).
- **`StillDeadcode`**: [`vllm-radiance`](https://codeberg.org/StillDeadcode/vllm-radiance) and [`libr4d`](https://codeberg.org/StillDeadcode/libr4d) (RDNA4 C++/HIP kernel library, Paged Attention, GDN, P2P All-Reduce).

All custom patches and hand-written HIP kernels originated in those repositories and have been updated here for modern ROCm and vLLM releases.

## Features & Upgrades
- **ROCm 10.0.0 / 7.2.4 Support**: Built against AMD's unified TheRock toolchain for Ubuntu 24.04 LTS.
- **vLLM 0.28+ Integration**: Cleaned patch suite removing upstreamed backports.
- **libr4d Custom Kernels**: High-performance paged attention ($S^T = K \cdot Q^T$), fused Gated Delta Net, TP=2 one-shot P2P all-reduce.
- **Native Quark MXFP4 W4A8 GEMM**: Hand-written fp8-WMMA kernel running at 325 TFLOP/s on gfx1201.
- **5-Stage Verification Gates**: Automated test scripts in `tests/` validating each build layer before deployment.

## Quickstart

### Build from Source
```bash
# 1. Run Gate 1 probe
./tests/test_hip_toolchain.sh

# 2. Build release image
docker build -t radiance-vllm:0.10.0 --build-arg RADIANCE_VERSION=0.10.0 .
```

### Run Tests
```bash
# Run standalone micro-kernel verification
python3 tests/test_micro_kernels.py

# Run MXFP4 layer selection verification
python3 tests/test_mxfp4_layer.py
```

## License
Apache License 2.0. See [LICENSE](LICENSE) for details.
