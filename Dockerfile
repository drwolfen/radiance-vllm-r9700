# radiance-vllm-rocm10: vLLM 0.28.0 / PyTorch 2.12.1 / Triton 3.7.x / AITER 0.1.20 stack for RDNA4 (gfx1201)
# Four-stage build on AMD ROCm dev image:
#   1. builder    compile torch/triton/torchvision/aiter/vLLM from source into /wheels
#   2. rocmprune  cut ROCm tree down to gfx1201
#   3. assemble   install wheels, apply patches, compile libr4d (main) and W4A8 fp8-WMMA kernel
#   4. final      clean Ubuntu 24.04 release image with pruned ROCm and venv

ARG ROCM_BASE=rocm/dev-ubuntu-24.04:7.14.0-full
ARG GFX_ARCH=gfx1201
ARG RELEASE_BASE=ubuntu:24.04

# Verified component pins
ARG TORCH_VERSION=2.12.1
ARG TRITON_REPO=https://github.com/ROCm/triton.git
ARG TRITON_COMMIT=f0b55c0
ARG TORCHVISION_VERSION=0.27.1
ARG AITER_VERSION=0.1.20
ARG VLLM_VERSION=0.28.0
ARG TRANSFORMERS_VERSION=5.14.1
ARG NUMPY_VERSION=2.3.5
ARG RBT_VERSION=rocm-6.4.4
ARG R4D_REPO=https://codeberg.org/StillDeadcode/libr4d.git
ARG R4D_VERSION=main

# =====================================================================================
# STAGE 1 builder: compile stack from source into /wheels
# =====================================================================================
FROM ${ROCM_BASE} AS builder
ARG GFX_ARCH
ARG TORCH_VERSION
ARG TRITON_REPO
ARG TRITON_COMMIT
ARG TORCHVISION_VERSION
ARG AITER_VERSION
ARG NUMPY_VERSION
ARG MAX_JOBS=32

ENV DEBIAN_FRONTEND=noninteractive \
    PYTORCH_ROCM_ARCH=${GFX_ARCH} \
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
    USE_ROCM=1 USE_CUDA=0 MAX_JOBS=${MAX_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
      git python3.12-venv build-essential ccache pkg-config \
      libdrm-dev libnuma-dev libelf-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/py
ENV PATH=/opt/py/bin:$PATH
RUN pip install -U pip wheel setuptools "setuptools-scm>=8.0" "cmake<4" ninja pybind11 "numpy==${NUMPY_VERSION}" \
      pyyaml typing_extensions cffi requests
RUN mkdir -p /wheels

# --- PyTorch 2.12.1 ---
RUN git clone --depth 1 -b v${TORCH_VERSION} --recurse-submodules --shallow-submodules \
        https://github.com/pytorch/pytorch.git /src/pytorch \
    && cd /src/pytorch \
    && pip install -r requirements.txt \
    && python tools/amd_build/build_amd.py \
    && USE_MAGMA=0 USE_MKLDNN=1 BUILD_TEST=0 USE_NCCL=1 USE_RCCL=1 \
       USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_AOTRITON=0 \
       PYTORCH_BUILD_VERSION=${TORCH_VERSION}+rocm7.14 PYTORCH_BUILD_NUMBER=1 \
       python setup.py bdist_wheel \
    && cp dist/*.whl /wheels/ && pip install dist/*.whl && rm -rf /src/pytorch

# --- Triton (ROCm fork @ f0b55c0) ---
RUN git clone ${TRITON_REPO} /src/triton \
    && cd /src/triton && git checkout ${TRITON_COMMIT} \
    && pip wheel --no-build-isolation --no-deps . -w /wheels \
    && pip install /wheels/triton-*.whl && rm -rf /src/triton

# --- Torchvision 0.27.1 ---
RUN git clone --depth 1 -b v${TORCHVISION_VERSION} https://github.com/pytorch/vision.git /src/vision \
    && cd /src/vision && FORCE_CUDA=1 USE_ROCM=1 pip wheel --no-build-isolation --no-deps . -w /wheels \
    && rm -rf /src/vision

# --- AITER v0.1.20 (gfx1201) ---
RUN git clone --recursive --shallow-submodules -b v${AITER_VERSION} https://github.com/ROCm/aiter.git /src/aiter \
    && cd /src/aiter && GPU_ARCHS=${GFX_ARCH} PREBUILD_KERNELS=0 \
       SETUPTOOLS_SCM_PRETEND_VERSION=${AITER_VERSION} \
       pip wheel --no-build-isolation --no-deps . -w /wheels \
    && pip install /wheels/*aiter-*.whl && rm -rf /src/aiter

# --- vLLM 0.28.0 ---
ARG VLLM_VERSION
RUN git clone --depth 1 -b v${VLLM_VERSION} https://github.com/vllm-project/vllm.git /src/vllm \
    && cd /src/vllm && python use_existing_torch.py \
    && pip install "setuptools-rust>=1.9.0" \
    && VLLM_TARGET_DEVICE=rocm VLLM_VERSION_OVERRIDE=${VLLM_VERSION} \
       pip wheel --no-build-isolation --no-deps . -w /wheels \
    && rm -rf /src/vllm

# --- rocm-bandwidth-test ---
ARG RBT_VERSION
RUN git clone --depth 1 -b ${RBT_VERSION} https://github.com/ROCm/rocm_bandwidth_test.git /src/rbt \
    && cmake -S /src/rbt -B /src/rbt/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm \
         -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,/opt/rocm/core/lib:/opt/rocm/lib" \
    && cmake --build /src/rbt/build -j 16 \
    && mkdir -p /artifacts && cp /src/rbt/build/rocm-bandwidth-test /artifacts/ \
    && rm -rf /src/rbt

# =====================================================================================
# STAGE 2 rocmprune: prune ROCm tree to gfx1201
# =====================================================================================
FROM ${ROCM_BASE} AS rocmprune
ARG GFX_ARCH
COPY prune_rocm.sh /tmp/prune_rocm.sh
RUN bash /tmp/prune_rocm.sh ${GFX_ARCH} && rm -f /tmp/prune_rocm.sh

# =====================================================================================
# STAGE 3 assemble: install wheels, apply patches, compile libr4d and W4A8 GEMM
# =====================================================================================
FROM ${ROCM_BASE} AS assemble
ARG GFX_ARCH
ARG AITER_VERSION
ARG VLLM_VERSION
ARG TRANSFORMERS_VERSION
ARG NUMPY_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12-venv git \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/vllm
RUN python3 -m venv /opt/vllm
ENV PATH=/opt/vllm/bin:$PATH
ENV SP=/opt/vllm/lib/python3.12/site-packages

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir -U pip wheel setuptools "numpy==${NUMPY_VERSION}" \
 && pip install --no-cache-dir --no-deps \
      /wheels/torch-*.whl /wheels/triton-*.whl /wheels/torchvision-*.whl /wheels/*aiter-*.whl \
 && pip install --no-cache-dir /wheels/vllm-*.whl "transformers==${TRANSFORMERS_VERSION}" \
 && pip install --no-cache-dir /opt/rocm/share/amd_smi pillow pybind11 "amd-quark==0.12.post1" \
 && rm -rf /wheels /root/.cache

ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_PLATFORM=amd \
    VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=${GFX_ARCH} RADIANCE_GFX_ARCH=${GFX_ARCH} \
    HIP_ARCHITECTURES=${GFX_ARCH} AMDGPU_TARGETS=${GFX_ARCH} GPU_ARCHS=${GFX_ARCH} \
    SAFETENSORS_FAST_GPU=1 TOKENIZERS_PARALLELISM=false TRITON_CACHE_AUTOTUNING=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy runtime modules and configs
COPY radiance_amdsmi.py radiance_amdsmi.pth \
     radiance_kernels.py radiance_vit_attn.py radiance_allreduce.py \
     radiance_draft.py radiance_draft_gpu.py radiance_drafthead.py radiance_gemm.py \
     radiance_r4d_attn.py radiance_gdn.py radiance_w4.py radiance_mxfp4.py ${SP}/
COPY fp8-configs/ ${SP}/vllm/model_executor/layers/quantization/utils/configs/
COPY moe-configs/ ${SP}/vllm/model_executor/layers/fused_moe/configs/
COPY mxfp4-configs/ ${SP}/aiter/ops/triton/configs/gemm/

# Apply patches
COPY patch_*.py install_radiance_hooks.py _patchlib.py /opt/patches/
RUN set -eu; cd /opt/patches; \
    for p in patch_gfx1201 patch_radiance_dispatch patch_skinny_gemm patch_unified_attention_lds \
             patch_gdn_wmma patch_preshuffle patch_radiance_fusion install_radiance_hooks \
             patch_unpad patch_mtp_mm_mask patch_mtp_loopbreak patch_qwen3_toolparse patch_from_json_filter \
             patch_dynamo_metrics patch_conv1d_blockn patch_r4d patch_dflash_base \
             patch_dflash_fused_kv_fp8 patch_dflash_w4 patch_gdn_metadata \
             patch_quark_mxfp4 patch_ar_maxbytes patch_topk_triton_rows patch_qwen3_thinkoff; do \
      echo "== applying $p =="; python "$p.py"; \
    done; \
    python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('${SP}/radiance_*.py')]; print('radiance modules parse OK')"

# Clone and compile libr4d (main @ 5dc6302)
ARG R4D_REPO
ARG R4D_VERSION
RUN git clone --depth 1 ${R4D_REPO} /src/libr4d \
 && cd /src/libr4d && GFX_ARCH=${GFX_ARCH} OUT=${SP}/r4d.so ./build.sh \
 && python -c "import torch, r4d; print('r4d', r4d.__version__, 'built:'); [print('   ', k['family'], k['name']) for k in r4d.kernels()]" \
 && rm -rf /src/libr4d

# Compile W4A8 fp8-WMMA GEMM
COPY radiance_mxfp4_fp8.hip /opt/patches/
RUN INC=$(python -m pybind11 --includes); \
    hipcc -O3 -std=c++17 -fPIC -shared --offload-arch=${GFX_ARCH} -Wno-unused-result \
      $INC /opt/patches/radiance_mxfp4_fp8.hip -o ${SP}/radiance_mxfp4_fp8.so \
 && python -c "import torch, radiance_mxfp4_fp8 as m; assert hasattr(m, 'launch'); print('radiance_mxfp4_fp8 built')"

# Strip debug symbols
RUN find /opt/vllm -type f -name '*.so*' ! -name 'r4d.so' ! -name 'radiance_mxfp4_fp8.so' \
      -exec strip --strip-unneeded {} + 2>/dev/null || true; \
    find /opt/vllm -name '__pycache__' -type d -prune -exec rm -rf {} + || true; \
    echo "extensions stripped"

# =====================================================================================
# STAGE 4 final: clean release image
# =====================================================================================
FROM ${RELEASE_BASE} AS final
ARG GFX_ARCH
ARG AITER_VERSION
ARG VLLM_VERSION
ARG TRANSFORMERS_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3.12 python3.12-dev g++ libnuma-dev numactl curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rocmprune /opt/rocm /opt/rocm
COPY --from=rocmprune /etc/alternatives /etc/alternatives
COPY --from=assemble /opt/vllm /opt/vllm
COPY --from=builder /artifacts/rocm-bandwidth-test /usr/local/bin/rocm-bandwidth-test

ENV VIRTUAL_ENV=/opt/vllm \
    PATH=/opt/vllm/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_PLATFORM=amd \
    VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=${GFX_ARCH} RADIANCE_GFX_ARCH=${GFX_ARCH} \
    HIP_ARCHITECTURES=${GFX_ARCH} AMDGPU_TARGETS=${GFX_ARCH} GPU_ARCHS=${GFX_ARCH} \
    SAFETENSORS_FAST_GPU=1 TOKENIZERS_PARALLELISM=false TRITON_CACHE_AUTOTUNING=1 \
    PYTHONDONTWRITEBYTECODE=1

ENV RADIANCE_USE_R4D=1 \
    RADIANCE_USE_R4D_AR=1 RADIANCE_USE_R4D_AR_QUANT=1 \
    RADIANCE_PRESHUFFLE=1 RADIANCE_FUSE_RMS_QUANT=1 \
    RADIANCE_DYNAMIC_DRAFT=1 RADIANCE_DRAFT_SCHEDULE=1:8,2:7,4:6,8:5,16:4 RADIANCE_DRAFT_TAU=0.35 \
    RADIANCE_RUN_BWTEST=1

# Copy test suite into image
COPY tests/ /opt/vllm/tests/

# Post-build verification
RUN python -c 'import os, torch, vllm._C, amdsmi, importlib.metadata as m; \
import r4d, radiance_mxfp4_fp8; \
assert hasattr(radiance_mxfp4_fp8, "launch"); \
print("Stack OK | vllm", m.version("vllm"), "| torch", torch.__version__, "| aiter", m.version("amd-aiter"), \
      "| torchvision", m.version("torchvision"), "| triton", m.version("triton"), \
      "| transformers", m.version("transformers"), "| r4d", r4d.__version__, "| mxfp4-w4a8 ok")'

# JIT toolchain probe
RUN printf '%s\n' \
      '#include <hip/hip_runtime.h>' \
      '#include <cmath>' \
      '#include <pybind11/pybind11.h>' \
      '__global__ void k(float* o) { o[threadIdx.x] = 1.0f; }' \
      'PYBIND11_MODULE(_jit_probe, m) { m.def("f", [](double x) { return std::sqrt(x); }); }' \
      > /tmp/_jit_probe.hip \
 && hipcc -O3 -fPIC -shared -std=c++20 --offload-arch=${GFX_ARCH} \
      $(python -m pybind11 --includes) /tmp/_jit_probe.hip -o /tmp/_jit_probe.so \
 && python -c "import torch, sys; sys.path.insert(0, '/tmp'); import _jit_probe; assert _jit_probe.f(4.0) == 2.0" \
 && rm -f /tmp/_jit_probe.hip /tmp/_jit_probe.so \
 && echo "runtime JIT toolchain OK (hipcc + libstdc++ headers + Python.h + pybind11)"

ARG RADIANCE_VERSION=0.10.0
ENV RADIANCE_VERSION=${RADIANCE_VERSION}
COPY VERSION /opt/radiance_version
COPY radiance_preamble.py /opt/radiance_preamble.py
COPY radiance_entrypoint.sh /opt/radiance_entrypoint.sh
RUN chmod +x /opt/radiance_entrypoint.sh
ENTRYPOINT ["/opt/radiance_entrypoint.sh"]
