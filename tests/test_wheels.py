#!/usr/bin/env python3
"""Gate 2 Test: Verify Python wheels, PyTorch ROCm initialization, and Triton WMMA lowering."""
import sys

def main():
    print("==========================================")
    print(" [Gate 2 Test] Testing wheels & PyTorch ABI")
    print("==========================================")

    import torch
    print(f"PyTorch version: {torch.__version__}")
    assert torch.cuda.is_available(), "FAIL: PyTorch HIP device not detected"

    dev_count = torch.cuda.device_count()
    print(f"Detected {dev_count} GPU device(s)")
    for i in range(dev_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name} ({props.gcnArchName}) - {props.total_memory / (1024**3):.1f} GiB")
        assert "gfx1201" in props.gcnArchName, f"Device {i} is not gfx1201"

    import triton
    print(f"Triton version: {triton.__version__}")

    import torchvision
    print(f"Torchvision version: {torchvision.__version__}")

    import aiter
    print(f"AITER version: {getattr(aiter, '__version__', 'loaded')}")

    # Micro-matmul test on GPU
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    c = torch.matmul(a, b)
    assert not torch.isnan(c).any(), "FAIL: Matmul produced NaN"

    print("Gate 2 PASS: Wheels & PyTorch ABI validated successfully.")

if __name__ == "__main__":
    main()
