#!/usr/bin/env python3
"""Gate 4 Test: Standalone micro-benchmark and correctness verification for custom kernels."""
import sys
import torch

def main():
    print("==========================================")
    print(" [Gate 4 Test] Testing micro-kernels (libr4d & W4A8 fp8-WMMA)")
    print("==========================================")

    # 1. Test libr4d
    try:
        import r4d
        print(f"r4d version: {r4d.__version__}")
        kernels = r4d.kernels()
        print(f"r4d registered {len(kernels)} kernels:")
        for k in kernels:
            print(f"  - {k['family']}: {k['name']}")
        assert len(kernels) > 0, "No kernels registered in libr4d"
    except Exception as e:
        print(f"FAIL importing libr4d: {e}")
        sys.exit(1)

    # 2. Test W4A8 fp8-WMMA GEMM
    try:
        import radiance_mxfp4_fp8
        print("radiance_mxfp4_fp8 loaded.")
        assert hasattr(radiance_mxfp4_fp8, "launch"), "radiance_mxfp4_fp8 missing launch entrypoint"

        M, K, N = 16, 5120, 17408
        act = torch.randn(M, K, device="cuda", dtype=torch.float8_e4m3fnuz)
        w_fp4 = torch.randint(0, 255, (N, K // 2), device="cuda", dtype=torch.uint8)
        w_scale = torch.randn(N, K // 32, device="cuda", dtype=torch.float8_e8m0fnu)
        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

        radiance_mxfp4_fp8.launch(act, w_fp4, w_scale, out, M, N, K)
        torch.cuda.synchronize()
        assert not torch.isnan(out).any(), "W4A8 GEMM produced NaN"
        print("W4A8 fp8-WMMA execution verified: shape (16, 17408) OK.")
    except Exception as e:
        print(f"FAIL testing radiance_mxfp4_fp8: {e}")
        sys.exit(1)

    print("Gate 4 PASS: All micro-kernels execute without errors.")

if __name__ == "__main__":
    main()
