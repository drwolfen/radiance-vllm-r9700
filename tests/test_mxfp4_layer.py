#!/usr/bin/env python3
"""Dedicated test for Quark MXFP4 layer selection and execution."""
import os
import sys

def main():
    print("==========================================")
    print(" [MXFP4 Test] Testing Quark MXFP4 layer selection on gfx1201")
    print("==========================================")

    os.environ["RADIANCE_MXFP4"] = "1"
    os.environ["RADIANCE_MXFP4_W4A8"] = "1"

    try:
        import radiance_mxfp4
        cls = radiance_mxfp4.kernel_class()
        assert cls is not None, "Failed to resolve radiance_mxfp4 kernel class"
        print(f"Resolved kernel class: {cls.__name__}")
        assert cls.is_supported(), "Kernel reports unsupported on current platform"
        print("Gate MXFP4 PASS: Quark MXFP4 layer selection validated on gfx1201.")
    except Exception as e:
        print(f"FAIL in MXFP4 layer test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
