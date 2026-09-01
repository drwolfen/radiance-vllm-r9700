#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " [Gate 1 Test] Testing HIP toolchain on gfx1201"
echo "=========================================="

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

cat << 'INNER' > "$TMPDIR/test_hip.cpp"
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void add_one(float* d) {
    *d += 1.0f;
}

int main() {
    int dev_count = 0;
    hipError_t err = hipGetDeviceCount(&dev_count);
    if (err != hipSuccess || dev_count == 0) {
        std::cerr << "FAIL: No HIP devices found (err=" << err << ")\n";
        return 1;
    }

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::cout << "Device 0: " << prop.name << " (" << prop.gcnArchName << ")\n";

    float h = 41.0f;
    float* d = nullptr;
    hipMalloc(&d, sizeof(float));
    hipMemcpy(d, &h, sizeof(float), hipMemcpyHostToDevice);
    add_one<<<1, 1>>>(d);
    hipDeviceSynchronize();
    hipMemcpy(&h, d, sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d);

    if (h == 42.0f) {
        std::cout << "SUCCESS: HIP kernel executed correctly on " << prop.gcnArchName << "\n";
        return 0;
    } else {
        std::cerr << "FAIL: Expected 42.0f, got " << h << "\n";
        return 1;
    }
}
INNER

hipcc -O3 --offload-arch=gfx1201 "$TMPDIR/test_hip.cpp" -o "$TMPDIR/test_hip"
"$TMPDIR/test_hip"
echo "Gate 1 PASS: Toolchain & GPU execution verified."
