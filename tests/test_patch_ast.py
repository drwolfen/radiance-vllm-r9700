#!/usr/bin/env python3
"""Gate 3 Test: Verify AST syntax validity of all patched modules in site-packages."""
import ast
import glob
import sys
import sysconfig

def main():
    print("==========================================")
    print(" [Gate 3 Test] Testing AST syntax of patched modules")
    print("==========================================")

    sp = sysconfig.get_paths()["purelib"]
    patterns = [
        f"{sp}/vllm/**/*.py",
        f"{sp}/radiance_*.py",
        f"{sp}/aiter/**/*.py"
    ]

    total_checked = 0
    errors = []

    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            total_checked += 1
            try:
                ast.parse(open(f, "r", encoding="utf-8").read(), filename=f)
            except Exception as e:
                errors.append((f, str(e)))

    if errors:
        print(f"FAIL: {len(errors)} file(s) failed AST parse:")
        for f, err in errors:
            print(f"  {f}: {err}")
        sys.exit(1)

    print(f"Gate 3 PASS: All {total_checked} Python files parsed cleanly.")

if __name__ == "__main__":
    main()
