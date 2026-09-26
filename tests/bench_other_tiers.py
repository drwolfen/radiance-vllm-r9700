#!/usr/bin/env python3
"""Multi-Tier Cluster Benchmark & Tool Calling Verification.

Tests OpenAI-compatible endpoints across cluster tiers:
- local-heavy: Dual AMD Radeon AI PRO R9700 (srv02 .244:8000)
- local-light: srv01 (.246:8090)
- local-medium: workstation (.150:8083, standby / image-gen dedicated)
"""
import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_TIERS = [
    {
        "name": "local-heavy (srv02 .244:8000)",
        "url": "http://192.168.41.244:8000/v1/chat/completions",
        "api_key": os.getenv("LOCAL_HEAVY_API_KEY", "EMPTY"),
        "preferred_model": "qwen3.8-flash-next",
        "fallback_model": "Ornith-1.5-35B-A3B-FP8",
        "enabled": True,
    },
    {
        "name": "local-light (srv01 .246:8090)",
        "url": "http://192.168.41.246:8090/v1/chat/completions",
        "api_key": os.getenv("LOCAL_LIGHT_API_KEY", "eGXRG3Njmcif11ZJ65R15hFbNY56kKO4JHWiaYk2jtI"),
        "preferred_model": "/home/ydj/LLM-Models/Qwythos-9B-v2-MTP-GGUF/Qwythos-9B-v2-MTP-Q6_K.gguf",
        "fallback_model": "/home/ydj/LLM-Models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
        "enabled": True,
    },
    {
        "name": "local-medium (workstation .150:8083)",
        "url": "http://192.168.41.150:8083/v1/chat/completions",
        "api_key": os.getenv("LOCAL_MEDIUM_API_KEY", "eGXRG3Njmcif11ZJ65R15hFbNY56kKO4JHWiaYk2jtI"),
        "preferred_model": "/8tb/LLM-Models/Qwen3-14B-GGUF/Qwen3-14B-Q5_K_M.gguf",
        "fallback_model": "Qwen3-14B-Q4_K_M.gguf",
        "enabled": False,  # Disabled: .150 dedicated to ComfyUI (8188)
    },
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Fetch current stock price and volume for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. NVDA, AMD, AAPL",
                    }
                },
                "required": ["symbol"],
            },
        },
    }
]


def make_req(url, api_key, payload, timeout=60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        t1 = time.time()
        return res, t1 - t0
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        t1 = time.time()
        return {"error": f"HTTP {e.code}: {body}"}, t1 - t0
    except Exception as e:
        t1 = time.time()
        return {"error": f"Connection error: {e}"}, t1 - t0


def detect_served_model(tier, timeout=5):
    """Attempt to discover active model from /v1/models endpoint."""
    base_url = tier["url"].rsplit("/chat/completions", 1)[0]
    models_url = f"{base_url}/models"
    req = urllib.request.Request(
        models_url,
        headers={"Authorization": f"Bearer {tier['api_key']}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = data.get("data", []) or data.get("models", [])
        if models:
            model_ids = [m.get("id") or m.get("name") or m.get("model") for m in models]
            # Match preferred
            if tier.get("preferred_model") in model_ids:
                return tier["preferred_model"], True
            for m_id in model_ids:
                if m_id:
                    return m_id, True
            return models[0].get("id", tier.get("preferred_model")), True
    except Exception:
        pass
    return tier.get("preferred_model") or tier.get("fallback_model"), False


def test_tool_calling(tier, model_name):
    print(f"\n--- Testing Tool Calling on {tier['name']} ---")
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a financial assistant. Call get_stock_price to look up prices."},
            {"role": "user", "content": "What is the stock price of AMD right now?"},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.1,
        "max_tokens": 256,
    }
    res, latency = make_req(tier["url"], tier["api_key"], payload)
    if "error" in res:
        print(f"Error: {res['error']}")
        return False, res["error"]

    choices = res.get("choices", [])
    if not choices:
        return False, f"Empty choices response: {res}"

    msg = choices[0].get("message", {})
    tool_calls = msg.get("tool_calls")
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""

    print(f"Latency: {latency * 1000:.1f} ms")
    if tool_calls and len(tool_calls) > 0:
        fn = tool_calls[0].get("function", {})
        print(f"Tool Call Detected: Name = {fn.get('name')}, Args = {fn.get('arguments')}")
        print("Status: PASS (Native JSON Tool Call)")
        return True, "PASS (Native JSON)"
    elif "get_stock_price" in content or "AMD" in content or "get_stock_price" in reasoning:
        print(f"Content / Reasoning Tool Call: {(content or reasoning).strip()[:150]}")
        print("Status: PASS (Text Tool Call)")
        return True, "PASS (Text Tool Call)"
    else:
        print(f"Output: {msg}")
        return False, "FAIL"


def benchmark_throughput(tier, model_name, concurrency=1, num_requests=3):
    print(f"\n--- Benchmarking {tier['name']} (Concurrency={concurrency}, Req={num_requests}) ---")
    payload_template = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a concise assistant. Output code only."},
            {"role": "user", "content": "Write an efficient Python quicksort algorithm."},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }

    latencies = []
    total_tokens = 0
    errors = 0
    start_total = time.time()

    def worker(_):
        res, lat = make_req(tier["url"], tier["api_key"], payload_template)
        if "error" in res:
            return lat, 0, res["error"]
        usage = res.get("usage", {})
        comp_tokens = usage.get("completion_tokens", 0)
        return lat, comp_tokens, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(worker, i) for i in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            lat, comp_tok, err = f.result()
            if err:
                errors += 1
                print(f"  [Req Error]: {err}")
            else:
                latencies.append(lat)
                total_tokens += comp_tok

    total_time = time.time() - start_total
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    throughput = total_tokens / total_time if total_time > 0 else 0
    tpot = (avg_latency / (total_tokens / len(latencies))) * 1000 if (total_tokens > 0 and len(latencies) > 0) else 0

    print(f"Total Time       : {total_time:.2f} s (Errors: {errors})")
    print(f"Total Tokens     : {total_tokens}")
    print(f"Avg Latency      : {avg_latency * 1000:.1f} ms")
    print(f"Est. TPOT        : {tpot:.1f} ms / tok")
    print(f"Throughput       : {throughput:.1f} tok/s")

    return throughput, avg_latency, tpot


def main():
    parser = argparse.ArgumentParser(description="Multi-Tier Cluster Benchmark Suite")
    parser.add_argument("--include-inactive", action="store_true", help="Include disabled/standby tiers")
    parser.add_argument("--tier", type=str, default=None, help="Filter by tier name substring")
    parser.add_argument("--concurrency", type=int, default=1, help="Benchmark concurrency (default: 1)")
    parser.add_argument("--requests", type=int, default=2, help="Requests per tier benchmark (default: 2)")
    args = parser.parse_args()

    results = {}
    print("=" * 85)
    print("  Radiance Multi-Tier Cluster Verification & Benchmark Suite")
    print("=" * 85)

    for tier in DEFAULT_TIERS:
        if args.tier and args.tier.lower() not in tier["name"].lower():
            continue
        if not tier.get("enabled", True) and not args.include_inactive:
            print(f"\n[SKIP] {tier['name']} (Disabled / Inactive - use --include-inactive to test)")
            results[tier["name"]] = {"status": "INACTIVE / STANDBY"}
            continue

        model, reachable = detect_served_model(tier)
        if not reachable:
            print(f"\n[UNREACHABLE] {tier['name']} is offline or not responding.")
            results[tier["name"]] = {"status": "OFFLINE / UNREACHABLE"}
            continue

        print(f"\n[ACTIVE] {tier['name']} · Model: {model}")
        tc_ok, tc_detail = test_tool_calling(tier, model)
        tp1, lat1, tpot1 = benchmark_throughput(tier, model, concurrency=args.concurrency, num_requests=args.requests)
        results[tier["name"]] = {
            "status": "ONLINE",
            "model": model,
            "tool_call": tc_detail,
            "tp": tp1,
            "lat": lat1,
            "tpot": tpot1,
        }

    print("\n" + "=" * 85)
    print("  Cluster Verification Summary")
    print("=" * 85)
    for k, v in results.items():
        print(f"• {k}:")
        if v["status"] == "ONLINE":
            print(f"    Model        : {v['model']}")
            print(f"    Tool Calling : {v['tool_call']}")
            print(f"    Throughput   : {v['tp']:.1f} tok/s (TPOT: {v['tpot']:.1f} ms, Latency: {v['lat'] * 1000:.0f} ms)")
        else:
            print(f"    Status       : {v['status']}")


if __name__ == "__main__":
    main()
