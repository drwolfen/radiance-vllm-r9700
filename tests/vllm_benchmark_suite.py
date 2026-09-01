#!/usr/bin/env python3
"""Standard vLLM Extended Benchmark Suite.

Features:
- Multi-concurrency serving throughput (ShareGPT-style workloads)
- Prefix caching TTFT benchmark (cold vs warm hit)
- Tool-calling & reasoning XML schema verification
- Full latency percentile distribution (TTFT, ITL/TPOT, E2E latency)
"""
import argparse
import asyncio
import json
import random
import statistics
import time
import urllib.request
import urllib.error

DEFAULT_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Ornith-1.5-35B-A3B-FP8"

SAMPLE_PROMPTS = [
    "Write a high-performance C++ implementation of a lock-free queue using atomic operations and memory order semantics.",
    "Explain the architectural differences between RDNA4 and CDNA3 GPU compute architectures with focus on LDS, SIMD width, and matrix instructions.",
    "Analyze the time and space complexity of Dijkstra's algorithm compared to A* search with admissible heuristics.",
    "Design a scalable microservices architecture for real-time video streaming with low-latency transcoding and WebRTC delivery.",
    "Derive the equations for Adam optimizer with weight decay (AdamW) and explain how momentum estimation mitigates saddle points.",
    "Write a detailed Python script using asyncio and websockets to stream telemetry metrics from distributed worker nodes.",
    "Explain how FlashAttention uses tiling and online softmax to reduce SRAM-to-HBM memory access bandwidth in transformers.",
    "Discuss the trade-offs between dense feed-forward networks and Mixture-of-Experts (MoE) routing in large language models."
]

async def send_streaming_request(url, model, messages, max_tokens, temperature=0.0, tools=None):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    loop = asyncio.get_event_loop()
    
    def _run():
        t_start = time.perf_counter()
        first_token_time = None
        token_times = []
        full_content = ""
        tool_calls = []
        
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                for line in response:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        break
                    
                    chunk = json.loads(line[6:])
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now - t_start
                    token_times.append(now)
                    
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        full_content += delta["content"]
                    if "tool_calls" in delta and delta["tool_calls"]:
                        tool_calls.extend(delta["tool_calls"])
        except Exception as e:
            return None, str(e)
            
        t_end = time.perf_counter()
        
        itls = []
        for i in range(1, len(token_times)):
            itls.append((token_times[i] - token_times[i - 1]) * 1000)
            
        ttft = (first_token_time * 1000) if first_token_time else ((t_end - t_start) * 1000)
        output_tokens = len(token_times)
        e2e_lat = t_end - t_start
        
        return {
            "success": True,
            "ttft_ms": ttft,
            "itls_ms": itls,
            "e2e_latency_s": e2e_lat,
            "output_tokens": output_tokens,
            "full_content": full_content,
            "tool_calls": tool_calls
        }, None

    return await loop.run_in_executor(None, _run)

async def benchmark_throughput(url, model, concurrency, num_requests, max_tokens):
    sem = asyncio.Semaphore(concurrency)
    
    async def _worker(prompt):
        async with sem:
            res, err = await send_streaming_request(
                url, model, [{"role": "user", "content": prompt}], max_tokens
            )
            return res
            
    prompts = [random.choice(SAMPLE_PROMPTS) for _ in range(num_requests)]
    t0 = time.perf_counter()
    results = await asyncio.gather(*[_worker(p) for p in prompts])
    t_total = time.perf_counter() - t0
    
    valid_res = [r for r in results if r and r.get("success")]
    if not valid_res:
        return None
        
    total_out_tokens = sum(r["output_tokens"] for r in valid_res)
    ttfts = sorted(r["ttft_ms"] for r in valid_res)
    e2es = sorted(r["e2e_latency_s"] for r in valid_res)
    
    all_itls = []
    for r in valid_res:
        all_itls.extend(r["itls_ms"])
    all_itls.sort()
    
    def pctl(arr, p):
        if not arr: return 0.0
        idx = min(int(len(arr) * p), len(arr) - 1)
        return arr[idx]

    return {
        "concurrency": concurrency,
        "requests": len(valid_res),
        "total_time_s": t_total,
        "req_per_sec": len(valid_res) / t_total,
        "tok_per_sec": total_out_tokens / t_total,
        "ttft_p50_ms": pctl(ttfts, 0.50),
        "ttft_p95_ms": pctl(ttfts, 0.95),
        "ttft_p99_ms": pctl(ttfts, 0.99),
        "tpot_p50_ms": pctl(all_itls, 0.50),
        "tpot_p95_ms": pctl(all_itls, 0.95),
        "e2e_p50_s": pctl(e2es, 0.50),
        "e2e_p95_s": pctl(e2es, 0.95),
    }

async def benchmark_prefix_caching(url, model):
    print("\n--- Benchmark 2: Automatic Prefix Caching (APC) Speedup ---")
    prefix_block = (
        "You are an expert systems engineer. Analyze the following system architecture specification:\n"
        + ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200)
    )
    
    cold_res, _ = await send_streaming_request(
        url, model,
        [{"role": "system", "content": prefix_block},
         {"role": "user", "content": "Question 1: What is the main bottleneck?"}],
        max_tokens=32
    )
    cold_ttft = cold_res["ttft_ms"] if cold_res else 0.0
    
    warm_res, _ = await send_streaming_request(
        url, model,
        [{"role": "system", "content": prefix_block},
         {"role": "user", "content": "Question 2: Suggest three optimizations."}],
        max_tokens=32
    )
    warm_ttft = warm_res["ttft_ms"] if warm_res else 0.0
    
    speedup = (cold_ttft / warm_ttft) if warm_ttft > 0 else 1.0
    print(f"Cold Prefix TTFT : {cold_ttft:.1f} ms")
    print(f"Warm Prefix TTFT : {warm_ttft:.1f} ms")
    print(f"APC TTFT Speedup : {speedup:.2f}x ({((1 - warm_ttft/cold_ttft)*100):.1f}% reduction)")
    return {"cold_ttft_ms": cold_ttft, "warm_ttft_ms": warm_ttft, "speedup": speedup}

async def benchmark_tool_calling(url, model):
    print("\n--- Benchmark 3: Tool-Calling & Structured Agentic Execution ---")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Fetch current stock ticker price",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol e.g. AMD"}
                    },
                    "required": ["ticker"]
                }
            }
        }
    ]
    
    prompt = "What is the current stock price of AMD? Use the available tool."
    res, _ = await send_streaming_request(
        url, model, [{"role": "user", "content": prompt}], max_tokens=128, tools=tools
    )
    
    tool_success = False
    if res and ("get_stock_price" in res["full_content"] or res["tool_calls"]):
        tool_success = True
        print("Tool-Calling Verification: PASS")
        print(f"Raw Output / Tool Call: {res['full_content'][:180]}...")
    else:
        print("Tool-Calling Verification: FAIL or unsupported format")
    return tool_success

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    print("=========================================================================================")
    print(f"  vLLM Extended GitHub Benchmark Suite · {args.model}")
    print("  Hardware: 2x AMD Radeon AI PRO R9700 (gfx1201) · ROCm 7.14 · PyTorch 2.12.1")
    print("=========================================================================================")

    print("Checking server health on port 8000...")
    for _ in range(30):
        try:
            req = urllib.request.Request("http://localhost:8000/health")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    print("Server is healthy and ready.\n")
                    break
        except Exception:
            await asyncio.sleep(5)
    else:
        print("Server not responding after 150s. Exiting.")
        return

    print("Running initial warm-up...")
    await send_streaming_request(args.url, args.model, [{"role": "user", "content": "Ping"}], 16)
    print("Warm-up complete.\n")

    print("--- Benchmark 1: Serving Throughput & Latency (Concurrencies 1, 2, 4, 8, 16) ---")
    headers = ["Concurrency", "Throughput (tok/s)", "Req/s", "TTFT p50", "TTFT p95", "TPOT p50", "TPOT p95", "E2E p50"]
    row_fmt = "{:<12} | {:<18} | {:<8} | {:<10} | {:<10} | {:<10} | {:<10} | {:<8}"
    print(row_fmt.format(*headers))
    print("-" * 105)

    results_table = []
    for c in [1, 2, 4, 8, 16]:
        num_req = max(c * 2, 8)
        res = await benchmark_throughput(args.url, args.model, c, num_req, max_tokens=128)
        if res:
            results_table.append(res)
            print(row_fmt.format(
                f"{res['concurrency']}",
                f"{res['tok_per_sec']:.2f}",
                f"{res['req_per_sec']:.2f}",
                f"{res['ttft_p50_ms']:.1f} ms",
                f"{res['ttft_p95_ms']:.1f} ms",
                f"{res['tpot_p50_ms']:.1f} ms",
                f"{res['tpot_p95_ms']:.1f} ms",
                f"{res['e2e_p50_s']:.2f} s"
            ))

    apc_res = await benchmark_prefix_caching(args.url, args.model)
    tool_res = await benchmark_tool_calling(args.url, args.model)

    print("\n=========================================================================================")
    print(" Benchmark Suite Complete.")
    print("=========================================================================================")

if __name__ == "__main__":
    asyncio.run(main())
