#!/usr/bin/env python3
"""Benchmark vLLM throughput, latency, and TTFT across concurrency levels."""
import argparse
import asyncio
import json
import time
import urllib.request
import urllib.error

PROMPT_TEXT = (
    "Explain in detail the mathematical derivation of backpropagation through time (BPTT) "
    "in recurrent neural networks, including gradient clipping techniques and vanishing gradient mitigations."
)

async def single_request(session, url, model, max_tokens, prompt):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    
    t0 = time.perf_counter()
    first_token_time = None
    token_count = 0
    
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    loop = asyncio.get_event_loop()
    
    def sync_fetch():
        nonlocal first_token_time, token_count
        t_start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                if first_token_time is None:
                    first_token_time = time.perf_counter() - t_start
                token_count += 1
        t_end = time.perf_counter()
        return t_start, first_token_time or (t_end - t_start), t_end, token_count

    return await loop.run_in_executor(None, sync_fetch)

async def run_benchmark_concurrency(url, model, concurrency, max_tokens, num_requests):
    tasks = []
    # Distribute requests
    for i in range(num_requests):
        tasks.append(single_request(None, url, model, max_tokens, PROMPT_TEXT))
    
    # Run in batches of `concurrency`
    sem = asyncio.Semaphore(concurrency)
    
    async def bound_req(t):
        async with sem:
            return await t

    t_bench_start = time.perf_counter()
    results = await asyncio.gather(*[bound_req(t) for t in tasks])
    t_bench_total = time.perf_counter() - t_bench_start

    ttfts = [r[1] * 1000 for r in results]
    latencies = [(r[2] - r[0]) for r in results]
    total_tokens = sum(r[3] for r in results)
    
    ttfts.sort()
    latencies.sort()
    
    tok_per_sec = total_tokens / t_bench_total
    
    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "total_tokens": total_tokens,
        "total_time_s": t_bench_total,
        "tok_per_sec": tok_per_sec,
        "ttft_p50_ms": ttfts[len(ttfts) // 2],
        "ttft_p90_ms": ttfts[int(len(ttfts) * 0.9)],
        "latency_p50_s": latencies[len(latencies) // 2],
        "latency_p90_s": latencies[int(len(latencies) * 0.9)],
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--model", default="Ornith-1.5-35B-A3B-FP8")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    print("================================================================================")
    print(f" RADIANCE vLLM 0.28.0 BENCHMARK · Model: {args.model}")
    print(f" Dual AMD Radeon AI PRO R9700 (gfx1201) · TP=2 · ROCm 7.14 / PyTorch 2.12.1")
    print("================================================================================")

    # Warmup
    print("Running warmup request...")
    await single_request(None, args.url, args.model, 16, "Hi")
    print("Warmup complete.\n")

    concurrencies = [1, 2, 4, 8]
    print(f"{'Concurrency':<12} | {'Tokens/s':<10} | {'TTFT p50 (ms)':<14} | {'TTFT p90 (ms)':<14} | {'Total Tok':<10} | {'Time (s)':<8}")
    print("-" * 80)

    for c in concurrencies:
        num_req = max(c * 2, 4)
        res = await run_benchmark_concurrency(args.url, args.model, c, args.max_tokens, num_req)
        print(f"{res['concurrency']:<12} | {res['tok_per_sec']:<10.2f} | {res['ttft_p50_ms']:<14.1f} | {res['ttft_p90_ms']:<14.1f} | {res['total_tokens']:<10} | {res['total_time_s']:<8.2f}")

    print("================================================================================")

if __name__ == "__main__":
    asyncio.run(main())
