#!/usr/bin/env python3
"""Summarize a Google Benchmark JSON file produced by bench_pricing.

Usage:
    ./build/release/bench_pricing --benchmark_out=bench.json --benchmark_out_format=json
    python3 scripts/summarize_bench.py bench.json
"""
import json
import re
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    with open(sys.argv[1]) as fh:
        data = json.load(fh)

    ctx = data.get("context", {})
    rows = {b["name"]: b for b in data["benchmarks"] if b.get("run_type", "iteration") == "iteration"}

    def rate(name):
        return rows[name]["items_per_second"] if name in rows else None

    print("## Environment\n")
    print(f"- host CPUs: {ctx.get('num_cpus')} @ {ctx.get('mhz_per_cpu')} MHz, "
          f"hardware threads: {ctx.get('hardware_threads', '?')}")
    print(f"- SIMD backend: {ctx.get('optengine_simd_backend', '?')}")
    print(f"- build type: {ctx.get('library_build_type', '?')}\n")

    print("## Analytical BSM (price + delta, gamma, vega, theta, rho)\n")
    print("| Contracts | Scalar (M options/s) | SIMD (M options/s) | Speedup |")
    print("|---:|---:|---:|---:|")
    for n in (1_000, 100_000, 1_000_000):
        s, v = rate(f"BM_BSM_Scalar/{n}"), rate(f"BM_BSM_SIMD/{n}")
        if s and v:
            print(f"| {n:,} | {s / 1e6:.1f} | {v / 1e6:.1f} | {v / s:.2f}x |")

    print("\n## Implied volatility (realistic OTM market chain)\n")
    print("| Contracts | Solver | M options/s | ns / option | Converged | Max abs error |")
    print("|---:|---|---:|---:|---:|---:|")
    for name, b in rows.items():
        m = re.fullmatch(r"(BM_IV_Solver(?:_BrentOnly)?)/(\d+)", name)
        if not m:
            continue
        label = "Newton + Brent fallback" if m.group(1) == "BM_IV_Solver" else "Brent only"
        ips = b["items_per_second"]
        print(f"| {int(m.group(2)):,} | {label} | {ips / 1e6:.2f} | {1e9 / ips:.0f} | "
              f"{b.get('converged_pct', float('nan')):.1f}% | {b.get('max_abs_err', float('nan')):.1e} |")

    for model, title in (("GBM", "1,000,000 paths x 252 steps"), ("Heston", "100,000 paths x 252 steps")):
        mc = []
        for name, b in rows.items():
            m = re.match(rf"BM_MonteCarlo_{model}/threads:(\d+)", name)
            if m:
                mc.append((int(m.group(1)), b))
        if not mc:
            continue
        mc.sort()
        base = mc[0][1]["real_time"]
        print(f"\n## Monte Carlo {model} ({title}, antithetic)\n")
        print("| Threads | Wall time (ms) | M path-steps/s | Speedup vs 1 thread | Price | Std error |")
        print("|---:|---:|---:|---:|---:|---:|")
        for threads, b in mc:
            print(f"| {threads} | {b['real_time']:.0f} | {b['items_per_second'] / 1e6:.0f} | "
                  f"{base / b['real_time']:.2f}x | {b.get('price', float('nan')):.5f} | "
                  f"{b.get('std_error', float('nan')):.5f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
