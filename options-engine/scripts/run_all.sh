#!/usr/bin/env bash
# Builds and verifies everything:
#   1. Debug   (-O0 -g, ASan + UBSan)  -> ctest
#   2. Release (-O3 -march=native -ffast-math) -> ctest, benchmarks, summary table
# Usage: scripts/run_all.sh [extra cmake args...]
set -euo pipefail
cd "$(dirname "$0")/.."

GENERATOR=()
if command -v ninja >/dev/null 2>&1; then GENERATOR=(-G Ninja); fi

echo "==> Debug build (sanitizers) + tests"
cmake -S . -B build/debug "${GENERATOR[@]}" -DCMAKE_BUILD_TYPE=Debug "$@"
cmake --build build/debug --parallel
ctest --test-dir build/debug --output-on-failure

echo "==> Release build + tests"
cmake -S . -B build/release "${GENERATOR[@]}" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build build/release --parallel
ctest --test-dir build/release --output-on-failure

echo "==> Benchmarks"
./build/release/bench_pricing \
  --benchmark_out=build/release/bench.json --benchmark_out_format=json
python3 scripts/summarize_bench.py build/release/bench.json | tee build/release/bench_summary.md
