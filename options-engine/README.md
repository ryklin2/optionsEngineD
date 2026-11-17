# optengine — vectorized options pricing & Greeks engine (C++20)

A deterministic, allocation-free quantitative pricing library with three engines:

| Engine | Header | What it does |
|---|---|---|
| Analytical BSM | `bsm_scalar.hpp`, `bsm_simd.hpp` | Price + delta, gamma, vega, theta, rho. Scalar reference and an AVX2+FMA kernel (4 options per iteration) over SoA chains, with a scalar fallback when AVX2/FMA are not compiled in. |
| Implied volatility | `iv_solver.hpp` | Newton-Raphson on a transformed objective with a bracketing Brent fallback. Never returns NaN; reports a status instead. |
| Monte Carlo | `monte_carlo.hpp` | GBM and Heston path simulators, PCG64 per worker thread, antithetic variates, `std::jthread` workers, payoffs via a C++20 concept (European, arithmetic/geometric Asian, up-and-out barrier). |

## Build, test, benchmark

Requirements: Linux x86-64, CMake >= 3.20, GCC or Clang with C++20. GoogleTest 1.17.0 and
Google Benchmark 1.9.5 are fetched by CMake (`FetchContent`).

```bash
scripts/run_all.sh          # Debug+sanitizers tests, Release tests, benchmarks, summary table
```

or by hand:

```bash
# Tests: -O0 -g -fsanitize=address,undefined
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug -j
ctest --test-dir build/debug --output-on-failure

# Benchmarks: -O3 -march=native -ffast-math
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release -j
./build/release/bench_pricing --benchmark_out=bench.json --benchmark_out_format=json
python3 scripts/summarize_bench.py bench.json
```

CMake options:

| Option | Default | Effect |
|---|---|---|
| `OPTENGINE_NATIVE_ARCH` | `ON` | Adds `-march=native`. Turn `OFF` to build and test the scalar fallback of `bsm_simd`. |
| `OPTENGINE_SANITIZE` | `ON` | ASan + UBSan in Debug builds. |
| `OPTENGINE_BUILD_TESTS` / `OPTENGINE_BUILD_BENCHMARKS` | `ON` | Build the test / benchmark targets. |

## Layout

```text
options-engine/
├── CMakeLists.txt
├── include/optengine/
│   ├── types.hpp          SoA OptionChain / GreeksBuffer (64-byte aligned arrays), views
│   ├── math_utils.hpp     Phi/phi, inverse normal CDF, AVX2 exp / log / Phi
│   ├── bsm_scalar.hpp     Reference scalar pricer & Greeks
│   ├── bsm_simd.hpp       AVX2 kernel + chain driver
│   ├── iv_solver.hpp      Newton + Brent implied volatility
│   └── monte_carlo.hpp    PCG64, payoffs, GBM & Heston simulators
├── src/                   Chain drivers, IV solver, thread dispatch
├── tests/
│   ├── test_bsm.cpp       Phi accuracy, Hull values, put-call parity, scalar vs SIMD
│   ├── test_iv.cpp        Round-trip stress grid, bounds, Brent path
│   ├── test_mc.cpp        PCG64 known answers, convergence, variance reduction, Heston
│   ├── alloc_hook.{hpp,cpp}  Global operator new/delete replacement + counter
│   └── test_util.hpp      Chain generators
├── benchmarks/bench_pricing.cpp
└── scripts/               run_all.sh, summarize_bench.py
```

## Quick example

```cpp
#include "optengine/bsm_simd.hpp"
#include "optengine/iv_solver.hpp"
#include "optengine/monte_carlo.hpp"
using namespace optengine;

OptionChain chain(3);                     // 64-byte aligned SoA storage
chain.set(0, {100.0, 100.0, 0.05, 0.02, 1.0, 0.20, true});
chain.set(1, {100.0, 110.0, 0.05, 0.02, 0.5, 0.25, true});
chain.set(2, {100.0,  90.0, 0.05, 0.02, 0.5, 0.30, false});
GreeksBuffer out(chain.size());
bsm_simd_chain(chain.view(), out.view()); // no allocation

IvResult iv = implied_volatility(out.price[0], chain.get(0));   // -> 0.20

McWorkspace scratch(1'000'000);           // allocated once, up front
McConfig cfg{.num_paths = 1'000'000, .num_steps = 252, .num_threads = 4, .seed = 42};
McResult mc = mc_price_gbm(GbmParams{100.0, 0.05, 0.02, 0.20, 1.0},
                           ArithmeticAsianCall{100.0}, cfg, scratch);
```

## Conventions and units

- Time in years; rates, dividend yield and volatility annualized (0.20 = 20%).
- `vega` and `rho` are per 1.00 change (100 vol points / 100% rate); divide by 100 for per-1%.
- `theta` is per **year**; `theta_per_calendar_day()` divides by 365.
- Preconditions for the pricing kernels: S, K, T, sigma > 0. They do not special-case expiry.
- Heston `v0` and `theta` are variances (0.04 = 20% vol); `xi` is vol-of-vol.

## Design notes

**Normal CDF.** `norm_dist(x)` returns Phi(x), 1 - Phi(x) and phi(x) from one `exp` and one rational
polynomial, chosen from three regions (Hart for |x| < 0.75, Cody's two erfc fits beyond). The rounding
error of x² is recovered with one FMA. Measured against 40-digit references: max absolute error
2.2e-16, and the *relative* error of the tail stays near 1e-15 out to |x| ≈ 37. Hart's formula alone
meets the 1e-15 absolute target but loses relative accuracy in the tail (3e-9 near |x| = 7), which
matters for deep out-of-the-money prices and for the IV solver. The AVX2 version evaluates all three
regions and blends: no branches, one division.

**AVX2 kernel.** Aligned loads/stores on `OptionChain` / `GreeksBuffer`; unaligned spans are detected
and handled. Calls and puts are both computed and blended by mask. A tail of `n % 4` options goes
through the same vector kernel on a padded stack buffer, so remainder elements get identical
arithmetic and nothing is read or written out of bounds. `exp` and `log` are in-house FMA
implementations (< 2 ulp against libm).

**Implied volatility.** Every option is mapped by put-call parity to the out-of-the-money option with
the same time value, normalized to `b(s) = beta` with `s = sigma * sqrt(T)`. Newton runs on
`-1/ln b` below the inflection point and on `ln(b_max - b)` above it; both are close to
quadratic/linear in `s`, so convergence takes 4-5 iterations even where vega has underflowed. A
bracket is maintained throughout; if a derivative is unusable or a step leaves the bracket, Brent's
method finishes on the bracket.

**Monte Carlo.** Paths are processed in fixed blocks of 2,048 samples. Each block re-seeds the worker's
thread-local PCG64 from (seed, block index) and block sums are reduced in block order, so a given
seed produces a **bit-identical result for any thread count**. Normals come from inverting the CDF
(Wichura AS241). Heston uses full-truncation Euler in log-spot. Terminal-only payoffs accumulate
Brownian increments and exponentiate once; path-dependent payoffs observe every step.

**Allocation.** Pricing functions take spans and caller-owned scratch. The test binary replaces global
`operator new` with a counting version and asserts zero allocations inside `bsm_simd_chain`,
`bsm_scalar_chain`, the IV solver (including its Brent and error paths) and single-threaded Monte
Carlo. With `num_threads > 1` the only allocations are those made by the OS thread launch.

## Verification

46 tests; all pass with GCC 13 in Debug (`-O0`, ASan + UBSan), Release (`-O3 -march=native
-ffast-math`) and Debug without `-march=native` (scalar fallback, 43 tests), and with Clang 18 in
Release and `-O0`.

| Requirement | How it is tested |
|---|---|
| Textbook values, 1e-12 scalar / 1e-11 SIMD | Four worked examples from Hull, calls and puts, price + 5 Greeks. Hull prints 2-4 significant figures, so the same contracts were re-priced in 50-digit arithmetic (mpmath) to get references good to 1e-12; the printed figures (4.76, 0.81, delta 0.522, ...) are checked separately. |
| Put-call parity | 10,000 random contracts, scalar and SIMD, including the differentiated parity relations for every Greek. |
| Scalar vs SIMD | N = 1, 2, 3, 4, 5, 7, 8, 9, 1023, 1024, 10000, element by element within 5e-15 (scaled); sentinel test that slots past N are untouched; unaligned spans. |
| IV round trip | sigma 1%..300% × S/K 0.5..2.0 × 8 expiries × call/put = 292,800 solves, with default settings and with Brent forced. |
| MC convergence | European calls/puts within 3 standard errors of BSM (1 to 52 steps, crude and antithetic); antithetic standard error < 0.85× crude at equal path count; geometric Asian against its closed form; Heston against a characteristic-function reference. |
| Zero allocation | `operator new` hook, see above. |

### Things worth knowing

- **IV round trip at the edges.** 93.6% of the stress grid is recovered to 1e-8 (worst error 8e-10).
  The rest cannot be recovered by any double-precision solver, and the test asserts the honest
  behaviour instead: 0.7% are out-of-the-money prices that underflow below 1e-250 (e.g. 1% vol,
  S/K = 0.5), and 5.7% are deep in-the-money options whose time value is smaller than the rounding
  error of the price itself. For those the solver must terminate, stay finite and either report
  `kBelowIntrinsic` or land within the precision the price allows.
- **Scalar vs SIMD is not bit-for-bit.** The SIMD path uses its own `exp`/`log`; the two agree to a few
  ulp (measured 3e-16 scaled over 200,000 contracts).
- **`-ffast-math`.** The Release build is tested, not just benchmarked. The library avoids relying on
  NaN/infinity semantics (bit-level `is_finite`, finite sentinels in the solver).
- **The scalar benchmark baseline is genuinely scalar**: auto-vectorization is disabled for that one
  loop, otherwise `-O3 -ffast-math` may vectorize it and the comparison would be meaningless.
- **Heston discretization bias.** Full-truncation Euler has O(dt) bias; the reference test allows 0.02
  on top of 3 standard errors.

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for the full table from the development machine (2 vCPUs).
Headline: the AVX2 kernel prices 64-76 million options per second with all five Greeks, 2.9-3.4×
the scalar reference; the IV solver handles about 2.9 million options per second; GBM Monte Carlo
runs 1,000,000 paths × 252 steps in 0.93 s on one thread and 0.46 s on two.
