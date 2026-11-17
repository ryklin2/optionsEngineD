# Benchmark results

Produced by `scripts/run_all.sh` (GCC 13.3, `-O3 -march=native -ffast-math`, Google Benchmark 1.9.5)
on a 2-vCPU Intel Xeon @ 2.30 GHz cloud VM. The 4- and 8-thread Monte Carlo rows oversubscribe
this machine's two hardware threads, so they show no further scaling here.

## Environment

- host CPUs: 2 @ 2300 MHz, hardware threads: 2
- SIMD backend: avx2+fma
- build type: release

## Analytical BSM (price + delta, gamma, vega, theta, rho)

| Contracts | Scalar (M options/s) | SIMD (M options/s) | Speedup |
|---:|---:|---:|---:|
| 1,000 | 26.2 | 75.9 | 2.89x |
| 100,000 | 21.8 | 64.3 | 2.94x |
| 1,000,000 | 19.9 | 67.2 | 3.38x |

## Implied volatility (realistic OTM market chain)

| Contracts | Solver | M options/s | ns / option | Converged | Max abs error |
|---:|---|---:|---:|---:|---:|
| 1,000 | Newton + Brent fallback | 2.95 | 338 | 100.0% | 2.6e-15 |
| 100,000 | Newton + Brent fallback | 2.91 | 344 | 100.0% | 4.8e-15 |
| 1,000 | Brent only | 1.69 | 593 | 100.0% | 1.8e-15 |
| 100,000 | Brent only | 1.62 | 616 | 100.0% | 4.1e-15 |

## Monte Carlo GBM (1,000,000 paths x 252 steps, antithetic)

| Threads | Wall time (ms) | M path-steps/s | Speedup vs 1 thread | Price | Std error |
|---:|---:|---:|---:|---:|---:|
| 1 | 927 | 272 | 1.00x | 9.24542 | 0.01031 |
| 2 | 464 | 543 | 2.00x | 9.24542 | 0.01031 |
| 4 | 451 | 558 | 2.06x | 9.24542 | 0.01031 |
| 8 | 456 | 552 | 2.03x | 9.24542 | 0.01031 |

## Monte Carlo Heston (100,000 paths x 252 steps, antithetic)

| Threads | Wall time (ms) | M path-steps/s | Speedup vs 1 thread | Price | Std error |
|---:|---:|---:|---:|---:|---:|
| 1 | 242 | 104 | 1.00x | 8.61384 | 0.02347 |
| 2 | 114 | 220 | 2.11x | 8.61384 | 0.02347 |
