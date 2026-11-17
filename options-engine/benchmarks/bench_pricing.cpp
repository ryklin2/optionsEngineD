// Google Benchmark suite for the optengine pricing engines.
//
//   BM_BSM_Scalar / BM_BSM_SIMD   price + 5 Greeks for chains of 1k, 100k, 1M options
//   BM_IV_Solver                  implied vol over a realistic listed-options chain
//   BM_MonteCarlo_GBM             1,000,000 paths x 252 steps on 1, 2, 4, 8 threads
//   BM_MonteCarlo_Heston          100,000 paths x 252 steps on 1, 2 threads
//
// Build in Release (-O3 -march=native -ffast-math) and run:
//   ./bench_pricing --benchmark_out=bench.json --benchmark_out_format=json
//   python3 scripts/summarize_bench.py bench.json
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "optengine/bsm_scalar.hpp"
#include "optengine/bsm_simd.hpp"
#include "optengine/iv_solver.hpp"
#include "optengine/monte_carlo.hpp"
#include "optengine/types.hpp"

namespace {

using namespace optengine;

/// Randomized chain: S in [50,150], K/S in [0.6,1.6], T in [0.02,3], vol in [5%,90%].
OptionChain make_random_chain(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> spot(50.0, 150.0);
  std::uniform_real_distribution<double> moneyness(0.6, 1.6);
  std::uniform_real_distribution<double> rate(0.0, 0.08);
  std::uniform_real_distribution<double> dividend(0.0, 0.05);
  std::uniform_real_distribution<double> expiry(0.02, 3.0);
  std::uniform_real_distribution<double> vol(0.05, 0.9);
  OptionChain chain(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double s = spot(gen);
    chain.set(i, {s, s * moneyness(gen), rate(gen), dividend(gen), expiry(gen), vol(gen),
                  (gen() & 1U) != 0});
  }
  return chain;
}

/// Listed-options style market: expiries from one week to two years, strikes
/// within +-3 standard deviations of the forward, skewed smile, OTM options.
OptionChain make_market_chain(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> spot(20.0, 500.0);
  std::uniform_real_distribution<double> expiry(7.0 / 365.0, 2.0);
  std::uniform_real_distribution<double> atm_vol(0.10, 0.60);
  std::uniform_real_distribution<double> std_devs(-3.0, 3.0);
  std::uniform_real_distribution<double> rate(0.0, 0.06);
  std::uniform_real_distribution<double> dividend(0.0, 0.03);
  OptionChain chain(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double s = spot(gen);
    const double t = expiry(gen);
    const double r = rate(gen);
    const double q = dividend(gen);
    const double base = atm_vol(gen);
    const double z = std_devs(gen);
    const double log_moneyness = z * base * std::sqrt(t);
    const double vol = base * (1.0 - 0.08 * z + 0.03 * z * z);
    chain.set(i, {s, s * std::exp((r - q) * t + log_moneyness), r, q, t, vol,
                  log_moneyness >= 0.0});
  }
  return chain;
}

void set_throughput(benchmark::State& state, double items_per_iteration, const char* counter) {
  const double items = static_cast<double>(state.iterations()) * items_per_iteration;
  state.SetItemsProcessed(static_cast<std::int64_t>(items));
  state.counters[counter] = benchmark::Counter(items / 1e6, benchmark::Counter::kIsRate);
}

// ---------------------------------------------------------------------------
// Analytical BSM: price + delta, gamma, vega, theta, rho
// ---------------------------------------------------------------------------
template <void (*Pricer)(ChainView, GreeksView) noexcept>
void run_bsm(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const OptionChain chain = make_random_chain(n, 12345);
  GreeksBuffer out(n);
  const ChainView in = chain.view();
  const GreeksView view = out.view();
  for (auto _ : state) {
    Pricer(in, view);
    benchmark::DoNotOptimize(out.price.data());
    benchmark::ClobberMemory();
  }
  set_throughput(state, static_cast<double>(n), "Moptions/s");
}

void BM_BSM_Scalar(benchmark::State& state) { run_bsm<&bsm_scalar_chain>(state); }
void BM_BSM_SIMD(benchmark::State& state) { run_bsm<&bsm_simd_chain>(state); }

BENCHMARK(BM_BSM_Scalar)->Arg(1'000)->Arg(100'000)->Arg(1'000'000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_BSM_SIMD)->Arg(1'000)->Arg(100'000)->Arg(1'000'000)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// Implied volatility
// ---------------------------------------------------------------------------
void run_iv(benchmark::State& state, const IvConfig& config) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const OptionChain chain = make_market_chain(n, 777);
  GreeksBuffer greeks(n);
  bsm_scalar_chain(chain.view(), greeks.view());  // "market" prices
  AlignedVector<double> iv(n);
  const ChainView in = chain.view();
  std::size_t converged = 0;
  for (auto _ : state) {
    converged = implied_volatility_chain(in, greeks.price, iv, {}, config);
    benchmark::DoNotOptimize(iv.data());
    benchmark::ClobberMemory();
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    worst = std::max(worst, std::fabs(iv[i] - chain.volatility[i]));
  }
  set_throughput(state, static_cast<double>(n), "Moptions/s");
  state.counters["converged_pct"] = 100.0 * static_cast<double>(converged) / static_cast<double>(n);
  state.counters["max_abs_err"] = worst;
}

void BM_IV_Solver(benchmark::State& state) { run_iv(state, IvConfig{}); }
void BM_IV_Solver_BrentOnly(benchmark::State& state) {
  IvConfig config;
  config.max_newton_iterations = 0;
  run_iv(state, config);
}

BENCHMARK(BM_IV_Solver)->Arg(1'000)->Arg(100'000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_IV_Solver_BrentOnly)->Arg(1'000)->Arg(100'000)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// Monte Carlo
// ---------------------------------------------------------------------------
constexpr std::size_t kMcSteps = 252;

void BM_MonteCarlo_GBM(benchmark::State& state) {
  constexpr std::size_t kPaths = 1'000'000;
  const GbmParams model{100.0, 0.05, 0.02, 0.20, 1.0};
  McWorkspace workspace(kPaths);
  McConfig config;
  config.num_paths = kPaths;
  config.num_steps = kMcSteps;
  config.num_threads = static_cast<unsigned>(state.range(0));
  config.antithetic = true;
  McResult result;
  for (auto _ : state) {
    result = mc_price_gbm(model, EuropeanCall{100.0}, config, workspace);
    benchmark::DoNotOptimize(result);
  }
  set_throughput(state, static_cast<double>(kPaths * kMcSteps), "Mpath_steps/s");
  const double exact = bsm_scalar_price({100.0, 100.0, 0.05, 0.02, 1.0, 0.20, true});
  state.counters["price"] = result.price;
  state.counters["std_error"] = result.std_error;
  state.counters["err_in_se"] = (result.price - exact) / result.std_error;
}

void BM_MonteCarlo_Heston(benchmark::State& state) {
  constexpr std::size_t kPaths = 100'000;
  const HestonParams model{100.0, 0.03, 0.01, 0.04, 2.0, 0.04, 0.3, -0.7, 1.0};
  McWorkspace workspace(kPaths);
  McConfig config;
  config.num_paths = kPaths;
  config.num_steps = kMcSteps;
  config.num_threads = static_cast<unsigned>(state.range(0));
  config.antithetic = true;
  McResult result;
  for (auto _ : state) {
    result = mc_price_heston(model, EuropeanCall{100.0}, config, workspace);
    benchmark::DoNotOptimize(result);
  }
  set_throughput(state, static_cast<double>(kPaths * kMcSteps), "Mpath_steps/s");
  state.counters["price"] = result.price;
  state.counters["std_error"] = result.std_error;
}

BENCHMARK(BM_MonteCarlo_GBM)
    ->ArgName("threads")->Arg(1)->Arg(2)->Arg(4)->Arg(8)
    ->Iterations(3)->UseRealTime()->MeasureProcessCPUTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MonteCarlo_Heston)
    ->ArgName("threads")->Arg(1)->Arg(2)
    ->Iterations(3)->UseRealTime()->MeasureProcessCPUTime()->Unit(benchmark::kMillisecond);

}  // namespace

int main(int argc, char** argv) {
  benchmark::AddCustomContext("optengine_simd_backend", std::string(bsm_simd_backend()));
  benchmark::AddCustomContext("hardware_threads",
                              std::to_string(std::thread::hardware_concurrency()));
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
