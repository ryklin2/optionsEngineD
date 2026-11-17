// Monte Carlo engine tests: RNG correctness, convergence to analytical prices
// within standard-error bounds, variance reduction, determinism.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "alloc_hook.hpp"
#include "optengine/bsm_scalar.hpp"
#include "optengine/math_utils.hpp"
#include "optengine/monte_carlo.hpp"

namespace optengine {
namespace {

using testing::AllocationCounter;

[[nodiscard]] double bsm(const GbmParams& m, double strike, bool is_call) {
  return bsm_scalar_price(
      {m.spot, strike, m.rate, m.dividend, m.time_to_expiry, m.volatility, is_call});
}

// ---------------------------------------------------------------------------
// Random numbers
// ---------------------------------------------------------------------------
TEST(Pcg64, MatchesReferenceImplementation) {
  // Known-answer vector from NumPy's PCG64 bit generator with the state and
  // increment produced by the standard PCG seeding of
  //   initstate = 0x0123456789ABCDEFFEDCBA9876543210, initseq = 0xDA3E39CB94B95BDB.
  const Pcg64::Uint128 init_state =
      (static_cast<Pcg64::Uint128>(0x0123456789ABCDEFULL) << 64U) | 0xFEDCBA9876543210ULL;
  Pcg64 rng(init_state, 0xDA3E39CB94B95BDBULL);
  for (std::uint64_t expected :
       {0x9ede70827c402be8ULL, 0x927bd6b6f163bb27ULL, 0xa727e00369c31417ULL,
        0x15e90fde605a4715ULL, 0x0de02d6df2f186e7ULL, 0x08eca932b96bb68cULL}) {
    EXPECT_EQ(rng.next(), expected);
  }
}

TEST(Pcg64, StreamsAreReproducibleAndDistinct) {
  Pcg64 a;
  Pcg64 b;
  Pcg64 c;
  a.seed_stream(42, 7);
  b.seed_stream(42, 7);
  c.seed_stream(42, 8);
  int equal_ab = 0;
  int equal_ac = 0;
  for (int i = 0; i < 1000; ++i) {
    const std::uint64_t x = a.next();
    equal_ab += x == b.next() ? 1 : 0;
    equal_ac += x == c.next() ? 1 : 0;
  }
  EXPECT_EQ(equal_ab, 1000);
  EXPECT_EQ(equal_ac, 0);
}

TEST(Pcg64, NormalVariatesHaveStandardMoments) {
  constexpr int kN = 400000;
  Pcg64 rng;
  rng.seed_stream(2024, 0);
  double m1 = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0;
  int beyond_3 = 0;
  for (int i = 0; i < kN; ++i) {
    const double u = rng.next_uniform();
    ASSERT_GT(u, 0.0);
    ASSERT_LT(u, 1.0);
    const double z = inv_norm_cdf(u);
    m1 += z;
    m2 += z * z;
    m3 += z * z * z;
    m4 += z * z * z * z;
    beyond_3 += std::fabs(z) > 3.0 ? 1 : 0;
  }
  const double n = kN;
  // Tolerances are 4 standard errors of each sample moment.
  EXPECT_NEAR(m1 / n, 0.0, 4.0 / std::sqrt(n));
  EXPECT_NEAR(m2 / n, 1.0, 4.0 * std::sqrt(2.0 / n));
  EXPECT_NEAR(m3 / n, 0.0, 4.0 * std::sqrt(15.0 / n));
  EXPECT_NEAR(m4 / n, 3.0, 4.0 * std::sqrt(96.0 / n));
  const double p3 = 2.0 * norm_dist(-3.0).cdf;  // P(|Z| > 3) = 0.0027
  EXPECT_NEAR(beyond_3 / n, p3, 4.0 * std::sqrt(p3 / n));
}

// ---------------------------------------------------------------------------
// GBM convergence to Black-Scholes-Merton
// ---------------------------------------------------------------------------
struct ConvergenceCase {
  const char* name;
  GbmParams model;
  double strike;
  bool is_call;
  std::size_t steps;
};

TEST(MonteCarloGbm, EuropeanPricesConvergeWithin3StandardErrors) {
  const ConvergenceCase cases[] = {
      {"ATM call, 1 step", {100.0, 0.05, 0.02, 0.20, 1.0}, 100.0, true, 1},
      {"ATM put, 1 step", {100.0, 0.05, 0.02, 0.20, 1.0}, 100.0, false, 1},
      {"OTM call, 1 step", {100.0, 0.03, 0.00, 0.35, 0.5}, 125.0, true, 1},
      {"ITM call, 1 step", {100.0, 0.03, 0.00, 0.35, 0.5}, 80.0, true, 1},
      {"OTM put, 1 step", {100.0, 0.01, 0.04, 0.15, 2.0}, 85.0, false, 1},
      {"ATM call, 16 steps", {100.0, 0.05, 0.02, 0.20, 1.0}, 100.0, true, 16},
      {"OTM put, 52 steps", {50.0, 0.04, 0.01, 0.45, 1.0}, 40.0, false, 52},
  };
  McWorkspace workspace(400'000);
  std::uint64_t seed = 1000;
  for (const ConvergenceCase& c : cases) {
    for (bool antithetic : {true, false}) {
      McConfig config;
      config.num_paths = c.steps == 1 ? 400'000 : 60'000;
      config.num_steps = c.steps;
      config.num_threads = 2;
      config.seed = ++seed;
      config.antithetic = antithetic;
      const McResult r = mc_european_gbm(c.model, c.strike, c.is_call, config, workspace);
      const double exact = bsm(c.model, c.strike, c.is_call);
      ASSERT_EQ(r.status, McStatus::kOk) << c.name;
      EXPECT_EQ(r.num_paths, config.num_paths);
      EXPECT_EQ(r.num_samples, antithetic ? config.num_paths / 2 : config.num_paths);
      EXPECT_GT(r.std_error, 0.0);
      EXPECT_LT(r.std_error, 0.02 * c.model.spot) << c.name;
      EXPECT_LE(std::fabs(r.price - exact), 3.0 * r.std_error)
          << c.name << (antithetic ? " [antithetic]" : " [crude]") << ": mc = " << r.price
          << ", bsm = " << exact << ", se = " << r.std_error << " ("
          << (r.price - exact) / r.std_error << " sigma)";
    }
  }
}

TEST(MonteCarloGbm, StandardErrorShrinksAsInverseSqrtN) {
  const GbmParams model{100.0, 0.05, 0.0, 0.25, 1.0};
  McWorkspace workspace(800'000);
  McConfig config;
  config.antithetic = false;
  config.num_threads = 2;
  config.num_paths = 50'000;
  const McResult small = mc_european_gbm(model, 100.0, true, config, workspace);
  config.num_paths = 800'000;
  const McResult large = mc_european_gbm(model, 100.0, true, config, workspace);
  // 16x the paths => 4x smaller standard error (within sampling noise).
  EXPECT_NEAR(small.std_error / large.std_error, 4.0, 0.15);
  const double exact = bsm(model, 100.0, true);
  EXPECT_LE(std::fabs(large.price - exact), 3.0 * large.std_error);
  EXPECT_LE(std::fabs(small.price - exact), 3.0 * small.std_error);
}

TEST(MonteCarloGbm, AntitheticVariatesReduceStandardError) {
  McWorkspace workspace(200'000);
  for (const auto& [strike, is_call] :
       {std::pair{100.0, true}, std::pair{100.0, false}, std::pair{90.0, true},
        std::pair{110.0, false}}) {
    const GbmParams model{100.0, 0.05, 0.02, 0.20, 1.0};
    McConfig config;
    config.num_paths = 200'000;  // identical path budget for both estimators
    config.seed = 77;
    config.antithetic = false;
    const McResult crude = mc_european_gbm(model, strike, is_call, config, workspace);
    config.antithetic = true;
    const McResult anti = mc_european_gbm(model, strike, is_call, config, workspace);
    const double exact = bsm(model, strike, is_call);

    EXPECT_LT(anti.std_error, 0.85 * crude.std_error)
        << "K = " << strike << (is_call ? " call" : " put") << ": crude se = "
        << crude.std_error << ", antithetic se = " << anti.std_error;
    EXPECT_LE(std::fabs(crude.price - exact), 3.0 * crude.std_error);
    EXPECT_LE(std::fabs(anti.price - exact), 3.0 * anti.std_error);
  }
}

TEST(MonteCarloGbm, ResultIsBitIdenticalAcrossThreadCounts) {
  const GbmParams model{100.0, 0.03, 0.01, 0.30, 0.75};
  McWorkspace workspace(100'000);
  McConfig config;
  config.num_paths = 100'000;
  config.num_steps = 8;
  config.seed = 5;
  config.num_threads = 1;
  const McResult reference = mc_price_gbm(model, EuropeanCall{105.0}, config, workspace);
  ASSERT_EQ(reference.status, McStatus::kOk);
  for (unsigned threads : {2U, 3U, 4U, 8U}) {
    config.num_threads = threads;
    const McResult r = mc_price_gbm(model, EuropeanCall{105.0}, config, workspace);
    EXPECT_EQ(r.price, reference.price) << threads << " threads";
    EXPECT_EQ(r.std_error, reference.std_error) << threads << " threads";
  }
  // Same seed reproduces, a different seed does not.
  config.num_threads = 2;
  EXPECT_EQ(mc_price_gbm(model, EuropeanCall{105.0}, config, workspace).price, reference.price);
  config.seed = 6;
  EXPECT_NE(mc_price_gbm(model, EuropeanCall{105.0}, config, workspace).price, reference.price);
}

TEST(MonteCarloGbm, RejectsBadConfigurationWithoutSimulating) {
  const GbmParams model{100.0, 0.03, 0.0, 0.2, 1.0};
  McWorkspace small(1000);
  McConfig config;
  config.num_paths = 1'000'000;
  EXPECT_EQ(mc_european_gbm(model, 100.0, true, config, small).status,
            McStatus::kWorkspaceTooSmall);
  McWorkspace workspace(10'000);
  config.num_paths = 10'000;
  config.num_steps = 0;
  EXPECT_EQ(mc_european_gbm(model, 100.0, true, config, workspace).status,
            McStatus::kInvalidConfig);
  config.num_steps = 1;
  config.num_paths = 2;  // a single antithetic pair has no variance estimate
  EXPECT_EQ(mc_european_gbm(model, 100.0, true, config, workspace).status,
            McStatus::kInvalidConfig);
  config.num_paths = 10'000;
  config.num_threads = 0;  // clamped to one worker
  EXPECT_EQ(mc_european_gbm(model, 100.0, true, config, workspace).status, McStatus::kOk);
}

// ---------------------------------------------------------------------------
// Path-dependent payoffs (static polymorphism)
// ---------------------------------------------------------------------------
static_assert(PathPayoff<EuropeanCall> && PathPayoff<EuropeanPut>);
static_assert(PathPayoff<ArithmeticAsianCall> && PathPayoff<GeometricAsianCall>);
static_assert(PathPayoff<UpAndOutCall>);

/// Closed form for a discretely monitored geometric-average call under GBM.
/// ln G is normal with
///   mean = ln S0 + (r - q - v^2/2) dt (n+1)/2,  var = v^2 dt (n+1)(2n+1)/(6n).
[[nodiscard]] double geometric_asian_call(const GbmParams& m, double strike, std::size_t steps) {
  const double n = static_cast<double>(steps);
  const double dt = m.time_to_expiry / n;
  const double mean = std::log(m.spot) +
                      (m.rate - m.dividend - 0.5 * m.volatility * m.volatility) * dt * (n + 1) / 2;
  const double var = m.volatility * m.volatility * dt * (n + 1) * (2 * n + 1) / (6 * n);
  const double sd = std::sqrt(var);
  const double d2 = (mean - std::log(strike)) / sd;
  const double d1 = d2 + sd;
  return std::exp(-m.rate * m.time_to_expiry) *
         (std::exp(mean + 0.5 * var) * norm_cdf(d1) - strike * norm_cdf(d2));
}

TEST(MonteCarloGbm, GeometricAsianMatchesClosedForm) {
  const GbmParams model{100.0, 0.05, 0.01, 0.30, 1.0};
  McWorkspace workspace(80'000);
  for (std::size_t steps : {std::size_t{4}, std::size_t{12}, std::size_t{50}}) {
    for (bool antithetic : {true, false}) {
      McConfig config;
      config.num_paths = 80'000;
      config.num_steps = steps;
      config.num_threads = 2;
      config.seed = 900 + steps;
      config.antithetic = antithetic;
      const McResult r = mc_price_gbm(model, GeometricAsianCall{100.0}, config, workspace);
      const double exact = geometric_asian_call(model, 100.0, steps);
      EXPECT_LE(std::fabs(r.price - exact), 3.0 * r.std_error)
          << steps << " steps: mc = " << r.price << ", exact = " << exact
          << ", se = " << r.std_error;
    }
  }
}

TEST(MonteCarloGbm, PathDependentPayoffOrdering) {
  const GbmParams model{100.0, 0.05, 0.0, 0.25, 1.0};
  McWorkspace workspace(60'000);
  McConfig config;
  config.num_paths = 60'000;
  config.num_steps = 24;
  config.num_threads = 2;
  config.seed = 31;

  const McResult european = mc_price_gbm(model, EuropeanCall{100.0}, config, workspace);
  const McResult arithmetic = mc_price_gbm(model, ArithmeticAsianCall{100.0}, config, workspace);
  const McResult geometric = mc_price_gbm(model, GeometricAsianCall{100.0}, config, workspace);
  // Same seed => same paths, and AM >= GM holds path by path.
  EXPECT_GT(arithmetic.price, geometric.price);
  // Averaging removes variance: an Asian call is worth less than the European.
  EXPECT_LT(arithmetic.price, european.price - 5.0 * european.std_error);

  // Barrier far away: identical to the vanilla call up to rounding.
  const McResult no_barrier = mc_price_gbm(model, UpAndOutCall{100.0, 1e9}, config, workspace);
  EXPECT_NEAR(no_barrier.price, european.price, 1e-9 * european.price);
  // Tighter barriers knock out more paths.
  const McResult wide = mc_price_gbm(model, UpAndOutCall{100.0, 150.0}, config, workspace);
  const McResult tight = mc_price_gbm(model, UpAndOutCall{100.0, 120.0}, config, workspace);
  EXPECT_LT(wide.price, no_barrier.price);
  EXPECT_LT(tight.price, wide.price);
  EXPECT_GT(tight.price, 0.0);
}

// ---------------------------------------------------------------------------
// Heston
// ---------------------------------------------------------------------------
TEST(MonteCarloHeston, DegeneratesToBlackScholesWithoutVolOfVol) {
  // xi = 0 and v0 = theta: variance stays at 0.04, i.e. GBM with 20% vol.
  const HestonParams heston{100.0, 0.04, 0.01, 0.04, 1.5, 0.04, 0.0, -0.6, 1.0};
  const GbmParams gbm{100.0, 0.04, 0.01, 0.20, 1.0};
  McWorkspace workspace(100'000);
  McConfig config;
  config.num_paths = 100'000;
  config.num_steps = 12;
  config.num_threads = 2;
  for (const auto& [strike, is_call] :
       {std::pair{100.0, true}, std::pair{115.0, true}, std::pair{95.0, false}}) {
    const McResult r = mc_european_heston(heston, strike, is_call, config, workspace);
    ASSERT_EQ(r.status, McStatus::kOk);
    EXPECT_LE(std::fabs(r.price - bsm(gbm, strike, is_call)), 3.0 * r.std_error)
        << "K = " << strike << ": mc = " << r.price << ", bsm = " << bsm(gbm, strike, is_call);
  }
}

TEST(MonteCarloHeston, MatchesSemiAnalyticalReference) {
  // Reference call prices from the Heston characteristic function
  // (Albrecher et al. formulation, adaptive quadrature, abs. error < 1e-7).
  // The same code reproduces Broadie & Kaya's published 6.8061.
  const HestonParams model{100.0, 0.03, 0.01, 0.04, 2.0, 0.04, 0.3, -0.7, 1.0};
  struct Ref {
    double strike;
    double price;
  };
  const Ref refs[] = {{90.0, 14.945071995558948}, {100.0, 8.599588101520503},
                      {110.0, 4.14834352981709}};
  McWorkspace workspace(120'000);
  McConfig config;
  config.num_paths = 120'000;
  config.num_steps = 100;
  config.num_threads = 2;
  config.seed = 2718;
  // Full-truncation Euler carries an O(dt) bias; with dt = 0.01 and the Feller
  // condition satisfied it is well below 0.02 for these contracts.
  constexpr double kDiscretizationAllowance = 0.02;
  for (const Ref& ref : refs) {
    const McResult call = mc_european_heston(model, ref.strike, true, config, workspace);
    EXPECT_LE(std::fabs(call.price - ref.price), 3.0 * call.std_error + kDiscretizationAllowance)
        << "K = " << ref.strike << ": mc = " << call.price << ", ref = " << ref.price
        << ", se = " << call.std_error;
    EXPECT_LT(call.std_error, 0.05);

    // Put-call parity must hold for the simulated measure as well.
    const McResult put = mc_european_heston(model, ref.strike, false, config, workspace);
    const double forward_value = model.spot * std::exp(-model.dividend * model.time_to_expiry) -
                                 ref.strike * std::exp(-model.rate * model.time_to_expiry);
    EXPECT_LE(std::fabs(call.price - put.price - forward_value),
              3.0 * (call.std_error + put.std_error) + kDiscretizationAllowance);
  }
}

TEST(MonteCarloHeston, NegativeCorrelationProducesDownsideSkew) {
  // With rho < 0 the OTM put is dearer and the OTM call cheaper than under
  // Black-Scholes at the same average variance.
  const HestonParams model{100.0, 0.0, 0.0, 0.04, 2.0, 0.04, 0.5, -0.8, 1.0};
  const GbmParams flat{100.0, 0.0, 0.0, 0.20, 1.0};
  McWorkspace workspace(100'000);
  McConfig config;
  config.num_paths = 100'000;
  config.num_steps = 100;
  config.num_threads = 2;
  const McResult put = mc_european_heston(model, 75.0, false, config, workspace);
  const McResult call = mc_european_heston(model, 125.0, true, config, workspace);
  EXPECT_GT(put.price, bsm(flat, 75.0, false) + 5.0 * put.std_error);
  EXPECT_LT(call.price, bsm(flat, 125.0, true) - 5.0 * call.std_error);
}

TEST(MonteCarloHeston, DeterministicAcrossThreadCounts) {
  const HestonParams model{100.0, 0.02, 0.0, 0.05, 1.0, 0.06, 0.4, -0.5, 0.5};
  McWorkspace workspace(40'000);
  McConfig config;
  config.num_paths = 40'000;
  config.num_steps = 20;
  config.num_threads = 1;
  const McResult reference = mc_price_heston(model, ArithmeticAsianCall{100.0}, config, workspace);
  config.num_threads = 4;
  const McResult threaded = mc_price_heston(model, ArithmeticAsianCall{100.0}, config, workspace);
  EXPECT_EQ(threaded.price, reference.price);
  EXPECT_EQ(threaded.std_error, reference.std_error);
  EXPECT_GT(reference.price, 0.0);
}

// ---------------------------------------------------------------------------
// Zero-allocation enforcement
// ---------------------------------------------------------------------------
TEST(ZeroAllocation, MonteCarloPathLoopsDoNotAllocate) {
  const GbmParams gbm{100.0, 0.05, 0.0, 0.2, 1.0};
  const HestonParams heston{100.0, 0.03, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7, 1.0};
  McWorkspace workspace(20'000);  // the only allocation, made up front
  McConfig config;
  config.num_paths = 20'000;
  config.num_steps = 8;
  config.num_threads = 1;  // threads > 1 allocate inside the OS thread launch only
  double sink = 0.0;
  {
    const AllocationCounter counter;
    sink += mc_price_gbm(gbm, EuropeanCall{100.0}, config, workspace).price;
    sink += mc_price_gbm(gbm, ArithmeticAsianCall{100.0}, config, workspace).price;
    sink += mc_price_gbm(gbm, UpAndOutCall{100.0, 130.0}, config, workspace).price;
    sink += mc_price_heston(heston, EuropeanPut{100.0}, config, workspace).price;
    Pcg64 rng;
    rng.seed_stream(1, 2);
    sink += simulate_gbm_block(gbm, GeometricAsianCall{100.0}, 500, 8, true, rng).sum;
    EXPECT_EQ(counter.allocations(), 0U) << "a Monte Carlo path loop allocated on the heap";
  }
  EXPECT_GT(sink, 0.0);
}

}  // namespace
}  // namespace optengine
