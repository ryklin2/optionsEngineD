// Implied-volatility round-trip and robustness tests.
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "alloc_hook.hpp"
#include "optengine/bsm_scalar.hpp"
#include "optengine/iv_solver.hpp"
#include "optengine/math_utils.hpp"
#include "test_util.hpp"

namespace optengine {
namespace {

using testing::AllocationCounter;

constexpr double kIvTolerance = 1e-8;
constexpr double kEps = std::numeric_limits<double>::epsilon();

/// Prices below this carry (almost) no significant bits: exp(-d^2/2) is in or
/// next to the denormal range, so no solver can recover sigma from them.
constexpr double kMinInformativePrice = 1e-250;

[[nodiscard]] bool is_out_of_the_money(const OptionParams& o) {
  const double spot_disc = o.spot * std::exp(-o.dividend * o.time_to_expiry);
  const double strike_disc = o.strike * std::exp(-o.rate * o.time_to_expiry);
  return o.is_call ? spot_disc <= strike_disc : spot_disc >= strike_disc;
}

struct GridStats {
  long total = 0;
  long recovered = 0;          // converged within 1e-8
  long underflow = 0;          // OTM price below double resolution
  long itm_ill_conditioned = 0;  // ITM time value below the price's rounding error
  long used_brent = 0;
  int max_newton = 0;
  int max_brent = 0;
  double worst_error = 0.0;
};

/// Round-trips sigma -> BSM price -> implied vol over the full stress grid:
///   sigma in [0.01, 3.00] step 0.01, S/K in [0.5, 2.0] step 0.025,
///   T from one day to five years, calls and puts.
GridStats run_stress_grid(const IvConfig& config) {
  GridStats stats;
  for (double t : {1.0 / 365.0, 7.0 / 365.0, 1.0 / 12.0, 0.25, 0.5, 1.0, 2.0, 5.0}) {
    for (int iv = 1; iv <= 300; ++iv) {
      const double vol = 0.01 * iv;
      for (int im = 0; im <= 60; ++im) {
        const double moneyness = 0.5 + 0.025 * im;
        for (bool is_call : {true, false}) {
          const OptionParams o{100.0 * moneyness, 100.0, 0.03, 0.01, t, vol, is_call};
          const Greeks g = bsm_scalar_greeks(o);
          const IvResult r = implied_volatility(g.price, o, config);
          ++stats.total;
          const auto where = [&] {
            return ::testing::Message() << "T=" << t << " vol=" << vol << " S/K=" << moneyness
                                        << (is_call ? " call" : " put") << " price=" << g.price
                                        << " status=" << to_string(r.status)
                                        << " iv=" << r.implied_vol;
          };

          // Unconditional guarantees: finite, non-negative, bounded work.
          EXPECT_TRUE(is_finite(r.implied_vol)) << where();
          EXPECT_GE(r.implied_vol, 0.0) << where();
          EXPECT_LE(r.newton_iterations, config.max_newton_iterations) << where();
          EXPECT_LE(r.brent_iterations, config.max_brent_iterations) << where();
          stats.max_newton = std::max<int>(stats.max_newton, r.newton_iterations);
          stats.max_brent = std::max<int>(stats.max_brent, r.brent_iterations);
          stats.used_brent += r.used_brent ? 1 : 0;

          const double error = std::fabs(r.implied_vol - vol);
          if (is_out_of_the_money(o)) {
            if (g.price < kMinInformativePrice) {
              // The price itself underflowed; the solver must say so or
              // return a finite best effort, never NaN / hang.
              ++stats.underflow;
              continue;
            }
            EXPECT_EQ(r.status, IvStatus::kConverged) << where();
            EXPECT_LE(error, kIvTolerance) << where();
            stats.worst_error = std::max(stats.worst_error, error);
            ++stats.recovered;
          } else {
            // ITM: price = intrinsic + time value. One ulp of the price moves
            // sigma by ulp / vega; that is the best any solver can do.
            const double resolution = 8.0 * kEps * g.price / std::max(g.vega, 1e-300);
            if (resolution < 0.1 * kIvTolerance) {
              EXPECT_EQ(r.status, IvStatus::kConverged) << where();
              EXPECT_LE(error, kIvTolerance) << where();
              stats.worst_error = std::max(stats.worst_error, error);
              ++stats.recovered;
            } else {
              ++stats.itm_ill_conditioned;
              if (r.status == IvStatus::kConverged) {
                EXPECT_LE(error, std::max(kIvTolerance, 100.0 * resolution)) << where();
              } else {
                EXPECT_TRUE(r.status == IvStatus::kBelowIntrinsic ||
                            r.status == IvStatus::kMaxIterations)
                    << where();
              }
            }
          }
        }
      }
    }
  }
  return stats;
}

TEST(IvSolver, StressGridRoundTripWithin1e8) {
  const GridStats s = run_stress_grid(IvConfig{});
  std::printf(
      "[ iv grid ] %ld solves: %ld recovered to 1e-8 (worst error %.2e), %ld OTM prices "
      "underflowed, %ld ITM beyond price resolution; Brent used %ld times; "
      "max iterations newton=%d brent=%d\n",
      s.total, s.recovered, s.worst_error, s.underflow, s.itm_ill_conditioned, s.used_brent,
      s.max_newton,
      s.max_brent);
  EXPECT_EQ(s.total, 8L * 300 * 61 * 2);
  EXPECT_EQ(s.recovered + s.underflow + s.itm_ill_conditioned, s.total);
  // Every option whose price carries the information is recovered.
  EXPECT_GT(s.recovered, s.total * 9 / 10);
  EXPECT_LT(s.worst_error, kIvTolerance);
  // The Newton stage does the bulk of the work; Brent only mops up.
  EXPECT_GT(s.used_brent, 0);
  EXPECT_LT(s.used_brent, s.total / 20);
}

TEST(IvSolver, BrentFallbackAloneMeetsTheSameTolerance) {
  IvConfig brent_only;
  brent_only.max_newton_iterations = 0;
  const GridStats s = run_stress_grid(brent_only);
  EXPECT_LT(s.worst_error, kIvTolerance);
  EXPECT_EQ(s.max_newton, 0);
  EXPECT_GT(s.recovered, s.total * 9 / 10);
  EXPECT_GE(s.used_brent, s.recovered);
}

TEST(IvSolver, VegaUnderflowIsHandledByBrentWithoutNaN) {
  // Far OTM with tiny total volatility: vega ~ exp(-d^2/2) is ~1e-285 while
  // the price is ~1e-290. Newton's derivative is unusable; Brent must finish.
  const OptionParams o{100.0, 100.0 * std::exp(0.364), 0.0, 0.0, 1.0, 0.01, true};
  const double price = bsm_scalar_price(o);
  ASSERT_GT(price, 0.0);
  ASSERT_LT(price, 1e-280);
  const IvResult r = implied_volatility(price, o);
  EXPECT_TRUE(is_finite(r.implied_vol));
  EXPECT_EQ(r.status, IvStatus::kConverged);
  EXPECT_NEAR(r.implied_vol, 0.01, kIvTolerance);

  // A price of exactly zero has no time value at all.
  const IvResult zero = implied_volatility(0.0, o);
  EXPECT_EQ(zero.status, IvStatus::kInvalidInput);
  EXPECT_EQ(zero.implied_vol, 0.0);
  const IvResult denormal = implied_volatility(5e-324, o);
  EXPECT_EQ(denormal.status, IvStatus::kBelowIntrinsic);
  EXPECT_EQ(denormal.implied_vol, 0.0);
}

TEST(IvSolver, HullImpliedVolatilityExample) {
  // Hull: call worth 1.875 with S=21, K=20, r=10%, T=0.25 implies 23.5% vol.
  const IvResult r = implied_volatility(1.875, {21.0, 20.0, 0.10, 0.0, 0.25, 0.0, true});
  EXPECT_EQ(r.status, IvStatus::kConverged);
  EXPECT_NEAR(r.implied_vol, 0.235, 0.0005);
}

TEST(IvSolver, RejectsPricesOutsideNoArbitrageBounds) {
  // Built from bit patterns: the Release build uses -ffast-math, under which
  // the compiler may assume NaN / infinity literals never occur.
  const double nan = std::bit_cast<double>(std::uint64_t{0x7ff8000000000000ULL});
  const double inf = std::bit_cast<double>(std::uint64_t{0x7ff0000000000000ULL});

  const OptionParams call{100.0, 90.0, 0.05, 0.02, 1.0, 0.0, true};
  const OptionParams put{100.0, 110.0, 0.05, 0.02, 1.0, 0.0, false};
  const double call_intrinsic = 100.0 * std::exp(-0.02) - 90.0 * std::exp(-0.05);
  const double put_intrinsic = 110.0 * std::exp(-0.05) - 100.0 * std::exp(-0.02);

  // "At intrinsic" is a knife edge at the ulp level (it depends on how the
  // compiler contracts S e^{-qT} - K e^{-rT}), so probe just inside the bounds.
  for (const auto& [price, option, expected] :
       {std::tuple{call_intrinsic - 1e-9, call, IvStatus::kBelowIntrinsic},
        std::tuple{call_intrinsic - 1.0, call, IvStatus::kBelowIntrinsic},
        std::tuple{100.0 * std::exp(-0.02) + 1e-9, call, IvStatus::kAboveUpperBound},
        std::tuple{250.0, call, IvStatus::kAboveUpperBound},
        std::tuple{put_intrinsic - 1e-9, put, IvStatus::kBelowIntrinsic},
        std::tuple{110.0 * std::exp(-0.05) + 1e-9, put, IvStatus::kAboveUpperBound},
        std::tuple{-1.0, call, IvStatus::kInvalidInput},
        std::tuple{nan, call, IvStatus::kInvalidInput},
        std::tuple{inf, call, IvStatus::kInvalidInput}}) {
    const IvResult r = implied_volatility(price, option);
    EXPECT_EQ(r.status, expected) << "price = " << price;
    EXPECT_EQ(r.implied_vol, 0.0);
  }

  // Just inside the bounds the solver must still produce a valid volatility.
  const IvResult low = implied_volatility(call_intrinsic + 1e-9, call);
  EXPECT_EQ(low.status, IvStatus::kConverged);
  EXPECT_GT(low.implied_vol, 0.0);
  const IvResult high = implied_volatility(100.0 * std::exp(-0.02) - 1e-9, call);
  EXPECT_EQ(high.status, IvStatus::kConverged);
  EXPECT_GT(high.implied_vol, 5.0);
  EXPECT_TRUE(is_finite(high.implied_vol));

  OptionParams bad = call;
  bad.spot = 0.0;
  EXPECT_EQ(implied_volatility(5.0, bad).status, IvStatus::kInvalidInput);
  bad = call;
  bad.strike = -90.0;
  EXPECT_EQ(implied_volatility(5.0, bad).status, IvStatus::kInvalidInput);
  bad = call;
  bad.time_to_expiry = 0.0;
  EXPECT_EQ(implied_volatility(5.0, bad).status, IvStatus::kInvalidInput);
  bad = call;
  bad.rate = nan;
  EXPECT_EQ(implied_volatility(5.0, bad).status, IvStatus::kInvalidInput);
}

TEST(IvSolver, ImpliedVolIsMonotoneInPrice) {
  const OptionParams o{100.0, 105.0, 0.02, 0.0, 0.75, 0.0, true};
  double previous = 0.0;
  for (double price = 0.05; price < 95.0; price *= 1.07) {
    const IvResult r = implied_volatility(price, o);
    ASSERT_EQ(r.status, IvStatus::kConverged) << price;
    EXPECT_GT(r.implied_vol, previous) << price;
    previous = r.implied_vol;
    // And the recovered vol reprices the option.
    OptionParams check = o;
    check.volatility = r.implied_vol;
    EXPECT_NEAR(bsm_scalar_price(check), price, 1e-12 * 100.0) << price;
  }
}

TEST(IvSolver, ChainApiRoundTripsMarketChain) {
  constexpr std::size_t kN = 20000;
  const OptionChain chain = testing::make_market_chain(kN, 314);
  GreeksBuffer greeks(kN);
  bsm_scalar_chain(chain.view(), greeks.view());

  AlignedVector<double> iv(kN, -1.0);
  std::vector<IvStatus> status(kN, IvStatus::kInvalidInput);
  const std::size_t converged = implied_volatility_chain(chain.view(), greeks.price, iv, status);
  EXPECT_EQ(converged, kN);
  double worst = 0.0;
  for (std::size_t i = 0; i < kN; ++i) {
    ASSERT_EQ(status[i], IvStatus::kConverged) << i;
    ASSERT_NEAR(iv[i], chain.volatility[i], kIvTolerance) << i;
    worst = std::max(worst, std::fabs(iv[i] - chain.volatility[i]));
  }
  EXPECT_LT(worst, 1e-11);  // in practice ~1e-13 on a realistic chain

  // The status span is optional.
  AlignedVector<double> iv2(kN, -1.0);
  EXPECT_EQ(implied_volatility_chain(chain.view(), greeks.price, iv2), kN);
  EXPECT_TRUE(std::equal(iv.begin(), iv.end(), iv2.begin()));
}

TEST(ZeroAllocation, IvSolverDoesNotAllocate) {
  constexpr std::size_t kN = 5000;
  const OptionChain chain = testing::make_market_chain(kN, 271);
  GreeksBuffer greeks(kN);
  bsm_scalar_chain(chain.view(), greeks.view());
  AlignedVector<double> iv(kN);
  std::vector<IvStatus> status(kN);
  const ChainView view = chain.view();
  const std::span<const double> prices = greeks.price;
  const std::span<double> iv_out = iv;
  const std::span<IvStatus> status_out = status;

  IvConfig brent_only;
  brent_only.max_newton_iterations = 0;
  std::size_t converged = 0;
  double sink = 0.0;
  {
    const AllocationCounter counter;
    converged += implied_volatility_chain(view, prices, iv_out, status_out);
    converged += implied_volatility_chain(view, prices, iv_out, status_out, brent_only);
    // Edge paths: underflow, bounds, invalid input.
    sink += implied_volatility(1e-300, {100.0, 200.0, 0.0, 0.0, 0.1, 0.0, true}).implied_vol;
    sink += implied_volatility(500.0, {100.0, 200.0, 0.0, 0.0, 0.1, 0.0, true}).implied_vol;
    sink += implied_volatility(-1.0, {100.0, 200.0, 0.0, 0.0, 0.1, 0.0, true}).implied_vol;
    EXPECT_EQ(counter.allocations(), 0U) << "the IV solver allocated on the heap";
  }
  EXPECT_EQ(converged, 2 * kN);
  EXPECT_TRUE(is_finite(sink));
}

}  // namespace
}  // namespace optengine
