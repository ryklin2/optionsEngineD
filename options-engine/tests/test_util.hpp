// Shared helpers for the optengine test suite.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

#include "optengine/types.hpp"

namespace optengine::testing {

/// Fills a chain with randomized but realistic contracts:
///   S in [50, 150], K/S in [0.6, 1.6], r in [0, 8%], q in [0, 5%],
///   T in [0.02, 3] years, sigma in [5%, 90%], call/put at random.
inline OptionChain make_random_chain(std::size_t n, std::uint64_t seed) {
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

/// A listed-options style market: several expiries, strikes within +-3
/// standard deviations of the forward, a volatility smile with skew, and the
/// out-of-the-money option quoted at each strike (calls above the forward,
/// puts below), which is what implied-volatility surfaces are built from.
inline OptionChain make_market_chain(std::size_t n, std::uint64_t seed) {
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
    const double log_moneyness = z * base * std::sqrt(t);  // ln(K / F)
    const double vol = base * (1.0 - 0.08 * z + 0.03 * z * z);  // skewed smile
    const double strike = s * std::exp((r - q) * t + log_moneyness);
    chain.set(i, {s, strike, r, q, t, vol, log_moneyness >= 0.0});
  }
  return chain;
}

}  // namespace optengine::testing
