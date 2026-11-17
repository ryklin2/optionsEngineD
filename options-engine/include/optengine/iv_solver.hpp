// optengine/iv_solver.hpp
//
// Implied volatility: hybrid Newton-Raphson with a Brent fallback.
//
// Problem transformation
// ----------------------
// With S' = S e^{-qT}, K' = K e^{-rT} (discounted forward and strike):
//
//   x     = -|ln(S'/K')|                         log-moneyness of the OTM option
//   s     = sigma * sqrt(T)                      total volatility (unknown)
//   beta  = (price - intrinsic) / sqrt(S' K')    normalized time value
//   b(s)  = e^{x/2} Phi(x/s + s/2) - e^{-x/2} Phi(x/s - s/2)
//   b'(s) = exp(-x^2/(2 s^2) - s^2/8) / sqrt(2 pi)      (normalized vega)
//
// Put-call parity maps every option to the out-of-the-money one with the same
// time value, so calls and puts, ITM and OTM all reduce to b(s) = beta with
// 0 < beta < b_max = e^{x/2}.
//
// Newton stage
// ------------
// b(s) has an inflection at s_c = sqrt(2|x|). Plain Newton on b(s) stalls far
// from the money, where vega underflows. Instead Newton runs on a transformed
// objective that is nearly quadratic/linear in s in each branch:
//
//   beta <  b(s_c) :  g(s) = -1/ln b(s) + 1/ln beta           (~ s^2 / x^2 * 2)
//   beta >= b(s_c) :  g(s) = ln(b_max - beta) - ln(b_max - b(s))
//
// Both are increasing in s. Every iterate tightens a bracket [lo, hi] around
// the root.
//
// Brent fallback
// --------------
// If the transformed derivative is unusable (vega or price underflowed), a
// Newton step leaves the bracket, or the Newton budget is exhausted, the
// solver finishes with Brent's method on g over the bracket, which cannot
// diverge and needs no derivative.
//
// Units: `price` in currency, result is annualized volatility (0.20 = 20%).
// The solver performs no heap allocation and never returns NaN: on failure
// `implied_vol` is 0 and `status` says why.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "optengine/types.hpp"

namespace optengine {

enum class IvStatus : std::uint8_t {
  kConverged = 0,     ///< implied_vol is valid
  kInvalidInput,      ///< non-finite or non-positive spot/strike/expiry/price
  kBelowIntrinsic,    ///< price <= discounted intrinsic value: no time value to invert
  kAboveUpperBound,   ///< price >= S e^{-qT} (call) or K e^{-rT} (put): no finite vol
  kMaxIterations,     ///< iteration budget exhausted; implied_vol is the best estimate
};

[[nodiscard]] std::string_view to_string(IvStatus status) noexcept;

struct IvConfig {
  /// Newton iterations before handing over to Brent. 0 forces the Brent path.
  int max_newton_iterations = 24;
  /// Brent iterations (each costs one objective evaluation).
  int max_brent_iterations = 128;
  /// Newton stops once the relative step |ds|/s is below this. Convergence is
  /// quadratic, so the error after the final step is far smaller (~tol^2).
  double newton_step_tolerance = 1e-9;
  /// Relative width of the Brent bracket at termination.
  double brent_tolerance = 1e-15;
};

struct IvResult {
  double implied_vol = 0.0;
  IvStatus status = IvStatus::kInvalidInput;
  std::uint8_t newton_iterations = 0;
  std::uint8_t brent_iterations = 0;
  bool used_brent = false;
};

/// Implied volatility of one European option. `option.volatility` is ignored.
[[nodiscard]] IvResult implied_volatility(double price, const OptionParams& option,
                                          const IvConfig& config = {}) noexcept;

/// Implied volatilities for a whole chain (`chain.volatility` is ignored and
/// may be empty). Writes `implied_vol[i]` (0 on failure) and, when `status` is
/// non-empty, `status[i]`. Returns the number of converged options.
std::size_t implied_volatility_chain(ChainView chain, std::span<const double> prices,
                                     std::span<double> implied_vol,
                                     std::span<IvStatus> status = {},
                                     const IvConfig& config = {}) noexcept;

}  // namespace optengine
