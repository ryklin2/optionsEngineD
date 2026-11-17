// optengine/bsm_scalar.hpp
//
// Reference scalar Black-Scholes-Merton pricer and Greeks for European
// options on an asset paying a continuous dividend yield q.
//
// Formulas (S spot, K strike, r rate, q dividend yield, T years, v volatility)
// -----------------------------------------------------------------------------
//   d1 = [ln(S/K) + (r - q + v^2/2) T] / (v sqrt(T))        d2 = d1 - v sqrt(T)
//
//   call  = S e^{-qT} Phi(d1)  - K e^{-rT} Phi(d2)
//   put   = K e^{-rT} Phi(-d2) - S e^{-qT} Phi(-d1)
//
//   delta = e^{-qT} Phi(d1)                    (call)
//         = -e^{-qT} Phi(-d1)                  (put)
//   gamma = e^{-qT} phi(d1) / (S v sqrt(T))
//   vega  = S e^{-qT} phi(d1) sqrt(T)
//   theta = -S e^{-qT} phi(d1) v / (2 sqrt(T)) - r K e^{-rT} Phi(d2)  + q S e^{-qT} Phi(d1)    (call)
//         = -S e^{-qT} phi(d1) v / (2 sqrt(T)) + r K e^{-rT} Phi(-d2) - q S e^{-qT} Phi(-d1)   (put)
//   rho   = K T e^{-rT} Phi(d2)                (call)
//         = -K T e^{-rT} Phi(-d2)              (put)
//
// Units
// -----
//   T is in years, v and r are annualized (0.20 = 20%).
//   vega  is per 1.00 change in volatility (100 vol points); divide by 100 for per-1%.
//   rho   is per 1.00 change in rate;                        divide by 100 for per-1%.
//   theta is dV/dt PER YEAR. Use `theta_per_calendar_day` (divide by 365) for the
//         one-calendar-day decay quoted on most trading screens.
//
// Preconditions: S > 0, K > 0, T > 0, v > 0. The kernels are branch-light and do
// not special-case expiry or zero volatility.
#pragma once

#include <cmath>

#include "optengine/math_utils.hpp"
#include "optengine/types.hpp"

namespace optengine {

/// Calendar days per year used to convert annual theta to daily theta.
inline constexpr double kCalendarDaysPerYear = 365.0;

/// Converts theta per year (as produced by the pricers) to theta per calendar day.
[[nodiscard]] constexpr double theta_per_calendar_day(double theta_per_year) noexcept {
  return theta_per_year / kCalendarDaysPerYear;
}

/// Price and Greeks of one European option (reference implementation).
[[nodiscard]] inline Greeks bsm_scalar_greeks(const OptionParams& o) noexcept {
  const double sqrt_t = std::sqrt(o.time_to_expiry);
  const double vol_sqrt_t = o.volatility * sqrt_t;
  const double drift =
      (o.rate - o.dividend + 0.5 * o.volatility * o.volatility) * o.time_to_expiry;
  const double d1 = (std::log(o.spot / o.strike) + drift) / vol_sqrt_t;
  const double d2 = d1 - vol_sqrt_t;

  const double disc_q = std::exp(-o.dividend * o.time_to_expiry);
  const double disc_r = std::exp(-o.rate * o.time_to_expiry);
  const double spot_disc = o.spot * disc_q;      // S e^{-qT}
  const double strike_disc = o.strike * disc_r;  // K e^{-rT}

  const NormalDist n1 = norm_dist(d1);
  const NormalDist n2 = norm_dist(d2);

  const double spot_pdf = spot_disc * n1.pdf;  // S e^{-qT} phi(d1)
  const double decay = -0.5 * spot_pdf * o.volatility / sqrt_t;

  Greeks g;
  g.gamma = disc_q * n1.pdf / (o.spot * vol_sqrt_t);
  g.vega = spot_pdf * sqrt_t;
  if (o.is_call) {
    g.price = spot_disc * n1.cdf - strike_disc * n2.cdf;
    g.delta = disc_q * n1.cdf;
    g.theta = decay - o.rate * strike_disc * n2.cdf + o.dividend * spot_disc * n1.cdf;
    g.rho = o.time_to_expiry * strike_disc * n2.cdf;
  } else {
    g.price = strike_disc * n2.ccdf - spot_disc * n1.ccdf;
    g.delta = -disc_q * n1.ccdf;
    g.theta = decay + o.rate * strike_disc * n2.ccdf - o.dividend * spot_disc * n1.ccdf;
    g.rho = -o.time_to_expiry * strike_disc * n2.ccdf;
  }
  return g;
}

/// Price only of one European option (reference implementation).
[[nodiscard]] inline double bsm_scalar_price(const OptionParams& o) noexcept {
  const double vol_sqrt_t = o.volatility * std::sqrt(o.time_to_expiry);
  const double drift =
      (o.rate - o.dividend + 0.5 * o.volatility * o.volatility) * o.time_to_expiry;
  const double d1 = (std::log(o.spot / o.strike) + drift) / vol_sqrt_t;
  const double d2 = d1 - vol_sqrt_t;
  const double spot_disc = o.spot * std::exp(-o.dividend * o.time_to_expiry);
  const double strike_disc = o.strike * std::exp(-o.rate * o.time_to_expiry);
  const NormalDist n1 = norm_dist(d1);
  const NormalDist n2 = norm_dist(d2);
  return o.is_call ? spot_disc * n1.cdf - strike_disc * n2.cdf
                   : strike_disc * n2.ccdf - spot_disc * n1.ccdf;
}

/// Prices a whole chain one option at a time. Deliberately NOT vectorized: this
/// is the baseline the SIMD engine is validated and benchmarked against.
/// `out` spans must have at least `in.size()` elements. No heap allocation.
void bsm_scalar_chain(ChainView in, GreeksView out) noexcept;

}  // namespace optengine
