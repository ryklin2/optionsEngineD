#include "optengine/iv_solver.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include "optengine/math_utils.hpp"

namespace optengine {

std::string_view to_string(IvStatus status) noexcept {
  switch (status) {
    case IvStatus::kConverged: return "converged";
    case IvStatus::kInvalidInput: return "invalid-input";
    case IvStatus::kBelowIntrinsic: return "below-intrinsic";
    case IvStatus::kAboveUpperBound: return "above-upper-bound";
    case IvStatus::kMaxIterations: return "max-iterations";
  }
  return "unknown";
}

namespace {

constexpr double kTiny = std::numeric_limits<double>::min();     // smallest normal double
constexpr double kEps = std::numeric_limits<double>::epsilon();
/// Total volatility above which b_max - b(s) underflows for every moneyness.
constexpr double kMaxTotalVol = 80.0;
/// Stand-in for "+infinity" objective values. Real objective values are bounded
/// by ~1.5e3 in magnitude, and a modest sentinel keeps Brent's interpolation
/// arithmetic finite (no IEEE infinities: the release build uses -ffast-math).
constexpr double kHuge = 1.0e4;

/// Normalized Black problem for one option: solve b(s) = beta for s.
struct Problem {
  double x;          ///< -|ln(S'/K')| <= 0
  double b_max;      ///< e^{x/2}
  double inv_b_max;  ///< e^{-x/2}
  bool low_branch;   ///< beta below the inflection value b(s_c)
  double target;     ///< -1/ln(beta) (low branch) or ln(b_max - beta) (high branch)

  struct Eval {
    double g;        ///< objective, increasing in s, root at the solution
    double dg;       ///< dg/ds, valid only if `has_derivative`
    bool has_derivative;
  };

  /// b(s), b_max - b(s) and b'(s). s > 0.
  struct Black {
    double b;
    double complement;
    double vega;
  };

  [[nodiscard]] Black black(double s) const noexcept {
    const double h = x / s;
    const double t = 0.5 * s;
    const NormalDist n1 = norm_dist(h + t);
    const NormalDist n2 = norm_dist(h - t);
    return {b_max * n1.cdf - inv_b_max * n2.cdf, b_max * n1.ccdf + inv_b_max * n2.cdf,
            b_max * n1.pdf};
  }

  [[nodiscard]] Eval eval(double s) const noexcept {
    if (!(s > 0.0)) {
      // b(0) = 0: the objective sits at its lower limit.
      return {low_branch ? -target : target - std::log(b_max), 0.0, false};
    }
    const Black k = black(s);
    if (low_branch) {
      if (!(k.b > kTiny)) return {-target, 0.0, false};   // price underflow
      const double ln_b = std::log(k.b);
      if (!(ln_b < 0.0)) return {kHuge, 0.0, false};
      const double g = -1.0 / ln_b - target;
      const bool ok = k.vega > kTiny;
      return {g, ok ? k.vega / (k.b * ln_b * ln_b) : 0.0, ok};
    }
    if (!(k.complement > kTiny)) return {kHuge, 0.0, false};  // saturated at b_max
    const double g = target - std::log(k.complement);
    const bool ok = k.vega > kTiny;
    return {g, ok ? k.vega / k.complement : 0.0, ok};
  }
};

struct RootResult {
  double s;
  bool converged;
  int iterations;
};

/// Brent's method on problem.eval(s).g over [lo, hi] with g(lo) <= 0 <= g(hi).
[[nodiscard]] RootResult brent(const Problem& problem, double lo, double g_lo, double hi,
                               double g_hi, const IvConfig& config) noexcept {
  double a = lo, fa = g_lo;
  double b = hi, fb = g_hi;
  double c = a, fc = fa;
  double d = b - a;
  double e = d;
  for (int it = 1; it <= config.max_brent_iterations; ++it) {
    if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
      c = a;
      fc = fa;
      d = b - a;
      e = d;
    }
    if (std::fabs(fc) < std::fabs(fb)) {
      a = b;  b = c;  c = a;
      fa = fb; fb = fc; fc = fa;
    }
    const double tol = 2.0 * kEps * std::fabs(b) + 0.5 * config.brent_tolerance * std::fabs(b) +
                       kTiny;
    const double mid = 0.5 * (c - b);
    if (std::fabs(mid) <= tol || fb == 0.0) return {b, true, it - 1};

    if (std::fabs(e) >= tol && std::fabs(fa) > std::fabs(fb)) {
      // Inverse quadratic interpolation (secant when only two points differ).
      const double r = fb / fa;
      double p;
      double q;
      if (a == c) {
        p = 2.0 * mid * r;
        q = 1.0 - r;
      } else {
        const double qa = fa / fc;
        const double rb = fb / fc;
        p = r * (2.0 * mid * qa * (qa - rb) - (b - a) * (rb - 1.0));
        q = (qa - 1.0) * (rb - 1.0) * (r - 1.0);
      }
      if (p > 0.0) q = -q;
      p = std::fabs(p);
      const double bound1 = 3.0 * mid * q - std::fabs(tol * q);
      const double bound2 = std::fabs(e * q);
      if (2.0 * p < std::min(bound1, bound2)) {
        e = d;
        d = p / q;
      } else {
        d = mid;
        e = d;
      }
    } else {
      d = mid;
      e = d;
    }
    a = b;
    fa = fb;
    b += std::fabs(d) > tol ? d : (mid > 0.0 ? tol : -tol);
    fb = problem.eval(b).g;
  }
  return {b, false, config.max_brent_iterations};
}

}  // namespace

IvResult implied_volatility(double price, const OptionParams& o, const IvConfig& config) noexcept {
  IvResult result;
  const bool finite = is_finite(price) && is_finite(o.spot) && is_finite(o.strike) &&
                      is_finite(o.rate) && is_finite(o.dividend) && is_finite(o.time_to_expiry);
  if (!finite || !(price > 0.0) || !(o.spot > 0.0) || !(o.strike > 0.0) ||
      !(o.time_to_expiry > 0.0)) {
    result.status = IvStatus::kInvalidInput;
    return result;
  }

  const double sqrt_t = std::sqrt(o.time_to_expiry);
  const double spot_disc = o.spot * std::exp(-o.dividend * o.time_to_expiry);
  const double strike_disc = o.strike * std::exp(-o.rate * o.time_to_expiry);
  if (!(spot_disc > kTiny) || !(strike_disc > kTiny) || !is_finite(spot_disc) ||
      !is_finite(strike_disc)) {
    result.status = IvStatus::kInvalidInput;
    return result;
  }

  // No-arbitrage bounds, then reduce to the OTM option's time value.
  const double upper = o.is_call ? spot_disc : strike_disc;
  const double intrinsic =
      std::max(o.is_call ? spot_disc - strike_disc : strike_disc - spot_disc, 0.0);
  if (!(price < upper)) {
    result.status = IvStatus::kAboveUpperBound;
    return result;
  }
  const double time_value = price - intrinsic;
  const double norm = std::sqrt(spot_disc) * std::sqrt(strike_disc);
  const double beta = time_value / norm;
  if (!(time_value > 0.0) || !(beta > kTiny)) {
    result.status = IvStatus::kBelowIntrinsic;
    return result;
  }
  const double otm_bound = std::min(spot_disc, strike_disc);
  const double complement_beta = (otm_bound - time_value) / norm;  // b_max - beta
  if (!(complement_beta > kTiny)) {
    result.status = IvStatus::kAboveUpperBound;
    return result;
  }

  Problem problem;
  problem.x = -std::fabs(std::log(spot_disc / strike_disc));
  problem.b_max = std::sqrt(otm_bound / std::max(spot_disc, strike_disc));
  problem.inv_b_max = 1.0 / problem.b_max;

  // Branch selection at the inflection point s_c = sqrt(2|x|).
  const double s_c = std::sqrt(-2.0 * problem.x);
  double b_c = 0.0;
  if (s_c > 0.0) b_c = problem.black(s_c).b;
  problem.low_branch = beta < b_c;

  double lo;
  double hi;
  double s;
  if (problem.low_branch) {
    // ln(beta) < ln(b_c) < 0, so the target is finite and positive.
    problem.target = -1.0 / std::log(beta);
    lo = 0.0;
    hi = s_c;
    // Asymptotically -1/ln b ~ 2 s^2 / x^2  =>  s ~ |x| sqrt(target / 2).
    s = std::min(s_c, -problem.x * std::sqrt(0.5 * problem.target));
  } else {
    problem.target = std::log(complement_beta);
    lo = s_c;
    hi = kMaxTotalVol;
    // Brenner-Subrahmanyam: at the money b(s) ~ s / sqrt(2 pi).
    s = std::min(std::max(s_c, kSqrt2Pi * beta), 0.5 * kMaxTotalVol);
  }

  // ---- Newton-Raphson on the transformed objective, bracket maintained ----
  bool converged = false;
  int newton_its = 0;
  while (newton_its < config.max_newton_iterations) {
    const Problem::Eval f = problem.eval(s);
    ++newton_its;
    if (f.g == 0.0) {
      converged = true;
      break;
    }
    if (f.g < 0.0) {
      lo = s;
    } else {
      hi = s;
    }
    if (!f.has_derivative) break;
    const double next = s - f.g / f.dg;
    if (!(next > lo) || !(next < hi)) break;  // left the bracket: hand over to Brent
    const double step = std::fabs(next - s);
    s = next;
    if (step <= config.newton_step_tolerance * s) {
      converged = true;
      break;
    }
  }
  result.newton_iterations = static_cast<std::uint8_t>(std::min(newton_its, 255));

  // ---- Brent fallback on the bracket ----
  if (!converged) {
    result.used_brent = true;
    const double g_lo = problem.eval(lo).g;
    const double g_hi = problem.eval(hi).g;
    if (g_lo > 0.0) {
      s = lo;  // root pinned at the lower end (rounding at the branch boundary)
      converged = true;
    } else if (g_hi < 0.0) {
      s = hi;  // root pinned at the upper end (rounding at the branch boundary)
      converged = true;
    } else {
      const RootResult root = brent(problem, lo, g_lo, hi, g_hi, config);
      s = root.s;
      converged = root.converged;
      result.brent_iterations = static_cast<std::uint8_t>(std::min(root.iterations, 255));
    }
  }

  result.implied_vol = s / sqrt_t;
  result.status = converged ? IvStatus::kConverged : IvStatus::kMaxIterations;
  return result;
}

std::size_t implied_volatility_chain(ChainView chain, std::span<const double> prices,
                                     std::span<double> implied_vol, std::span<IvStatus> status,
                                     const IvConfig& config) noexcept {
  const std::size_t n = prices.size();
  assert(chain.spot.size() >= n && chain.strike.size() >= n && chain.rate.size() >= n &&
         chain.dividend.size() >= n && chain.time_to_expiry.size() >= n &&
         chain.is_call.size() >= n);
  assert(implied_vol.size() >= n && (status.empty() || status.size() >= n));

  std::size_t converged = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const OptionParams option{chain.spot[i], chain.strike[i], chain.rate[i], chain.dividend[i],
                              chain.time_to_expiry[i], 0.0, chain.is_call[i] != 0};
    const IvResult r = implied_volatility(prices[i], option, config);
    implied_vol[i] = r.implied_vol;
    if (!status.empty()) status[i] = r.status;
    converged += r.status == IvStatus::kConverged ? 1U : 0U;
  }
  return converged;
}

}  // namespace optengine
