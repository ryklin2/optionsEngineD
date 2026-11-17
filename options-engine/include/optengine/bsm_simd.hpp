// optengine/bsm_simd.hpp
//
// Explicitly vectorized (AVX2 + FMA) Black-Scholes-Merton pricer and Greeks
// operating on Structure-of-Arrays option chains, four options per iteration.
//
// Same formulas, units and preconditions as bsm_scalar.hpp. Differences:
//   * exp / log / Phi come from the vector kernels in math_utils.hpp instead
//     of libm, so results agree with the scalar reference to a few ulp
//     (not bit-for-bit).
//   * Calls and puts are both evaluated and blended by mask (branch-free).
//   * A tail of n % 4 options is run through the same vector kernel on a
//     padded stack buffer, so remainder elements are computed identically to
//     full-width elements and nothing is read or written out of bounds.
//
// When the translation unit is compiled without AVX2/FMA (`__AVX2__` or
// `__FMA__` undefined) `bsm_simd_chain` falls back to the scalar reference.
#pragma once

#include <string_view>

#include "optengine/math_utils.hpp"
#include "optengine/types.hpp"

namespace optengine {

/// True when the AVX2 path is compiled in.
inline constexpr bool kHasAvx2 = OPTENGINE_HAS_AVX2 != 0;

/// "avx2+fma" or "scalar-fallback".
[[nodiscard]] std::string_view bsm_simd_backend() noexcept;

/// Prices a chain and fills price, delta, gamma, vega, theta (per year), rho.
///
/// `out` spans must have at least `in.size()` elements and must not alias the
/// inputs. Aligned loads/stores are used when every array is 32-byte aligned
/// (always true for OptionChain / GreeksBuffer); otherwise unaligned ones.
/// Performs no heap allocation.
void bsm_simd_chain(ChainView in, GreeksView out) noexcept;

#if OPTENGINE_HAS_AVX2

/// Results of the 4-wide kernel.
struct GreeksPd {
  __m256d price;
  __m256d delta;
  __m256d gamma;
  __m256d vega;
  __m256d theta;
  __m256d rho;
};

/// Four-option BSM kernel. `is_put` lanes are all-ones for puts, zero for calls.
[[nodiscard]] inline GreeksPd bsm_kernel_avx2(__m256d spot, __m256d strike, __m256d rate,
                                              __m256d dividend, __m256d expiry, __m256d vol,
                                              __m256d is_put) noexcept {
  const __m256d half = _mm256_set1_pd(0.5);
  const __m256d sqrt_t = _mm256_sqrt_pd(expiry);
  const __m256d vol_sqrt_t = _mm256_mul_pd(vol, sqrt_t);

  // d1 = [ln(S/K) + (r - q + v^2/2) T] / (v sqrt(T)),  d2 = d1 - v sqrt(T)
  const __m256d carry = _mm256_sub_pd(rate, dividend);
  const __m256d drift =
      _mm256_mul_pd(_mm256_fmadd_pd(_mm256_mul_pd(half, vol), vol, carry), expiry);
  const __m256d log_moneyness = log_pd(_mm256_div_pd(spot, strike));
  const __m256d d1 = _mm256_div_pd(_mm256_add_pd(log_moneyness, drift), vol_sqrt_t);
  const __m256d d2 = _mm256_sub_pd(d1, vol_sqrt_t);

  const __m256d neg_t = _mm256_sub_pd(_mm256_setzero_pd(), expiry);
  const __m256d disc_q = exp_pd(_mm256_mul_pd(neg_t, dividend));
  const __m256d disc_r = exp_pd(_mm256_mul_pd(neg_t, rate));
  const __m256d spot_disc = _mm256_mul_pd(spot, disc_q);      // S e^{-qT}
  const __m256d strike_disc = _mm256_mul_pd(strike, disc_r);  // K e^{-rT}

  const NormalDistPd n1 = norm_dist_pd(d1);
  const NormalDistPd n2 = norm_dist_pd(d2);

  const __m256d spot_pdf = _mm256_mul_pd(spot_disc, n1.pdf);  // S e^{-qT} phi(d1)
  // decay = -S e^{-qT} phi(d1) v / (2 sqrt(T))
  const __m256d decay = _mm256_div_pd(
      _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(-0.5), spot_pdf), vol), sqrt_t);

  // Call leg.
  const __m256d call_s = _mm256_mul_pd(spot_disc, n1.cdf);    // S e^{-qT} Phi(d1)
  const __m256d call_k = _mm256_mul_pd(strike_disc, n2.cdf);  // K e^{-rT} Phi(d2)
  const __m256d call_price = _mm256_sub_pd(call_s, call_k);
  const __m256d call_delta = _mm256_mul_pd(disc_q, n1.cdf);
  const __m256d call_theta =
      _mm256_fmadd_pd(dividend, call_s, _mm256_fnmadd_pd(rate, call_k, decay));
  const __m256d call_rho = _mm256_mul_pd(expiry, call_k);

  // Put leg.
  const __m256d put_s = _mm256_mul_pd(spot_disc, n1.ccdf);    // S e^{-qT} Phi(-d1)
  const __m256d put_k = _mm256_mul_pd(strike_disc, n2.ccdf);  // K e^{-rT} Phi(-d2)
  const __m256d put_price = _mm256_sub_pd(put_k, put_s);
  const __m256d put_delta = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_mul_pd(disc_q, n1.ccdf));
  const __m256d put_theta =
      _mm256_fnmadd_pd(dividend, put_s, _mm256_fmadd_pd(rate, put_k, decay));
  const __m256d put_rho = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_mul_pd(expiry, put_k));

  GreeksPd g;
  g.price = _mm256_blendv_pd(call_price, put_price, is_put);
  g.delta = _mm256_blendv_pd(call_delta, put_delta, is_put);
  g.gamma = _mm256_div_pd(_mm256_mul_pd(disc_q, n1.pdf), _mm256_mul_pd(spot, vol_sqrt_t));
  g.vega = _mm256_mul_pd(spot_pdf, sqrt_t);
  g.theta = _mm256_blendv_pd(call_theta, put_theta, is_put);
  g.rho = _mm256_blendv_pd(call_rho, put_rho, is_put);
  return g;
}

#endif  // OPTENGINE_HAS_AVX2

}  // namespace optengine
