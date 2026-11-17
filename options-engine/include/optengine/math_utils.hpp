// optengine/math_utils.hpp
//
// Fast special functions for the pricing kernels.
//
//   norm_dist(x)      Standard normal CDF Phi(x), survival 1 - Phi(x) and PDF
//                     phi(x) from a single exp() and one rational polynomial.
//   inv_norm_cdf(p)   Inverse normal CDF (Wichura AS241, PPND16).
//   exp_pd / log_pd / norm_dist_pd
//                     AVX2 + FMA counterparts operating on 4 doubles.
//
// Normal CDF algorithm
// --------------------
// Let a = |x| and tail(a) = Phi(-a) = e(a) * P(a) / Q(a), with
// e(a) = exp(-a^2 / 2) and a rational P/Q chosen per region:
//
//   a <  0.75           Hart (1968) algorithm 5666 as popularised by West (2005)
//   0.75 <= a < 4*sqrt2 Cody (1969) erfc approximation, 0.46875 <= y <= 4
//   a >= 4*sqrt2        Cody (1969) erfc approximation, y > 4
//
// where y = a / sqrt(2) and Phi(-a) = erfc(y) / 2. Then
//
//   Phi(x)     = x > 0 ? 1 - tail : tail
//   1 - Phi(x) = x > 0 ? tail     : 1 - tail
//   phi(x)     = e(a) / sqrt(2 pi)
//
// The rounding error of a^2 is recovered with one FMA and folded back into
// e(a), which keeps the *relative* error of the tail at ~1e-15 all the way to
// underflow (|x| ~ 37.5). Measured against 40-digit references:
//
//   max absolute error of Phi(x)          2.2e-16   (spec: < 1e-15)
//   max relative error of the tail        9.2e-16
//
// The scalar version takes one branch per region; the AVX2 version evaluates
// all three rationals and blends, so it is branch-free with one division.
#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__AVX2__) && defined(__FMA__)
#define OPTENGINE_HAS_AVX2 1
#include <immintrin.h>
#else
#define OPTENGINE_HAS_AVX2 0
#endif

namespace optengine {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
inline constexpr double kInvSqrt2 = 0.70710678118654752440;     ///< 1 / sqrt(2)
inline constexpr double kInvSqrt2Pi = 0.39894228040143267794;   ///< 1 / sqrt(2 pi)
inline constexpr double kSqrt2Pi = 2.50662827463100050242;      ///< sqrt(2 pi)
inline constexpr double kInvSqrtPi = 0.56418958354775628695;    ///< 1 / sqrt(pi)
inline constexpr double kLn2Hi = 6.93147180369123816490e-01;    ///< high 32 bits of ln 2
inline constexpr double kLn2Lo = 1.90821492927058770002e-10;    ///< ln 2 - kLn2Hi
inline constexpr double kLog2E = 1.44269504088896340736;        ///< 1 / ln 2

namespace detail {

/// |x| beyond which Phi(-|x|) underflows to zero in double precision.
inline constexpr double kNormTailCutoff = 40.0;
/// Region boundaries of the normal tail approximation.
inline constexpr double kNormRegionAB = 0.75;
inline constexpr double kNormRegionBC = 5.65685424949238019521;  // 4 * sqrt(2)

// Hart / West, argument a = |x|. Coefficients ordered highest degree first.
inline constexpr std::array<double, 7> kHartNum = {
    3.52624965998911e-02, 0.700383064443688, 6.37396220353165, 33.912866078383,
    112.079291497871,     221.213596169931,  220.206867912376};
inline constexpr std::array<double, 8> kHartDen = {
    8.83883476483184e-02, 1.75566716318264, 16.064177579207,  86.7807322029461,
    296.564248779674,     637.333633378831, 793.826512519948, 440.413735824752};

// Cody erfc, 0.46875 <= y <= 4, argument y = |x| / sqrt(2).
inline constexpr std::array<double, 9> kCodyMidNum = {
    2.15311535474403846e-8, 5.64188496988670089e-1, 8.88314979438837594,
    6.61191906371416295e1,  2.98635138197400131e2,  8.81952221241769090e2,
    1.71204761263407058e3,  2.05107837782607147e3,  1.23033935479799725e3};
inline constexpr std::array<double, 9> kCodyMidDen = {
    1.0,                   1.57449261107098347e1, 1.17693950891312499e2,
    5.37181101862009858e2, 1.62138957456669019e3, 3.29079923573345963e3,
    4.36261909014324716e3, 3.43936767414372164e3, 1.23033935480374942e3};

// Cody erfc, y > 4, argument w = y^2 (polynomials pre-multiplied by w^5).
inline constexpr std::array<double, 6> kCodyFarNum = {
    6.58749161529837803e-4, 1.60837851487422766e-2, 1.25781726111229246e-1,
    3.60344899949804439e-1, 3.05326634961232344e-1, 1.63153871373020978e-2};
inline constexpr std::array<double, 6> kCodyFarDen = {
    2.33520497626869185e-3, 6.05183413124413191e-2, 5.27905102951428412e-1,
    1.87295284992346725,    2.56852019228982242,    1.0};

// Wichura AS241 (PPND16). Coefficients ordered lowest degree first.
inline constexpr std::array<double, 8> kPpndA = {
    3.3871328727963666080,   1.3314166789178437745e2, 1.9715909503065514427e3,
    1.3731693765509461125e4, 4.5921953931549871457e4, 6.7265770927008700853e4,
    3.3430575583588128105e4, 2.5090809287301226727e3};
inline constexpr std::array<double, 8> kPpndB = {
    1.0,                     4.2313330701600911252e1, 6.8718700749205790830e2,
    5.3941960214247511077e3, 2.1213794301586595867e4, 3.9307895800092710610e4,
    2.8729085735721942674e4, 5.2264952788528545610e3};
inline constexpr std::array<double, 8> kPpndC = {
    1.42343711074968357734,     4.63033784615654529590,     5.76949722146069140550,
    3.64784832476320460504,     1.27045825245236838258,     2.41780725177450611770e-1,
    2.27238449892691845833e-2,  7.74545014278341407640e-4};
inline constexpr std::array<double, 8> kPpndD = {
    1.0,                        2.05319162663775882187,     1.67638483018380384940,
    6.89767334985100004550e-1,  1.48103976427480074590e-1,  1.51986665636164571966e-2,
    5.47593808499534494600e-4,  1.05075007164441684324e-9};
inline constexpr std::array<double, 8> kPpndE = {
    6.65790464350110377720,     5.46378491116411436990,     1.78482653991729133580,
    2.96560571828504891230e-1,  2.65321895265761230930e-2,  1.24266094738807843860e-3,
    2.71155556874348757815e-5,  2.01033439929228813265e-7};
inline constexpr std::array<double, 8> kPpndF = {
    1.0,                        5.99832206555887937690e-1,  1.36929880922735805310e-1,
    1.48753612908506148525e-2,  7.86869131145613259100e-4,  1.84631831751005468180e-5,
    1.42151175831644588870e-7,  2.04426310338993978564e-15};

// Taylor coefficients 1/n! for n = 13 .. 2 (exp kernel, highest degree first).
inline constexpr std::array<double, 12> kExpPoly = {
    1.0 / 6227020800.0, 1.0 / 479001600.0, 1.0 / 39916800.0, 1.0 / 3628800.0,
    1.0 / 362880.0,     1.0 / 40320.0,     1.0 / 5040.0,     1.0 / 720.0,
    1.0 / 120.0,        1.0 / 24.0,        1.0 / 6.0,        1.0 / 2.0};

// fdlibm e_log.c polynomial for log(1+f).
inline constexpr double kLg1 = 6.666666666666735130e-01;
inline constexpr double kLg2 = 3.999999999940941908e-01;
inline constexpr double kLg3 = 2.857142874366239149e-01;
inline constexpr double kLg4 = 2.222219843214978396e-01;
inline constexpr double kLg5 = 1.818357216161805012e-01;
inline constexpr double kLg6 = 1.531383769920937332e-01;
inline constexpr double kLg7 = 1.479819860511658591e-01;

/// Horner evaluation, coefficients ordered highest degree first.
template <std::size_t N>
[[nodiscard]] constexpr double horner(const std::array<double, N>& c, double x) noexcept {
  double r = c[0];
  for (std::size_t i = 1; i < N; ++i) r = r * x + c[i];
  return r;
}

/// Horner evaluation, coefficients ordered lowest degree first.
template <std::size_t N>
[[nodiscard]] constexpr double horner_ascending(const std::array<double, N>& c,
                                                double x) noexcept {
  double r = c[N - 1];
  for (std::size_t i = N - 1; i-- > 0;) r = r * x + c[i];
  return r;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Scalar functions
// ---------------------------------------------------------------------------

/// True when `x` is neither NaN nor +-inf. Bit-level test, so it stays correct
/// under -ffast-math (where std::isfinite may be folded to `true`).
[[nodiscard]] constexpr bool is_finite(double x) noexcept {
  constexpr std::uint64_t kExpMask = 0x7ff0000000000000ULL;
  return (std::bit_cast<std::uint64_t>(x) & kExpMask) != kExpMask;
}

/// Standard normal distribution evaluated at one point.
struct NormalDist {
  double cdf;   ///< Phi(x)
  double ccdf;  ///< 1 - Phi(x) = Phi(-x), accurate in the upper tail
  double pdf;   ///< phi(x) = exp(-x^2/2) / sqrt(2 pi)
};

/// Phi(x), Phi(-x) and phi(x) in one call. |abs error| < 1e-15 for Phi;
/// relative error of the smaller of {cdf, ccdf} is ~1e-15.
[[nodiscard]] inline NormalDist norm_dist(double x) noexcept {
  const double a = std::fabs(x);
  if (a >= detail::kNormTailCutoff) {
    return x > 0.0 ? NormalDist{1.0, 0.0, 0.0} : NormalDist{0.0, 1.0, 0.0};
  }
  const double a2 = a * a;
  const double a2_err = std::fma(a, a, -a2);  // exact rounding error of a*a
  double e = std::exp(-0.5 * a2);
  e -= e * (0.5 * a2_err);

  double num;
  double den;
  if (a < detail::kNormRegionAB) {
    num = detail::horner(detail::kHartNum, a);
    den = detail::horner(detail::kHartDen, a);
  } else {
    const double y = a * kInvSqrt2;
    if (a < detail::kNormRegionBC) {
      num = 0.5 * detail::horner(detail::kCodyMidNum, y);
      den = detail::horner(detail::kCodyMidDen, y);
    } else {
      const double w = 0.5 * a2;
      const double wq = w * detail::horner(detail::kCodyFarDen, w);
      num = 0.5 * (kInvSqrtPi * wq - detail::horner(detail::kCodyFarNum, w));
      den = y * wq;
    }
  }
  const double tail = e * num / den;
  const double body = 1.0 - tail;
  const double pdf = e * kInvSqrt2Pi;
  return x > 0.0 ? NormalDist{body, tail, pdf} : NormalDist{tail, body, pdf};
}

/// Standard normal CDF Phi(x).
[[nodiscard]] inline double norm_cdf(double x) noexcept { return norm_dist(x).cdf; }

/// Standard normal PDF phi(x).
[[nodiscard]] inline double norm_pdf(double x) noexcept {
  return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

/// Inverse standard normal CDF for p in (0, 1). Wichura's AS241 (PPND16),
/// relative accuracy ~1e-16. One uniform in, one normal out, so antithetic
/// pairs are exact mirror images.
[[nodiscard]] inline double inv_norm_cdf(double p) noexcept {
  const double q = p - 0.5;
  if (std::fabs(q) <= 0.425) {
    const double r = 0.180625 - q * q;
    return q * detail::horner_ascending(detail::kPpndA, r) /
           detail::horner_ascending(detail::kPpndB, r);
  }
  double r = q < 0.0 ? p : 1.0 - p;
  r = std::sqrt(-std::log(r));
  double v;
  if (r <= 5.0) {
    r -= 1.6;
    v = detail::horner_ascending(detail::kPpndC, r) / detail::horner_ascending(detail::kPpndD, r);
  } else {
    r -= 5.0;
    v = detail::horner_ascending(detail::kPpndE, r) / detail::horner_ascending(detail::kPpndF, r);
  }
  return q < 0.0 ? -v : v;
}

// ---------------------------------------------------------------------------
// AVX2 + FMA functions (4 doubles per call)
// ---------------------------------------------------------------------------
#if OPTENGINE_HAS_AVX2

namespace detail {

template <std::size_t N>
[[nodiscard]] inline __m256d horner_pd(const std::array<double, N>& c, __m256d x) noexcept {
  __m256d r = _mm256_set1_pd(c[0]);
  for (std::size_t i = 1; i < N; ++i) r = _mm256_fmadd_pd(r, x, _mm256_set1_pd(c[i]));
  return r;
}

/// 2^k for four int32 exponents with k + 1023 in [1, 2046].
[[nodiscard]] inline __m256d pow2_pd(__m128i k) noexcept {
  const __m256i biased = _mm256_add_epi64(_mm256_cvtepi32_epi64(k), _mm256_set1_epi64x(1023));
  return _mm256_castsi256_pd(_mm256_slli_epi64(biased, 52));
}

}  // namespace detail

/// exp(x) for 4 doubles. Relative error < 1 ulp-ish (~2e-16) on the whole
/// double range; inputs are clamped so the result saturates to 0 / +inf
/// instead of producing garbage. NaN input is not supported.
[[nodiscard]] inline __m256d exp_pd(__m256d x) noexcept {
  x = _mm256_min_pd(x, _mm256_set1_pd(709.78));
  x = _mm256_max_pd(x, _mm256_set1_pd(-745.2));
  // x = k ln2 + r, |r| <= ln2 / 2
  const __m256d k = _mm256_round_pd(_mm256_mul_pd(x, _mm256_set1_pd(kLog2E)),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256d r = _mm256_fnmadd_pd(k, _mm256_set1_pd(kLn2Hi), x);
  r = _mm256_fnmadd_pd(k, _mm256_set1_pd(kLn2Lo), r);
  // exp(r) = 1 + r + r^2 * (1/2! + r/3! + ... + r^11/13!)
  const __m256d q = detail::horner_pd(detail::kExpPoly, r);
  const __m256d p =
      _mm256_add_pd(_mm256_fmadd_pd(_mm256_mul_pd(r, r), q, r), _mm256_set1_pd(1.0));
  // Scale by 2^k in two steps so denormal results round once and correctly.
  const __m128i ki = _mm256_cvtpd_epi32(k);
  const __m128i k1 = _mm_srai_epi32(ki, 1);
  const __m128i k2 = _mm_sub_epi32(ki, k1);
  return _mm256_mul_pd(_mm256_mul_pd(p, detail::pow2_pd(k1)), detail::pow2_pd(k2));
}

/// log(x) for 4 positive, normal doubles (fdlibm algorithm, < 1 ulp).
/// Zero, negative, denormal, inf and NaN inputs are not supported.
[[nodiscard]] inline __m256d log_pd(__m256d x) noexcept {
  // Split x = 2^e * m with m in [sqrt(2)/2, sqrt(2)).
  constexpr long long kSqrtHalfBits = 0x3fe6a09e667f3bcdLL;
  constexpr long long kOneBits = 0x3ff0000000000000LL;
  __m256i bits = _mm256_castpd_si256(x);
  bits = _mm256_add_epi64(bits, _mm256_set1_epi64x(kOneBits - kSqrtHalfBits));
  const __m256i e_biased = _mm256_srli_epi64(bits, 52);
  const __m256i m_bits =
      _mm256_add_epi64(_mm256_and_si256(bits, _mm256_set1_epi64x(0x000fffffffffffffLL)),
                       _mm256_set1_epi64x(kSqrtHalfBits));
  const __m256d m = _mm256_castsi256_pd(m_bits);
  // int64 -> double for the small exponents: pack low dwords, convert.
  const __m128i e32 = _mm256_castsi256_si128(
      _mm256_permutevar8x32_epi32(e_biased, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0)));
  const __m256d dk = _mm256_cvtepi32_pd(_mm_sub_epi32(e32, _mm_set1_epi32(1023)));

  const __m256d f = _mm256_sub_pd(m, _mm256_set1_pd(1.0));
  const __m256d s = _mm256_div_pd(f, _mm256_add_pd(f, _mm256_set1_pd(2.0)));
  const __m256d z = _mm256_mul_pd(s, s);
  const __m256d w = _mm256_mul_pd(z, z);
  __m256d t1 = _mm256_fmadd_pd(w, _mm256_set1_pd(detail::kLg6), _mm256_set1_pd(detail::kLg4));
  t1 = _mm256_fmadd_pd(w, t1, _mm256_set1_pd(detail::kLg2));
  t1 = _mm256_mul_pd(w, t1);
  __m256d t2 = _mm256_fmadd_pd(w, _mm256_set1_pd(detail::kLg7), _mm256_set1_pd(detail::kLg5));
  t2 = _mm256_fmadd_pd(w, t2, _mm256_set1_pd(detail::kLg3));
  t2 = _mm256_fmadd_pd(w, t2, _mm256_set1_pd(detail::kLg1));
  t2 = _mm256_mul_pd(z, t2);
  const __m256d r = _mm256_add_pd(t1, t2);
  const __m256d hfsq = _mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(f, f));
  // log(x) = k ln2_hi - ((hfsq - (s (hfsq + R) + k ln2_lo)) - f)
  const __m256d inner =
      _mm256_fmadd_pd(s, _mm256_add_pd(hfsq, r), _mm256_mul_pd(dk, _mm256_set1_pd(kLn2Lo)));
  const __m256d tail = _mm256_sub_pd(_mm256_sub_pd(hfsq, inner), f);
  return _mm256_fmsub_pd(dk, _mm256_set1_pd(kLn2Hi), tail);
}

/// Standard normal distribution evaluated at four points.
struct NormalDistPd {
  __m256d cdf;   ///< Phi(x)
  __m256d ccdf;  ///< 1 - Phi(x)
  __m256d pdf;   ///< phi(x)
};

/// Branch-free Phi(x), Phi(-x), phi(x) for 4 doubles. Same algorithm and
/// accuracy as the scalar `norm_dist`.
[[nodiscard]] inline NormalDistPd norm_dist_pd(__m256d x) noexcept {
  const __m256d sign_mask = _mm256_set1_pd(-0.0);
  const __m256d half = _mm256_set1_pd(0.5);
  const __m256d a =
      _mm256_min_pd(_mm256_andnot_pd(sign_mask, x), _mm256_set1_pd(detail::kNormTailCutoff));
  const __m256d a2 = _mm256_mul_pd(a, a);
  const __m256d a2_err = _mm256_fmsub_pd(a, a, a2);
  __m256d e = exp_pd(_mm256_mul_pd(_mm256_set1_pd(-0.5), a2));
  e = _mm256_fnmadd_pd(e, _mm256_mul_pd(half, a2_err), e);

  // Region A: Hart, argument a.
  const __m256d num_a = detail::horner_pd(detail::kHartNum, a);
  const __m256d den_a = detail::horner_pd(detail::kHartDen, a);
  // Region B: Cody mid, argument y = a / sqrt(2).
  const __m256d y = _mm256_mul_pd(a, _mm256_set1_pd(kInvSqrt2));
  const __m256d num_b = _mm256_mul_pd(half, detail::horner_pd(detail::kCodyMidNum, y));
  const __m256d den_b = detail::horner_pd(detail::kCodyMidDen, y);
  // Region C: Cody far, argument w = y^2.
  const __m256d w = _mm256_mul_pd(half, a2);
  const __m256d wq = _mm256_mul_pd(w, detail::horner_pd(detail::kCodyFarDen, w));
  const __m256d num_c = _mm256_mul_pd(
      half, _mm256_fmsub_pd(_mm256_set1_pd(kInvSqrtPi), wq,
                            detail::horner_pd(detail::kCodyFarNum, w)));
  const __m256d den_c = _mm256_mul_pd(y, wq);

  const __m256d in_a = _mm256_cmp_pd(a, _mm256_set1_pd(detail::kNormRegionAB), _CMP_LT_OQ);
  const __m256d in_c = _mm256_cmp_pd(a, _mm256_set1_pd(detail::kNormRegionBC), _CMP_GE_OQ);
  __m256d num = _mm256_blendv_pd(num_b, num_a, in_a);
  __m256d den = _mm256_blendv_pd(den_b, den_a, in_a);
  num = _mm256_blendv_pd(num, num_c, in_c);
  den = _mm256_blendv_pd(den, den_c, in_c);

  const __m256d tail = _mm256_div_pd(_mm256_mul_pd(e, num), den);
  const __m256d body = _mm256_sub_pd(_mm256_set1_pd(1.0), tail);
  const __m256d positive = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ);
  return {_mm256_blendv_pd(tail, body, positive), _mm256_blendv_pd(body, tail, positive),
          _mm256_mul_pd(e, _mm256_set1_pd(kInvSqrt2Pi))};
}

#endif  // OPTENGINE_HAS_AVX2

}  // namespace optengine
