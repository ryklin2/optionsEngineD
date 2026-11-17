// Analytical accuracy, put-call parity and scalar-vs-SIMD parity tests.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "alloc_hook.hpp"
#include "optengine/bsm_scalar.hpp"
#include "optengine/bsm_simd.hpp"
#include "optengine/math_utils.hpp"
#include "optengine/types.hpp"
#include "test_util.hpp"

namespace optengine {
namespace {

using testing::AllocationCounter;
using testing::make_random_chain;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
TEST(Types, SoaArraysAreCacheLineAligned) {
  for (std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{1023}, std::size_t{10000}}) {
    OptionChain chain(n);
    GreeksBuffer out(n);
    const auto aligned = [](const void* p) {
      return reinterpret_cast<std::uintptr_t>(p) % kCacheLineBytes == 0;
    };
    EXPECT_TRUE(aligned(chain.spot.data()));
    EXPECT_TRUE(aligned(chain.strike.data()));
    EXPECT_TRUE(aligned(chain.rate.data()));
    EXPECT_TRUE(aligned(chain.dividend.data()));
    EXPECT_TRUE(aligned(chain.time_to_expiry.data()));
    EXPECT_TRUE(aligned(chain.volatility.data()));
    EXPECT_TRUE(aligned(chain.is_call.data()));
    EXPECT_TRUE(aligned(out.price.data()));
    EXPECT_TRUE(aligned(out.delta.data()));
    EXPECT_TRUE(aligned(out.gamma.data()));
    EXPECT_TRUE(aligned(out.vega.data()));
    EXPECT_TRUE(aligned(out.theta.data()));
    EXPECT_TRUE(aligned(out.rho.data()));
    EXPECT_EQ(chain.view().size(), n);
    EXPECT_EQ(out.view().size(), n);
  }
  EXPECT_EQ(alignof(OptionChain), kCacheLineBytes);
  EXPECT_EQ(alignof(GreeksBuffer), kCacheLineBytes);
}

// ---------------------------------------------------------------------------
// Normal distribution
// ---------------------------------------------------------------------------

// Phi(x) to 20 significant digits (mpmath, 50-digit arithmetic).
struct PhiRef {
  double x;
  double phi;
};
constexpr PhiRef kPhiRefs[] = {
    {-37.5, 4.6053530095819548438e-308},
    {-20.0, 2.7536241186062336951e-89},
    {-10.0, 7.619853024160526066e-24},
    {-7.5, 3.1908916729108962278e-14},
    {-7.0710678118654755, 7.6872989721401600678e-13},
    {-5.65685424949238, 7.7086289501400181899e-9},
    {-5.0, 2.8665157187919391167e-7},
    {-3.0, 0.0013498980316300945267},
    {-1.0, 0.15865525393145705141},
    {-0.75, 0.22662735237686819933},
    {-0.5, 0.30853753872598689636},
    {-1e-9, 0.4999999996010577196},
    {0.0, 0.5},
    {0.25, 0.59870632568292372424},
    {0.75, 0.77337264762313180067},
    {1.0, 0.84134474606854294859},
    {2.0, 0.9772498680518207928},
    {3.5, 0.99976737092096447496},
    {6.0, 0.99999999901341235496},
    {8.5, 0.99999999999999999052},
    // Random lower-tail points (relative accuracy of the tail).
    {-37.36793, 6.4848971740162456995e-306},
    {-37.090968, 1.9642722377450503515e-301},
    {-36.012457, 2.669960803138475062e-284},
    {-33.795966, 1.1302747131473093471e-250},
    {-33.770599, 2.6649646783517366138e-250},
    {-31.163183, 1.6810767097270601867e-213},
    {-26.537263, 1.8015046403890872547e-155},
    {-26.039861, 8.7630771888378982152e-150},
    {-24.919029, 2.3140062851112688619e-137},
    {-24.509746, 5.8143309991562873897e-133},
    {-24.220758, 6.7238242027145992674e-130},
    {-23.422181, 1.270077913528391509e-121},
    {-22.000545, 1.422695900773751195e-107},
    {-18.910964, 4.6323782461094748838e-80},
    {-18.815363, 2.826001017949863557e-79},
    {-17.496122, 7.6681133921227810679e-69},
    {-17.275273, 3.611541596730110752e-67},
    {-15.60203, 3.5257873649814259926e-55},
    {-14.549975, 2.9212198988245609826e-48},
    {-14.481149, 7.9704977049123571486e-48},
    {-13.05454, 2.9935281846909307662e-39},
    {-10.417043, 1.0364910852130234141e-25},
    {-10.107527, 2.5579665306998681114e-24},
    {-9.705828, 1.4237738990088754129e-22},
    {-9.217933, 1.514411589540932261e-20},
    {-8.942776, 1.8975724563184581705e-19},
    {-8.305833, 4.9560783053403594248e-17},
    {-7.674095, 8.329529347727768347e-15},
    {-5.830678, 2.760130876980444047e-9},
    {-5.248977, 7.6473083854410483429e-8},
    {-4.785298, 8.5366911518698840355e-7},
    {-3.896015, 0.000048894167596621226832},
    {-3.734818, 0.000093925494611869678101},
    {-2.968945, 0.0014941203837566560637},
    {-2.921663, 0.0017408401914149098513},
    {-1.33568, 0.090826944594192129254},
};

TEST(NormalDistribution, CdfMatchesHighPrecisionReferences) {
  for (const PhiRef& ref : kPhiRefs) {
    const NormalDist n = norm_dist(ref.x);
    EXPECT_NEAR(n.cdf, ref.phi, 1e-15) << "x = " << ref.x;
    // The reference values carry relative precision too (tails included).
    EXPECT_LE(std::fabs(n.cdf - ref.phi), 4e-15 * ref.phi) << "x = " << ref.x;
    EXPECT_EQ(norm_cdf(ref.x), n.cdf);
  }
}

TEST(NormalDistribution, CdfAbsoluteErrorBelow1e15OnDenseGrid) {
  // glibc's erfc is accurate to < 1 ulp, so 0.5 * erfc(-x / sqrt 2) is an
  // independent reference good to ~1e-16.
  double worst_abs = 0.0;
  double worst_tail_rel = 0.0;
  for (int i = -400000; i <= 400000; ++i) {
    const double x = static_cast<double>(i) * 1e-4;  // [-40, 40], step 1e-4
    const NormalDist n = norm_dist(x);
    const double ref_cdf = 0.5 * std::erfc(-x * kInvSqrt2);
    const double ref_ccdf = 0.5 * std::erfc(x * kInvSqrt2);
    worst_abs = std::max({worst_abs, std::fabs(n.cdf - ref_cdf), std::fabs(n.ccdf - ref_ccdf)});
    if (std::fabs(x) < 37.0) {
      // Rounding x / sqrt(2) perturbs the erfc reference itself by ~x^2 ulp,
      // so the relative check is normalised by that conditioning. The sharp
      // tail check is CdfMatchesHighPrecisionReferences.
      const double tail = std::min(n.cdf, n.ccdf);
      const double ref_tail = std::min(ref_cdf, ref_ccdf);
      const double conditioning = 1.0 + x * x;
      worst_tail_rel =
          std::max(worst_tail_rel, std::fabs(tail - ref_tail) / (ref_tail * conditioning));
    }
  }
  RecordProperty("max_abs_error", std::to_string(worst_abs));
  EXPECT_LT(worst_abs, 1e-15);
  EXPECT_LT(worst_tail_rel, 1e-15);
}

TEST(NormalDistribution, CdfComplementAndSymmetry) {
  std::mt19937_64 gen(7);
  std::uniform_real_distribution<double> dist(-12.0, 12.0);
  for (int i = 0; i < 20000; ++i) {
    const double x = dist(gen);
    const NormalDist n = norm_dist(x);
    const NormalDist m = norm_dist(-x);
    EXPECT_EQ(n.cdf, m.ccdf);
    EXPECT_EQ(n.ccdf, m.cdf);
    EXPECT_EQ(n.pdf, m.pdf);
    EXPECT_NEAR(n.cdf + n.ccdf, 1.0, 2e-16);
    // norm_pdf() squares x without error compensation: allow x^2 ulp.
    EXPECT_NEAR(n.pdf, norm_pdf(x), (2.0 + x * x) * 2.3e-16 * norm_pdf(x));
    EXPECT_GE(n.cdf, 0.0);
    EXPECT_LE(n.cdf, 1.0);
  }
  // Saturation far in the tails.
  EXPECT_EQ(norm_dist(50.0).cdf, 1.0);
  EXPECT_EQ(norm_dist(50.0).ccdf, 0.0);
  EXPECT_EQ(norm_dist(-50.0).cdf, 0.0);
  EXPECT_EQ(norm_dist(1e10).pdf, 0.0);
}

TEST(NormalDistribution, InverseCdfRoundTrip) {
  EXPECT_EQ(inv_norm_cdf(0.5), 0.0);
  EXPECT_NEAR(inv_norm_cdf(0.975), 1.9599639845400542355, 1e-14);
  EXPECT_NEAR(inv_norm_cdf(0.001), -3.0902323061678135415, 1e-14);
  std::mt19937_64 gen(11);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (int i = 0; i < 20000; ++i) {
    const double p = std::max(dist(gen), 1e-300);
    const double z = inv_norm_cdf(p);
    ASSERT_TRUE(is_finite(z));
    const NormalDist n = norm_dist(z);
    // |Phi(z) - p| translated into z-space through the density.
    EXPECT_LE(std::fabs(n.cdf - p), 4e-15 * std::max(1.0, std::fabs(z)) * n.pdf + 1e-16)
        << "p = " << p;
  }
  // Odd symmetry about p = 1/2, on probabilities where 1/2 +- q is exact.
  for (int k = 1; k < 1024; ++k) {
    const double q = static_cast<double>(k) / 2048.0;
    EXPECT_EQ(inv_norm_cdf(0.5 + q), -inv_norm_cdf(0.5 - q)) << "q = " << q;
  }
  // Deep tails stay finite and monotone.
  double prev = inv_norm_cdf(1e-300);
  for (double p : {1e-200, 1e-100, 1e-50, 1e-20, 1e-10, 1e-5}) {
    const double z = inv_norm_cdf(p);
    EXPECT_GT(z, prev);
    prev = z;
  }
}

#if OPTENGINE_HAS_AVX2
std::array<double, 4> to_array(__m256d v) {
  alignas(32) std::array<double, 4> a{};
  _mm256_store_pd(a.data(), v);
  return a;
}

TEST(SimdMath, ExpMatchesLibm) {
  std::mt19937_64 gen(3);
  std::uniform_real_distribution<double> dist(-700.0, 700.0);
  double worst = 0.0;
  for (int i = 0; i < 50000; ++i) {
    alignas(32) double x[4] = {dist(gen), dist(gen) * 0.01, dist(gen) * 1e-4, dist(gen) * 0.1};
    const auto r = to_array(exp_pd(_mm256_load_pd(x)));
    for (std::size_t j = 0; j < 4; ++j) {
      const double ref = std::exp(x[j]);
      worst = std::max(worst, std::fabs(r[j] - ref) / ref);
    }
  }
  EXPECT_LT(worst, 4.5e-16);  // < 2 ulp
  alignas(32) double edge[4] = {0.0, -800.0, -1e300, 1.0};
  const auto r = to_array(exp_pd(_mm256_load_pd(edge)));
  EXPECT_EQ(r[0], 1.0);
  EXPECT_EQ(r[1], 0.0);
  EXPECT_EQ(r[2], 0.0);
  EXPECT_NEAR(r[3], 2.718281828459045, 5e-16);
}

TEST(SimdMath, LogMatchesLibm) {
  std::mt19937_64 gen(4);
  std::uniform_real_distribution<double> mantissa(0.5, 2.0);
  std::uniform_int_distribution<int> exponent(-300, 300);
  double worst = 0.0;
  for (int i = 0; i < 50000; ++i) {
    alignas(32) double x[4] = {mantissa(gen), mantissa(gen) * 1e-3, mantissa(gen) * 1e3,
                               std::ldexp(mantissa(gen), exponent(gen))};
    const auto r = to_array(log_pd(_mm256_load_pd(x)));
    for (std::size_t j = 0; j < 4; ++j) {
      const double ref = std::log(x[j]);
      worst = std::max(worst, std::fabs(r[j] - ref) / std::max(std::fabs(ref), 1e-300));
    }
  }
  EXPECT_LT(worst, 4.5e-16);
  alignas(32) double one[4] = {1.0, 1.0, 1.0, 1.0};
  EXPECT_EQ(to_array(log_pd(_mm256_load_pd(one)))[0], 0.0);
}

TEST(SimdMath, NormDistMatchesScalar) {
  std::mt19937_64 gen(5);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  double worst_abs = 0.0;
  double worst_tail_rel = 0.0;
  for (int i = 0; i < 100000; ++i) {
    alignas(32) double x[4] = {dist(gen), 3.0 * dist(gen), 8.0 * dist(gen), 37.0 * dist(gen)};
    if (i == 0) {
      x[0] = 0.0; x[1] = 0.75; x[2] = -5.65685424949238019521; x[3] = 1e6;
    }
    const NormalDistPd v = norm_dist_pd(_mm256_load_pd(x));
    const auto cdf = to_array(v.cdf);
    const auto ccdf = to_array(v.ccdf);
    const auto pdf = to_array(v.pdf);
    for (std::size_t j = 0; j < 4; ++j) {
      const NormalDist s = norm_dist(x[j]);
      ASSERT_TRUE(is_finite(cdf[j]) && is_finite(ccdf[j]) && is_finite(pdf[j])) << x[j];
      worst_abs = std::max({worst_abs, std::fabs(cdf[j] - s.cdf), std::fabs(ccdf[j] - s.ccdf),
                            std::fabs(pdf[j] - s.pdf)});
      const double tail = std::min(s.cdf, s.ccdf);
      if (tail > 1e-300) {
        worst_tail_rel =
            std::max(worst_tail_rel, std::fabs(std::min(cdf[j], ccdf[j]) - tail) / tail);
      }
      // And against the independent libm reference.
      EXPECT_NEAR(cdf[j], 0.5 * std::erfc(-x[j] * kInvSqrt2), 1e-15);
    }
  }
  EXPECT_LT(worst_abs, 5e-16);
  EXPECT_LT(worst_tail_rel, 5e-15);
}
#endif  // OPTENGINE_HAS_AVX2

// ---------------------------------------------------------------------------
// Textbook reference values
// ---------------------------------------------------------------------------
//
// Worked examples from J. C. Hull, "Options, Futures, and Other Derivatives".
// Hull prints 2-4 significant figures; to test at 1e-12 the same contracts
// were re-priced with mpmath in 50-digit arithmetic (price, delta, gamma,
// vega, theta per year, rho). `printed_*` are the figures as printed in the
// book and are checked to the precision Hull quotes.
struct HullCase {
  const char* name;
  OptionParams option;  // is_call is overridden per leg
  Greeks call;
  Greeks put;
};

const HullCase kHullCases[] = {
    {"stock option: S=42 K=40 r=10% vol=20% T=6m (c=4.76, p=0.81)",
     {42.0, 40.0, 0.10, 0.0, 0.5, 0.20, true},
     {4.7594223928715332196, 0.77913129094266894042, 0.049962670405911855657,
      8.8134150596028513379, -4.5590921945926264954, 13.982045913360281139},
     {0.80859937290009358326, -0.22086870905733105958, 0.049962670405911855657,
      8.8134150596028513379, -0.75417449658977045902, -5.0425425766539990428}},
    {"Greek letters: S=49 K=50 r=5% vol=20% T=20 weeks",
     {49.0, 50.0, 0.05, 0.0, 20.0 / 52.0, 0.20, true},
     {2.4005273232717144929, 0.52160466106639638876, 0.06554403934784439766,
      12.105479882628799906, -4.3053298229325734034, 8.9069619496083494447},
     {2.4481754412818494915, -0.47839533893360361124, 0.06554403934784439766,
      12.105479882628799906, -1.8529474170320666535, -9.9575180957801640163}},
    {"index option: S=930 K=900 r=8% q=3% vol=20% T=2m (c=51.83)",
     {930.0, 900.0, 0.08, 0.03, 2.0 / 12.0, 0.20, true},
     {51.832956796490848896, 0.70341800860119177858, 0.0045074038616943379059,
      129.94845333264776183, -106.53137285582480689, 100.39096520043625086},
     {14.550996773772443245, -0.29159447059149053478, 0.0045074038616943379059,
      129.94845333264776183, -63.245849375182551769, -47.622309070643106764}},
    {"currency option: S=1.6 K=1.6 r=8% rf=11% vol=14.1% T=4m (c=0.043)",
     {1.6, 1.6, 0.08, 0.11, 4.0 / 12.0, 0.141, true},
     {0.042957730192595755173, 0.45044588657627712262, 2.9426761920544325521,
      0.35406279942798932467, -0.049826261107950779867, 0.22591856277648254701},
     {0.05845906632400323855, -0.51355152769473820232, 2.9426761920544325521,
      0.35406279942798932467, -0.094858030102446916794, -0.29338050354519478742}},
};

void expect_greeks_near(const Greeks& actual, const Greeks& expected, double tol,
                        const char* name, const char* leg) {
  EXPECT_NEAR(actual.price, expected.price, tol) << name << " [" << leg << "] price";
  EXPECT_NEAR(actual.delta, expected.delta, tol) << name << " [" << leg << "] delta";
  EXPECT_NEAR(actual.gamma, expected.gamma, tol) << name << " [" << leg << "] gamma";
  EXPECT_NEAR(actual.vega, expected.vega, tol) << name << " [" << leg << "] vega";
  EXPECT_NEAR(actual.theta, expected.theta, tol) << name << " [" << leg << "] theta";
  EXPECT_NEAR(actual.rho, expected.rho, tol) << name << " [" << leg << "] rho";
}

TEST(BsmScalar, HullReferenceValuesWithin1e12) {
  for (const HullCase& c : kHullCases) {
    OptionParams o = c.option;
    o.is_call = true;
    expect_greeks_near(bsm_scalar_greeks(o), c.call, 1e-12, c.name, "call");
    EXPECT_NEAR(bsm_scalar_price(o), c.call.price, 1e-12) << c.name;
    o.is_call = false;
    expect_greeks_near(bsm_scalar_greeks(o), c.put, 1e-12, c.name, "put");
    EXPECT_NEAR(bsm_scalar_price(o), c.put.price, 1e-12) << c.name;
  }
}

TEST(BsmScalar, MatchesFiguresPrintedInHull) {
  // Example: European call 4.76, put 0.81.
  OptionParams o{42.0, 40.0, 0.10, 0.0, 0.5, 0.20, true};
  EXPECT_NEAR(bsm_scalar_price(o), 4.76, 0.005);
  o.is_call = false;
  EXPECT_NEAR(bsm_scalar_price(o), 0.81, 0.005);

  // Greek letters chapter: delta 0.522, gamma 0.066, vega 12.1,
  // theta -4.31 per year = -0.0118 per calendar day, rho 8.91.
  const Greeks g = bsm_scalar_greeks({49.0, 50.0, 0.05, 0.0, 20.0 / 52.0, 0.20, true});
  EXPECT_NEAR(g.delta, 0.522, 0.0005);
  EXPECT_NEAR(g.gamma, 0.066, 0.0005);
  EXPECT_NEAR(g.vega, 12.1, 0.05);
  EXPECT_NEAR(g.theta, -4.31, 0.005);
  EXPECT_NEAR(theta_per_calendar_day(g.theta), -0.0118, 0.00005);
  EXPECT_NEAR(g.rho, 8.91, 0.005);

  // Index option 51.83; currency option 4.3 cents.
  EXPECT_NEAR(bsm_scalar_price({930.0, 900.0, 0.08, 0.03, 2.0 / 12.0, 0.20, true}), 51.83, 0.005);
  EXPECT_NEAR(bsm_scalar_price({1.6, 1.6, 0.08, 0.11, 4.0 / 12.0, 0.141, true}), 0.043, 0.0005);
}

TEST(BsmScalar, GreeksMatchFiniteDifferences) {
  const OptionChain chain = make_random_chain(500, 21);
  for (std::size_t i = 0; i < chain.size(); ++i) {
    const OptionParams o = chain.get(i);
    const Greeks g = bsm_scalar_greeks(o);
    const auto bump = [&o](double OptionParams::* field, double h) {
      OptionParams up = o;
      OptionParams dn = o;
      up.*field += h;
      dn.*field -= h;
      return std::pair{bsm_scalar_price(up), bsm_scalar_price(dn)};
    };
    const double hs = 1e-4 * o.spot;
    const auto [s_up, s_dn] = bump(&OptionParams::spot, hs);
    EXPECT_NEAR(g.delta, (s_up - s_dn) / (2 * hs), 1e-6) << i;
    EXPECT_NEAR(g.gamma, (s_up - 2 * g.price + s_dn) / (hs * hs), 1e-5) << i;
    const auto [v_up, v_dn] = bump(&OptionParams::volatility, 1e-5);
    EXPECT_NEAR(g.vega, (v_up - v_dn) / 2e-5, 1e-5 * (1.0 + std::fabs(g.vega))) << i;
    const auto [t_up, t_dn] = bump(&OptionParams::time_to_expiry, 1e-6);
    EXPECT_NEAR(g.theta, -(t_up - t_dn) / 2e-6, 1e-5 * (1.0 + std::fabs(g.theta))) << i;
    const auto [r_up, r_dn] = bump(&OptionParams::rate, 1e-6);
    EXPECT_NEAR(g.rho, (r_up - r_dn) / 2e-6, 1e-5 * (1.0 + std::fabs(g.rho))) << i;
  }
}

TEST(BsmScalar, ChainMatchesSingleOptionApi) {
  // Same kernel, but the optimizer may contract multiply-adds into FMAs
  // differently once it is inlined into the chain loop (-ffp-contract=fast,
  // -ffast-math), so agreement is to a few ulp rather than bit-for-bit.
  const auto expect_ulp_close = [](double actual, double expected, std::size_t i) {
    EXPECT_NEAR(actual, expected, 8.0 * std::numeric_limits<double>::epsilon() *
                                      std::max(1.0, std::fabs(expected)))
        << "i = " << i;
  };
  const OptionChain chain = make_random_chain(257, 33);
  GreeksBuffer out(chain.size());
  bsm_scalar_chain(chain.view(), out.view());
  for (std::size_t i = 0; i < chain.size(); ++i) {
    const Greeks g = bsm_scalar_greeks(chain.get(i));
    expect_ulp_close(out.price[i], g.price, i);
    expect_ulp_close(out.delta[i], g.delta, i);
    expect_ulp_close(out.gamma[i], g.gamma, i);
    expect_ulp_close(out.vega[i], g.vega, i);
    expect_ulp_close(out.theta[i], g.theta, i);
    expect_ulp_close(out.rho[i], g.rho, i);
    expect_ulp_close(bsm_scalar_price(chain.get(i)), g.price, i);
  }
}

TEST(BsmSimd, HullReferenceValuesWithin1e11) {
  // Lay the cases out as a chain: call and put legs of every example, which
  // also exercises the non-multiple-of-4 tail when padded with one extra.
  const std::size_t n = 2 * std::size(kHullCases);
  OptionChain chain(n);
  for (std::size_t c = 0; c < std::size(kHullCases); ++c) {
    OptionParams o = kHullCases[c].option;
    o.is_call = true;
    chain.set(2 * c, o);
    o.is_call = false;
    chain.set(2 * c + 1, o);
  }
  GreeksBuffer out(n);
  bsm_simd_chain(chain.view(), out.view());
  for (std::size_t c = 0; c < std::size(kHullCases); ++c) {
    expect_greeks_near(out.get(2 * c), kHullCases[c].call, 1e-11, kHullCases[c].name, "call");
    expect_greeks_near(out.get(2 * c + 1), kHullCases[c].put, 1e-11, kHullCases[c].name, "put");
  }
}

// ---------------------------------------------------------------------------
// Put-call parity:  C - P = S e^{-qT} - K e^{-rT}
// ---------------------------------------------------------------------------
using ChainPricer = void (*)(ChainView, GreeksView) noexcept;

void check_put_call_parity(ChainPricer pricer, const char* name) {
  constexpr std::size_t kContracts = 10000;
  OptionChain calls = make_random_chain(kContracts, 2024);
  OptionChain puts = calls;
  std::fill(calls.is_call.begin(), calls.is_call.end(), std::uint8_t{1});
  std::fill(puts.is_call.begin(), puts.is_call.end(), std::uint8_t{0});
  GreeksBuffer c(kContracts);
  GreeksBuffer p(kContracts);
  pricer(calls.view(), c.view());
  pricer(puts.view(), p.view());

  double worst = 0.0;
  for (std::size_t i = 0; i < kContracts; ++i) {
    const double s = calls.spot[i];
    const double k = calls.strike[i];
    const double t = calls.time_to_expiry[i];
    const double disc_q = std::exp(-calls.dividend[i] * t);
    const double disc_r = std::exp(-calls.rate[i] * t);
    const double forward_value = s * disc_q - k * disc_r;
    const double scale = s + k;
    ASSERT_NEAR(c.price[i] - p.price[i], forward_value, 1e-13 * scale) << name << " i=" << i;
    worst = std::max(worst, std::fabs(c.price[i] - p.price[i] - forward_value) / scale);
    // Parity differentiated: delta, gamma, vega, theta, rho.
    ASSERT_NEAR(c.delta[i] - p.delta[i], disc_q, 1e-13) << name << " i=" << i;
    ASSERT_NEAR(c.gamma[i], p.gamma[i], 1e-13) << name << " i=" << i;
    ASSERT_NEAR(c.vega[i], p.vega[i], 1e-13 * scale) << name << " i=" << i;
    ASSERT_NEAR(c.theta[i] - p.theta[i],
                calls.dividend[i] * s * disc_q - calls.rate[i] * k * disc_r, 1e-13 * scale)
        << name << " i=" << i;
    ASSERT_NEAR(c.rho[i] - p.rho[i], t * k * disc_r, 1e-13 * scale * (1.0 + t))
        << name << " i=" << i;
    ASSERT_GE(c.price[i], 0.0);
    ASSERT_GE(p.price[i], 0.0);
  }
  ::testing::Test::RecordProperty(std::string("worst_relative_parity_gap_") + name,
                                  std::to_string(worst));
}

TEST(PutCallParity, ScalarRandomChainOf10000) { check_put_call_parity(&bsm_scalar_chain, "scalar"); }
TEST(PutCallParity, SimdRandomChainOf10000) { check_put_call_parity(&bsm_simd_chain, "simd"); }

// ---------------------------------------------------------------------------
// Scalar vs SIMD equivalence
// ---------------------------------------------------------------------------

/// Element-wise comparison with a tolerance of a few ulp of the operands that
/// produce each quantity (SIMD uses FMA polynomial exp/log instead of libm, so
/// agreement is to rounding, not bit-for-bit).
void expect_equivalent(const OptionChain& chain, const GreeksBuffer& scalar,
                       const GreeksBuffer& simd, std::size_t n) {
  constexpr double kRel = 5e-15;
  for (std::size_t i = 0; i < n; ++i) {
    const double notional = chain.spot[i] + chain.strike[i];
    const double t = chain.time_to_expiry[i];
    ASSERT_NEAR(simd.price[i], scalar.price[i], kRel * notional) << "price i=" << i;
    ASSERT_NEAR(simd.delta[i], scalar.delta[i], kRel) << "delta i=" << i;
    ASSERT_NEAR(simd.gamma[i], scalar.gamma[i], kRel * (1.0 + std::fabs(scalar.gamma[i])))
        << "gamma i=" << i;
    ASSERT_NEAR(simd.vega[i], scalar.vega[i], kRel * notional) << "vega i=" << i;
    ASSERT_NEAR(simd.theta[i], scalar.theta[i], kRel * (notional + std::fabs(scalar.theta[i])))
        << "theta i=" << i;
    ASSERT_NEAR(simd.rho[i], scalar.rho[i], kRel * notional * (1.0 + t)) << "rho i=" << i;
  }
}

TEST(BsmSimd, MatchesScalarAcrossTailSizes) {
  for (std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4},
                        std::size_t{5}, std::size_t{7}, std::size_t{8}, std::size_t{9},
                        std::size_t{1023}, std::size_t{1024}, std::size_t{10000}}) {
    SCOPED_TRACE("n = " + std::to_string(n));
    const OptionChain chain = make_random_chain(n, 1000 + n);
    GreeksBuffer scalar(n);
    GreeksBuffer simd(n);
    bsm_scalar_chain(chain.view(), scalar.view());
    bsm_simd_chain(chain.view(), simd.view());
    expect_equivalent(chain, scalar, simd, n);
  }
}

TEST(BsmSimd, EmptyChainIsANoOp) {
  OptionChain chain(0);
  GreeksBuffer out(0);
  bsm_simd_chain(chain.view(), out.view());
  bsm_scalar_chain(chain.view(), out.view());
  SUCCEED();
}

TEST(BsmSimd, RemainderLoopNeverWritesPastN) {
  constexpr double kSentinel = -12345.678;
  constexpr std::size_t kCapacity = 32;
  const OptionChain chain = make_random_chain(kCapacity, 77);
  for (std::size_t n = 0; n <= 17; ++n) {
    SCOPED_TRACE("n = " + std::to_string(n));
    GreeksBuffer out(kCapacity);
    for (auto* v : {&out.price, &out.delta, &out.gamma, &out.vega, &out.theta, &out.rho}) {
      std::fill(v->begin(), v->end(), kSentinel);
    }
    const ChainView in = chain.view();
    const ChainView head{in.spot.first(n),           in.strike.first(n),
                         in.rate.first(n),           in.dividend.first(n),
                         in.time_to_expiry.first(n), in.volatility.first(n),
                         in.is_call.first(n)};
    const GreeksView all = out.view();
    bsm_simd_chain(head, {all.price.first(n), all.delta.first(n), all.gamma.first(n),
                          all.vega.first(n), all.theta.first(n), all.rho.first(n)});
    for (std::size_t i = 0; i < kCapacity; ++i) {
      for (const auto* v : {&out.price, &out.delta, &out.gamma, &out.vega, &out.theta, &out.rho}) {
        if (i < n) {
          ASSERT_NE((*v)[i], kSentinel) << "slot " << i << " was not written";
        } else {
          ASSERT_EQ((*v)[i], kSentinel) << "slot " << i << " was clobbered";
        }
      }
    }
  }
}

TEST(BsmSimd, UnalignedSpansGiveIdenticalResults) {
  constexpr std::size_t kN = 203;
  const OptionChain chain = make_random_chain(kN + 1, 99);
  GreeksBuffer aligned(kN + 1);
  GreeksBuffer shifted(kN + 1);
  // Aligned run over elements [1, kN] copied into an aligned chain.
  OptionChain copy(kN);
  for (std::size_t i = 0; i < kN; ++i) copy.set(i, chain.get(i + 1));
  bsm_simd_chain(copy.view(), {std::span(aligned.price).first(kN), std::span(aligned.delta).first(kN),
                               std::span(aligned.gamma).first(kN), std::span(aligned.vega).first(kN),
                               std::span(aligned.theta).first(kN), std::span(aligned.rho).first(kN)});
  // Same contracts through spans that start 8 bytes off a 64-byte boundary.
  const ChainView in = chain.view();
  const ChainView off{in.spot.subspan(1),           in.strike.subspan(1),
                      in.rate.subspan(1),           in.dividend.subspan(1),
                      in.time_to_expiry.subspan(1), in.volatility.subspan(1),
                      in.is_call.subspan(1)};
  const GreeksView out = shifted.view();
  bsm_simd_chain(off, {out.price.subspan(1), out.delta.subspan(1), out.gamma.subspan(1),
                       out.vega.subspan(1), out.theta.subspan(1), out.rho.subspan(1)});
  for (std::size_t i = 0; i < kN; ++i) {
    ASSERT_EQ(shifted.price[i + 1], aligned.price[i]) << i;
    ASSERT_EQ(shifted.delta[i + 1], aligned.delta[i]) << i;
    ASSERT_EQ(shifted.gamma[i + 1], aligned.gamma[i]) << i;
    ASSERT_EQ(shifted.vega[i + 1], aligned.vega[i]) << i;
    ASSERT_EQ(shifted.theta[i + 1], aligned.theta[i]) << i;
    ASSERT_EQ(shifted.rho[i + 1], aligned.rho[i]) << i;
  }
}

TEST(BsmSimd, ExtremeMoneynessStaysFiniteAndBounded) {
  // Deep ITM / OTM, tiny and huge total volatility: |d1| from ~0 to > 1000.
  std::vector<OptionParams> cases;
  for (double moneyness : {0.01, 0.5, 0.9, 1.0, 1.1, 2.0, 100.0}) {
    for (double vol : {0.01, 0.2, 3.0}) {
      for (double t : {1.0 / 365.0, 1.0, 30.0}) {
        for (bool is_call : {true, false}) {
          cases.push_back({100.0 * moneyness, 100.0, 0.03, 0.01, t, vol, is_call});
        }
      }
    }
  }
  OptionChain chain(cases.size());
  for (std::size_t i = 0; i < cases.size(); ++i) chain.set(i, cases[i]);
  GreeksBuffer scalar(cases.size());
  GreeksBuffer simd(cases.size());
  bsm_scalar_chain(chain.view(), scalar.view());
  bsm_simd_chain(chain.view(), simd.view());
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const Greeks g = simd.get(i);
    for (double v : {g.price, g.delta, g.gamma, g.vega, g.theta, g.rho}) {
      ASSERT_TRUE(is_finite(v)) << "case " << i;
    }
    ASSERT_GE(g.price, 0.0) << "case " << i;
    ASSERT_LE(std::fabs(g.delta), 1.0) << "case " << i;
    ASSERT_GE(g.gamma, 0.0) << "case " << i;
    ASSERT_GE(g.vega, 0.0) << "case " << i;
  }
  expect_equivalent(chain, scalar, simd, cases.size());
}

TEST(BsmSimd, BackendMatchesBuildFlags) {
  if (kHasAvx2) {
    EXPECT_EQ(bsm_simd_backend(), "avx2+fma");
  } else {
    EXPECT_EQ(bsm_simd_backend(), "scalar-fallback");
  }
}

// ---------------------------------------------------------------------------
// Zero-allocation enforcement
// ---------------------------------------------------------------------------
TEST(ZeroAllocation, CounterDetectsAllocations) {
  const AllocationCounter counter;
  // Let the pointers escape so the optimizer cannot elide the new/delete pair.
  double* volatile escaped = new double[16];
  delete[] escaped;
  EXPECT_EQ(counter.allocations(), 1U) << "plain operator new[] is not counted";
  {
    AlignedVector<double> v(100);
    double* volatile data = v.data();
    EXPECT_NE(data, nullptr);
  }
  EXPECT_EQ(counter.allocations(), 2U) << "aligned operator new is not counted";
}

TEST(ZeroAllocation, BsmChainsDoNotAllocate) {
  for (std::size_t n : {std::size_t{3}, std::size_t{1023}, std::size_t{100000}}) {
    const OptionChain chain = make_random_chain(n, 5);
    GreeksBuffer out(n);
    const ChainView in = chain.view();
    const GreeksView view = out.view();
    {
      const AllocationCounter counter;
      bsm_simd_chain(in, view);
      EXPECT_EQ(counter.allocations(), 0U) << "bsm_simd_chain allocated, n = " << n;
    }
    {
      const AllocationCounter counter;
      bsm_scalar_chain(in, view);
      EXPECT_EQ(counter.allocations(), 0U) << "bsm_scalar_chain allocated, n = " << n;
    }
  }
}

}  // namespace
}  // namespace optengine
