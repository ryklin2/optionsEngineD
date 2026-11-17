#include "optengine/bsm_simd.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "optengine/bsm_scalar.hpp"

namespace optengine {

std::string_view bsm_simd_backend() noexcept {
  return kHasAvx2 ? std::string_view{"avx2+fma"} : std::string_view{"scalar-fallback"};
}

#if OPTENGINE_HAS_AVX2

namespace {

constexpr std::size_t kVectorBytes = 32;

[[nodiscard]] bool is_vector_aligned(const double* p) noexcept {
  return (reinterpret_cast<std::uintptr_t>(p) % kVectorBytes) == 0;
}

struct AlignedIo {
  [[nodiscard]] static __m256d load(const double* p) noexcept { return _mm256_load_pd(p); }
  static void store(double* p, __m256d v) noexcept { _mm256_store_pd(p, v); }
};
struct UnalignedIo {
  [[nodiscard]] static __m256d load(const double* p) noexcept { return _mm256_loadu_pd(p); }
  static void store(double* p, __m256d v) noexcept { _mm256_storeu_pd(p, v); }
};

/// Expands four is_call bytes into a 4x64-bit "is put" lane mask.
[[nodiscard]] __m256d put_mask(const std::uint8_t* flags) noexcept {
  std::uint32_t packed;
  std::memcpy(&packed, flags, sizeof(packed));
  const __m256i wide = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(static_cast<int>(packed)));
  return _mm256_castsi256_pd(_mm256_cmpeq_epi64(wide, _mm256_setzero_si256()));
}

template <typename Io>
void price_blocks(const ChainView& in, const GreeksView& out, std::size_t blocks) noexcept {
  const double* spot = in.spot.data();
  const double* strike = in.strike.data();
  const double* rate = in.rate.data();
  const double* dividend = in.dividend.data();
  const double* expiry = in.time_to_expiry.data();
  const double* vol = in.volatility.data();
  const std::uint8_t* is_call = in.is_call.data();
  double* price = out.price.data();
  double* delta = out.delta.data();
  double* gamma = out.gamma.data();
  double* vega = out.vega.data();
  double* theta = out.theta.data();
  double* rho = out.rho.data();

  for (std::size_t b = 0; b < blocks; ++b) {
    const std::size_t i = b * kSimdWidth;
    const GreeksPd g =
        bsm_kernel_avx2(Io::load(spot + i), Io::load(strike + i), Io::load(rate + i),
                        Io::load(dividend + i), Io::load(expiry + i), Io::load(vol + i),
                        put_mask(is_call + i));
    Io::store(price + i, g.price);
    Io::store(delta + i, g.delta);
    Io::store(gamma + i, g.gamma);
    Io::store(vega + i, g.vega);
    Io::store(theta + i, g.theta);
    Io::store(rho + i, g.rho);
  }
}

/// Prices the last `count` (1..3) options through the vector kernel using
/// padded stack buffers. Padding lanes hold a benign at-the-money option.
void price_tail(const ChainView& in, const GreeksView& out, std::size_t first,
                std::size_t count) noexcept {
  alignas(kVectorBytes) double spot[kSimdWidth] = {100.0, 100.0, 100.0, 100.0};
  alignas(kVectorBytes) double strike[kSimdWidth] = {100.0, 100.0, 100.0, 100.0};
  alignas(kVectorBytes) double rate[kSimdWidth] = {};
  alignas(kVectorBytes) double dividend[kSimdWidth] = {};
  alignas(kVectorBytes) double expiry[kSimdWidth] = {1.0, 1.0, 1.0, 1.0};
  alignas(kVectorBytes) double vol[kSimdWidth] = {0.2, 0.2, 0.2, 0.2};
  std::uint8_t is_call[kSimdWidth] = {1, 1, 1, 1};
  for (std::size_t j = 0; j < count; ++j) {
    spot[j] = in.spot[first + j];
    strike[j] = in.strike[first + j];
    rate[j] = in.rate[first + j];
    dividend[j] = in.dividend[first + j];
    expiry[j] = in.time_to_expiry[first + j];
    vol[j] = in.volatility[first + j];
    is_call[j] = in.is_call[first + j];
  }
  const GreeksPd g = bsm_kernel_avx2(_mm256_load_pd(spot), _mm256_load_pd(strike),
                                     _mm256_load_pd(rate), _mm256_load_pd(dividend),
                                     _mm256_load_pd(expiry), _mm256_load_pd(vol),
                                     put_mask(is_call));
  alignas(kVectorBytes) double price[kSimdWidth];
  alignas(kVectorBytes) double delta[kSimdWidth];
  alignas(kVectorBytes) double gamma[kSimdWidth];
  alignas(kVectorBytes) double vega[kSimdWidth];
  alignas(kVectorBytes) double theta[kSimdWidth];
  alignas(kVectorBytes) double rho[kSimdWidth];
  _mm256_store_pd(price, g.price);
  _mm256_store_pd(delta, g.delta);
  _mm256_store_pd(gamma, g.gamma);
  _mm256_store_pd(vega, g.vega);
  _mm256_store_pd(theta, g.theta);
  _mm256_store_pd(rho, g.rho);
  for (std::size_t j = 0; j < count; ++j) {
    out.price[first + j] = price[j];
    out.delta[first + j] = delta[j];
    out.gamma[first + j] = gamma[j];
    out.vega[first + j] = vega[j];
    out.theta[first + j] = theta[j];
    out.rho[first + j] = rho[j];
  }
}

}  // namespace

void bsm_simd_chain(ChainView in, GreeksView out) noexcept {
  const std::size_t n = in.size();
  assert(in.strike.size() == n && in.rate.size() == n && in.dividend.size() == n &&
         in.time_to_expiry.size() == n && in.volatility.size() == n && in.is_call.size() == n);
  assert(out.price.size() >= n && out.delta.size() >= n && out.gamma.size() >= n &&
         out.vega.size() >= n && out.theta.size() >= n && out.rho.size() >= n);

  const std::size_t blocks = n / kSimdWidth;
  const bool aligned =
      is_vector_aligned(in.spot.data()) && is_vector_aligned(in.strike.data()) &&
      is_vector_aligned(in.rate.data()) && is_vector_aligned(in.dividend.data()) &&
      is_vector_aligned(in.time_to_expiry.data()) && is_vector_aligned(in.volatility.data()) &&
      is_vector_aligned(out.price.data()) && is_vector_aligned(out.delta.data()) &&
      is_vector_aligned(out.gamma.data()) && is_vector_aligned(out.vega.data()) &&
      is_vector_aligned(out.theta.data()) && is_vector_aligned(out.rho.data());
  if (aligned) {
    price_blocks<AlignedIo>(in, out, blocks);
  } else {
    price_blocks<UnalignedIo>(in, out, blocks);
  }
  const std::size_t done = blocks * kSimdWidth;
  if (done < n) price_tail(in, out, done, n - done);
}

#else  // !OPTENGINE_HAS_AVX2

void bsm_simd_chain(ChainView in, GreeksView out) noexcept { bsm_scalar_chain(in, out); }

#endif

}  // namespace optengine
