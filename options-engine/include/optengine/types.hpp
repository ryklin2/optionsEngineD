// optengine/types.hpp
//
// Structure-of-Arrays (SoA) containers for option chains and pricing results.
//
// Every field lives in its own contiguous, 64-byte (cache-line) aligned array
// so that AVX2 kernels can use aligned loads (`_mm256_load_pd`) and never need
// gather instructions.
//
// Units used throughout the library
// ---------------------------------
//   spot, strike    : currency units of the underlying (must be > 0)
//   rate            : continuously compounded risk-free rate, per year (0.05 = 5%)
//   dividend        : continuous dividend / foreign rate, per year
//   time_to_expiry  : years (must be > 0)
//   volatility      : annualized volatility (0.20 = 20%, must be > 0)
//   is_call         : non-zero for a call, zero for a put
#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <span>
#include <vector>

namespace optengine {

/// Cache-line size used for all SoA array alignment.
inline constexpr std::size_t kCacheLineBytes = 64;

/// Number of doubles per AVX2 vector.
inline constexpr std::size_t kSimdWidth = 4;

/// Minimal allocator returning storage aligned to `kCacheLineBytes`. The byte
/// count is rounded up to a whole number of cache lines.
template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;

  constexpr AlignedAllocator() noexcept = default;
  template <typename U>
  constexpr AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

  [[nodiscard]] T* allocate(std::size_t n) {
    const std::size_t bytes =
        (n * sizeof(T) + kCacheLineBytes - 1) / kCacheLineBytes * kCacheLineBytes;
    return static_cast<T*>(::operator new(bytes, std::align_val_t{kCacheLineBytes}));
  }
  void deallocate(T* p, std::size_t) noexcept {
    ::operator delete(p, std::align_val_t{kCacheLineBytes});
  }

  template <typename U>
  [[nodiscard]] constexpr bool operator==(const AlignedAllocator<U>&) const noexcept {
    return true;
  }
};

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

/// Parameters of a single European option (scalar API).
struct OptionParams {
  double spot;            ///< S  > 0
  double strike;          ///< K  > 0
  double rate;            ///< r, per year, continuously compounded
  double dividend;        ///< q, per year, continuously compounded
  double time_to_expiry;  ///< T  > 0, years
  double volatility;      ///< sigma > 0, annualized
  bool is_call;
};

/// Price and first/second-order Greeks of a single option.
///
///   delta : dV/dS                (per 1.00 move in spot)
///   gamma : d2V/dS2
///   vega  : dV/dsigma            (per 1.00 = 100 vol points; divide by 100 for per-1%)
///   theta : dV/dt = -dV/dT       (per YEAR; divide by 365 for per calendar day)
///   rho   : dV/dr                (per 1.00 = 100% rate; divide by 100 for per-1%)
struct Greeks {
  double price;
  double delta;
  double gamma;
  double vega;
  double theta;
  double rho;
};

/// Read-only SoA view of an option chain. All spans must have equal size.
struct ChainView {
  std::span<const double> spot;
  std::span<const double> strike;
  std::span<const double> rate;
  std::span<const double> dividend;
  std::span<const double> time_to_expiry;
  std::span<const double> volatility;
  std::span<const std::uint8_t> is_call;

  [[nodiscard]] constexpr std::size_t size() const noexcept { return spot.size(); }
};

/// Writable SoA view of pre-allocated outputs. All spans must have equal size.
struct GreeksView {
  std::span<double> price;
  std::span<double> delta;
  std::span<double> gamma;
  std::span<double> vega;
  std::span<double> theta;
  std::span<double> rho;

  [[nodiscard]] constexpr std::size_t size() const noexcept { return price.size(); }
};

/// Owning SoA option chain. Each array is 64-byte aligned.
struct alignas(kCacheLineBytes) OptionChain {
  AlignedVector<double> spot;
  AlignedVector<double> strike;
  AlignedVector<double> rate;
  AlignedVector<double> dividend;
  AlignedVector<double> time_to_expiry;
  AlignedVector<double> volatility;
  AlignedVector<std::uint8_t> is_call;

  OptionChain() = default;
  explicit OptionChain(std::size_t n) { resize(n); }

  /// Resizes every array. New elements hold benign values (S = K = 100,
  /// T = 1, sigma = 0.2) so that pricing an unset slot is well defined.
  void resize(std::size_t n) {
    spot.resize(n, 100.0);
    strike.resize(n, 100.0);
    rate.resize(n, 0.0);
    dividend.resize(n, 0.0);
    time_to_expiry.resize(n, 1.0);
    volatility.resize(n, 0.2);
    is_call.resize(n, std::uint8_t{1});
  }

  [[nodiscard]] std::size_t size() const noexcept { return spot.size(); }

  void set(std::size_t i, const OptionParams& o) noexcept {
    spot[i] = o.spot;
    strike[i] = o.strike;
    rate[i] = o.rate;
    dividend[i] = o.dividend;
    time_to_expiry[i] = o.time_to_expiry;
    volatility[i] = o.volatility;
    is_call[i] = o.is_call ? std::uint8_t{1} : std::uint8_t{0};
  }

  [[nodiscard]] OptionParams get(std::size_t i) const noexcept {
    return {spot[i],           strike[i],     rate[i],        dividend[i],
            time_to_expiry[i], volatility[i], is_call[i] != 0};
  }

  [[nodiscard]] ChainView view() const noexcept {
    return {spot, strike, rate, dividend, time_to_expiry, volatility, is_call};
  }
};

/// Owning SoA output buffer. Each array is 64-byte aligned.
struct alignas(kCacheLineBytes) GreeksBuffer {
  AlignedVector<double> price;
  AlignedVector<double> delta;
  AlignedVector<double> gamma;
  AlignedVector<double> vega;
  AlignedVector<double> theta;
  AlignedVector<double> rho;

  GreeksBuffer() = default;
  explicit GreeksBuffer(std::size_t n) { resize(n); }

  void resize(std::size_t n) {
    price.resize(n);
    delta.resize(n);
    gamma.resize(n);
    vega.resize(n);
    theta.resize(n);
    rho.resize(n);
  }

  [[nodiscard]] std::size_t size() const noexcept { return price.size(); }

  [[nodiscard]] GreeksView view() noexcept { return {price, delta, gamma, vega, theta, rho}; }

  [[nodiscard]] Greeks get(std::size_t i) const noexcept {
    return {price[i], delta[i], gamma[i], vega[i], theta[i], rho[i]};
  }
};

}  // namespace optengine
