// optengine/monte_carlo.hpp
//
// Multi-threaded Monte Carlo path simulators for Geometric Brownian Motion
// and the Heston stochastic-volatility model.
//
// Models (risk-neutral measure, time in years, rates annualized)
// --------------------------------------------------------------
//   GBM     dS = (r - q) S dt + sigma S dW
//           exact log-step:  ln S_{t+dt} = ln S_t + (r - q - sigma^2/2) dt + sigma sqrt(dt) Z
//
//   Heston  dS = (r - q) S dt + sqrt(v) S dW1
//           dv = kappa (theta - v) dt + xi sqrt(v) dW2,   d<W1, W2> = rho dt
//           full-truncation Euler in ln S (Lord, Koekkoek & van Dijk 2010), v+ = max(v, 0):
//             ln S += (r - q - v+/2) dt + sqrt(v+ dt) Z1
//             v    += kappa (theta - v+) dt + xi sqrt(v+ dt) (rho Z1 + sqrt(1 - rho^2) Z2)
//           (v0, theta are variances: 0.04 = 20% volatility; xi is vol-of-vol.)
//
// Estimator
// ---------
//   price     = e^{-rT} * mean(payoff)
//   std_error = e^{-rT} * stddev(sample) / sqrt(#samples)
// With antithetic variates each sample is the average payoff of the mirrored
// pair (Z, -Z), so `num_paths` paths yield num_paths / 2 independent samples.
//
// Determinism
// -----------
// Paths are split into fixed blocks of `kMcBlockSamples` samples. Each block
// re-seeds the worker's thread-local PCG64 from (seed, block index), and block
// sums are reduced serially in block order. The result is therefore
// bit-identical for a given seed regardless of thread count or scheduling.
//
// Allocation
// ----------
// The caller supplies an `McWorkspace` sized for the run. Path loops perform
// no heap allocation; the only allocations are made by the OS thread launch
// when num_threads > 1.
//
// Payoffs use static polymorphism (the `PathPayoff` concept); there is no
// virtual dispatch anywhere in the simulation kernels.
#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>

#include "optengine/math_utils.hpp"
#include "optengine/types.hpp"

namespace optengine {

// ---------------------------------------------------------------------------
// Random number generation
// ---------------------------------------------------------------------------

/// SplitMix64: used only to expand (seed, block) pairs into PCG64 seeds.
class SplitMix64 {
 public:
  constexpr explicit SplitMix64(std::uint64_t state) noexcept : state_(state) {}

  [[nodiscard]] constexpr std::uint64_t next() noexcept {
    std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

 private:
  std::uint64_t state_;
};

/// PCG64 (pcg_setseq_128_xsl_rr_64): 128-bit LCG state, 64-bit output,
/// period 2^128, 2^127 selectable streams. Output-compatible with the PCG
/// reference implementation and NumPy's PCG64 for identical (state, inc).
class Pcg64 {
 public:
  __extension__ using Uint128 = unsigned __int128;

  constexpr Pcg64() noexcept { seed(0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL); }
  constexpr Pcg64(Uint128 init_state, Uint128 init_seq) noexcept { seed(init_state, init_seq); }

  /// Standard PCG seeding: `init_seq` selects the stream.
  constexpr void seed(Uint128 init_state, Uint128 init_seq) noexcept {
    state_ = 0;
    inc_ = (init_seq << 1U) | 1U;
    step();
    state_ += init_state;
    step();
  }

  /// Seeds a statistically independent generator for (seed, stream).
  constexpr void seed_stream(std::uint64_t seed_value, std::uint64_t stream) noexcept {
    SplitMix64 mix(seed_value ^ (0xd1b54a32d192ed03ULL * (stream + 1)));
    const Uint128 s = (static_cast<Uint128>(mix.next()) << 64U) | mix.next();
    const Uint128 q = (static_cast<Uint128>(mix.next()) << 64U) | mix.next();
    seed(s, q);
  }

  /// Next 64 random bits.
  [[nodiscard]] constexpr std::uint64_t next() noexcept {
    step();
    const auto hi = static_cast<std::uint64_t>(state_ >> 64U);
    const auto lo = static_cast<std::uint64_t>(state_);
    const std::uint64_t x = hi ^ lo;
    const unsigned rot = static_cast<unsigned>(hi >> 58U);
    return (x >> rot) | (x << ((64U - rot) & 63U));
  }

  /// Uniform double strictly inside (0, 1) with 53 random bits.
  [[nodiscard]] constexpr double next_uniform() noexcept {
    return (static_cast<double>(next() >> 11U) + 0.5) * 0x1.0p-53;
  }

  /// Standard normal variate by inversion.
  [[nodiscard]] double next_normal() noexcept { return inv_norm_cdf(next_uniform()); }

 private:
  constexpr void step() noexcept { state_ = state_ * kMultiplier + inc_; }

  static constexpr Uint128 kMultiplier =
      (static_cast<Uint128>(0x2360ed051fc65da4ULL) << 64U) | 0x4385df649fccf645ULL;

  Uint128 state_ = 0;
  Uint128 inc_ = 1;
};

// ---------------------------------------------------------------------------
// Payoffs (static polymorphism)
// ---------------------------------------------------------------------------

/// A payoff evaluated along one simulated path.
///
///   State init()                     fresh per-path state
///   void  observe(State&, s)         called at every monitoring date t_1 .. t_n
///                                    (only when kPathDependent is true)
///   double terminal(State, s_T)      undiscounted payoff at expiry
template <typename P>
concept PathPayoff = requires(const P& payoff, typename P::State& state, double spot) {
  typename P::State;
  { P::kPathDependent } -> std::convertible_to<bool>;
  { payoff.init() } noexcept -> std::same_as<typename P::State>;
  { payoff.observe(state, spot) } noexcept;
  { payoff.terminal(state, spot) } noexcept -> std::convertible_to<double>;
};

struct NoPathState {};

/// European call: max(S_T - K, 0).
struct EuropeanCall {
  using State = NoPathState;
  static constexpr bool kPathDependent = false;
  double strike;

  [[nodiscard]] State init() const noexcept { return {}; }
  void observe(State&, double) const noexcept {}
  [[nodiscard]] double terminal(const State&, double s_t) const noexcept {
    return std::max(s_t - strike, 0.0);
  }
};

/// European put: max(K - S_T, 0).
struct EuropeanPut {
  using State = NoPathState;
  static constexpr bool kPathDependent = false;
  double strike;

  [[nodiscard]] State init() const noexcept { return {}; }
  void observe(State&, double) const noexcept {}
  [[nodiscard]] double terminal(const State&, double s_t) const noexcept {
    return std::max(strike - s_t, 0.0);
  }
};

struct AverageState {
  double sum = 0.0;
  double count = 0.0;
};

/// Arithmetic-average Asian call: max(mean(S_{t_1..t_n}) - K, 0).
struct ArithmeticAsianCall {
  using State = AverageState;
  static constexpr bool kPathDependent = true;
  double strike;

  [[nodiscard]] State init() const noexcept { return {}; }
  void observe(State& st, double s) const noexcept {
    st.sum += s;
    st.count += 1.0;
  }
  [[nodiscard]] double terminal(const State& st, double) const noexcept {
    return std::max(st.sum / st.count - strike, 0.0);
  }
};

/// Geometric-average Asian call: max(exp(mean(ln S_{t_1..t_n})) - K, 0).
/// Has a closed form under GBM, which makes it a sharp test of the path engine.
struct GeometricAsianCall {
  using State = AverageState;
  static constexpr bool kPathDependent = true;
  double strike;

  [[nodiscard]] State init() const noexcept { return {}; }
  void observe(State& st, double s) const noexcept {
    st.sum += std::log(s);
    st.count += 1.0;
  }
  [[nodiscard]] double terminal(const State& st, double) const noexcept {
    return std::max(std::exp(st.sum / st.count) - strike, 0.0);
  }
};

struct BarrierState {
  bool alive = true;
};

/// Discretely monitored up-and-out call: max(S_T - K, 0) unless S_{t_i} >= barrier.
struct UpAndOutCall {
  using State = BarrierState;
  static constexpr bool kPathDependent = true;
  double strike;
  double barrier;

  [[nodiscard]] State init() const noexcept { return {}; }
  void observe(State& st, double s) const noexcept { st.alive = st.alive && s < barrier; }
  [[nodiscard]] double terminal(const State& st, double s_t) const noexcept {
    return st.alive ? std::max(s_t - strike, 0.0) : 0.0;
  }
};

// ---------------------------------------------------------------------------
// Models, configuration, results
// ---------------------------------------------------------------------------

struct GbmParams {
  double spot;            ///< S0 > 0
  double rate;            ///< r, per year
  double dividend;        ///< q, per year
  double volatility;      ///< sigma, annualized
  double time_to_expiry;  ///< T, years
};

struct HestonParams {
  double spot;            ///< S0 > 0
  double rate;            ///< r, per year
  double dividend;        ///< q, per year
  double v0;              ///< initial variance (sigma0^2)
  double kappa;           ///< mean-reversion speed, per year
  double theta;           ///< long-run variance
  double xi;              ///< volatility of variance ("vol of vol")
  double rho;             ///< correlation between spot and variance shocks, in [-1, 1]
  double time_to_expiry;  ///< T, years
};

/// Samples per work block (a sample is one path, or one antithetic pair).
inline constexpr std::size_t kMcBlockSamples = 2048;
/// Upper bound on worker threads.
inline constexpr unsigned kMcMaxThreads = 64;

struct McConfig {
  std::size_t num_paths = 100'000;  ///< total simulated paths (rounded down to even if antithetic)
  std::size_t num_steps = 1;        ///< time steps (monitoring dates) per path, >= 1
  unsigned num_threads = 1;         ///< worker threads, clamped to [1, kMcMaxThreads]
  std::uint64_t seed = 42;          ///< master seed
  bool antithetic = true;           ///< antithetic variates (Z, -Z)
};

enum class McStatus : std::uint8_t {
  kOk = 0,
  kInvalidConfig,      ///< zero paths/steps or fewer than two samples
  kWorkspaceTooSmall,  ///< McWorkspace cannot hold this run; nothing was simulated
};

struct McResult {
  double price = 0.0;           ///< discounted Monte Carlo estimate
  double std_error = 0.0;       ///< standard error of `price`
  std::size_t num_samples = 0;  ///< independent samples (paths, or pairs if antithetic)
  std::size_t num_paths = 0;    ///< paths actually simulated
  McStatus status = McStatus::kInvalidConfig;
};

/// Caller-owned scratch memory: per-block partial sums.
class McWorkspace {
 public:
  McWorkspace() = default;
  /// Allocates scratch for runs of up to `max_paths` paths.
  explicit McWorkspace(std::size_t max_paths) { reserve(max_paths); }

  /// Grows the scratch (allocates). Call outside timing-critical code.
  void reserve(std::size_t max_paths) {
    const std::size_t blocks = max_paths / kMcBlockSamples + 1;
    if (blocks > sum_.size()) {
      sum_.resize(blocks);
      sum_sq_.resize(blocks);
    }
  }

  [[nodiscard]] std::size_t block_capacity() const noexcept { return sum_.size(); }
  [[nodiscard]] std::span<double> block_sum() noexcept { return sum_; }
  [[nodiscard]] std::span<double> block_sum_sq() noexcept { return sum_sq_; }

 private:
  AlignedVector<double> sum_;
  AlignedVector<double> sum_sq_;
};

namespace detail {

/// Sum and sum of squares of the samples in one block.
struct BlockSums {
  double sum = 0.0;
  double sum_sq = 0.0;
};

/// Runs `fn(ctx, block)` for block = 0 .. num_blocks-1 on `num_threads` workers
/// (the calling thread is one of them). Blocks are claimed dynamically.
void run_blocks(std::size_t num_blocks, unsigned num_threads,
                void (*fn)(void* ctx, std::size_t block), void* ctx) noexcept;

/// Reduces block sums (in block order) into a discounted price and std error.
[[nodiscard]] McResult reduce_blocks(std::span<const double> sum, std::span<const double> sum_sq,
                                     std::size_t num_samples, std::size_t num_paths,
                                     double discount) noexcept;

}  // namespace detail

// ---------------------------------------------------------------------------
// Block kernels (single-threaded, allocation-free)
// ---------------------------------------------------------------------------

/// Simulates `num_samples` GBM samples and accumulates undiscounted payoffs.
template <PathPayoff Payoff>
[[nodiscard]] detail::BlockSums simulate_gbm_block(const GbmParams& m, const Payoff& payoff,
                                                   std::size_t num_samples, std::size_t num_steps,
                                                   bool antithetic, Pcg64& rng) noexcept {
  const double dt = m.time_to_expiry / static_cast<double>(num_steps);
  const double drift = (m.rate - m.dividend - 0.5 * m.volatility * m.volatility) * dt;
  const double diffusion = m.volatility * std::sqrt(dt);
  detail::BlockSums acc;

  if constexpr (!Payoff::kPathDependent) {
    // Terminal payoff only: accumulate the Brownian increments, exponentiate once.
    const double total_drift = drift * static_cast<double>(num_steps);
    for (std::size_t p = 0; p < num_samples; ++p) {
      double z_sum = 0.0;
      for (std::size_t t = 0; t < num_steps; ++t) z_sum += rng.next_normal();
      const typename Payoff::State st = payoff.init();
      double sample = payoff.terminal(st, m.spot * std::exp(total_drift + diffusion * z_sum));
      if (antithetic) {
        sample = 0.5 * (sample + payoff.terminal(
                                     st, m.spot * std::exp(total_drift - diffusion * z_sum)));
      }
      acc.sum += sample;
      acc.sum_sq += sample * sample;
    }
  } else {
    // Path-dependent: S+ *= c * e^{dZ}, S- *= c / e^{dZ}  (one exp per step for the pair).
    const double growth = std::exp(drift);
    for (std::size_t p = 0; p < num_samples; ++p) {
      typename Payoff::State st_up = payoff.init();
      typename Payoff::State st_dn = payoff.init();
      double s_up = m.spot;
      double s_dn = m.spot;
      for (std::size_t t = 0; t < num_steps; ++t) {
        const double shock = std::exp(diffusion * rng.next_normal());
        s_up *= growth * shock;
        payoff.observe(st_up, s_up);
        if (antithetic) {
          s_dn *= growth / shock;
          payoff.observe(st_dn, s_dn);
        }
      }
      double sample = payoff.terminal(st_up, s_up);
      if (antithetic) sample = 0.5 * (sample + payoff.terminal(st_dn, s_dn));
      acc.sum += sample;
      acc.sum_sq += sample * sample;
    }
  }
  return acc;
}

/// Simulates `num_samples` Heston samples (full-truncation Euler in ln S).
template <PathPayoff Payoff>
[[nodiscard]] detail::BlockSums simulate_heston_block(const HestonParams& m, const Payoff& payoff,
                                                      std::size_t num_samples,
                                                      std::size_t num_steps, bool antithetic,
                                                      Pcg64& rng) noexcept {
  const double dt = m.time_to_expiry / static_cast<double>(num_steps);
  const double sqrt_dt = std::sqrt(dt);
  const double carry_dt = (m.rate - m.dividend) * dt;
  const double kappa_dt = m.kappa * dt;
  const double rho_perp = std::sqrt(std::max(1.0 - m.rho * m.rho, 0.0));
  const int num_legs = antithetic ? 2 : 1;
  detail::BlockSums acc;

  for (std::size_t p = 0; p < num_samples; ++p) {
    double x[2] = {0.0, 0.0};      // ln(S / S0) for the path and its mirror
    double v[2] = {m.v0, m.v0};    // variance
    typename Payoff::State st[2] = {payoff.init(), payoff.init()};
    for (std::size_t t = 0; t < num_steps; ++t) {
      const double z1 = rng.next_normal();
      const double z2 = rng.next_normal();
      const double zv = m.rho * z1 + rho_perp * z2;
      double sign = 1.0;
      for (int leg = 0; leg < num_legs; ++leg) {
        const double v_plus = std::max(v[leg], 0.0);
        const double vol_step = std::sqrt(v_plus) * sqrt_dt;
        x[leg] += carry_dt - 0.5 * v_plus * dt + vol_step * (sign * z1);
        v[leg] += kappa_dt * (m.theta - v_plus) + m.xi * vol_step * (sign * zv);
        if constexpr (Payoff::kPathDependent) payoff.observe(st[leg], m.spot * std::exp(x[leg]));
        sign = -1.0;
      }
    }
    double sample = payoff.terminal(st[0], m.spot * std::exp(x[0]));
    if (antithetic) sample = 0.5 * (sample + payoff.terminal(st[1], m.spot * std::exp(x[1])));
    acc.sum += sample;
    acc.sum_sq += sample * sample;
  }
  return acc;
}

// ---------------------------------------------------------------------------
// Multi-threaded drivers
// ---------------------------------------------------------------------------
namespace detail {

template <typename Model, typename Payoff, typename BlockFn>
[[nodiscard]] McResult run_monte_carlo(const Model& model, const Payoff& payoff,
                                       const McConfig& config, McWorkspace& workspace,
                                       BlockFn block_fn) noexcept {
  McResult failure;
  const std::size_t paths_per_sample = config.antithetic ? 2 : 1;
  const std::size_t num_samples = config.num_paths / paths_per_sample;
  if (num_samples < 2 || config.num_steps == 0) {
    failure.status = McStatus::kInvalidConfig;
    return failure;
  }
  const std::size_t num_blocks = (num_samples + kMcBlockSamples - 1) / kMcBlockSamples;
  if (num_blocks > workspace.block_capacity()) {
    failure.status = McStatus::kWorkspaceTooSmall;
    return failure;
  }

  struct Context {
    const Model* model;
    const Payoff* payoff;
    const McConfig* config;
    BlockFn* block_fn;
    double* sum;
    double* sum_sq;
    std::size_t num_samples;
  };
  Context ctx{&model,
              &payoff,
              &config,
              &block_fn,
              workspace.block_sum().data(),
              workspace.block_sum_sq().data(),
              num_samples};

  auto trampoline = [](void* raw, std::size_t block) {
    const auto* c = static_cast<const Context*>(raw);
    // One generator per worker thread, re-seeded per block for determinism.
    thread_local Pcg64 rng;
    rng.seed_stream(c->config->seed, block);
    const std::size_t first = block * kMcBlockSamples;
    const std::size_t count = std::min(kMcBlockSamples, c->num_samples - first);
    const BlockSums sums = (*c->block_fn)(*c->model, *c->payoff, count, c->config->num_steps,
                                          c->config->antithetic, rng);
    c->sum[block] = sums.sum;
    c->sum_sq[block] = sums.sum_sq;
  };
  run_blocks(num_blocks, config.num_threads, trampoline, &ctx);

  return reduce_blocks(workspace.block_sum().first(num_blocks),
                       workspace.block_sum_sq().first(num_blocks), num_samples,
                       num_samples * paths_per_sample, std::exp(-model.rate * model.time_to_expiry));
}

}  // namespace detail

/// Monte Carlo price of `payoff` under GBM.
template <PathPayoff Payoff>
[[nodiscard]] McResult mc_price_gbm(const GbmParams& model, const Payoff& payoff,
                                    const McConfig& config, McWorkspace& workspace) noexcept {
  return detail::run_monte_carlo(
      model, payoff, config, workspace,
      [](const GbmParams& m, const Payoff& p, std::size_t n, std::size_t steps, bool anti,
         Pcg64& rng) noexcept { return simulate_gbm_block(m, p, n, steps, anti, rng); });
}

/// Monte Carlo price of `payoff` under Heston.
template <PathPayoff Payoff>
[[nodiscard]] McResult mc_price_heston(const HestonParams& model, const Payoff& payoff,
                                       const McConfig& config, McWorkspace& workspace) noexcept {
  return detail::run_monte_carlo(
      model, payoff, config, workspace,
      [](const HestonParams& m, const Payoff& p, std::size_t n, std::size_t steps, bool anti,
         Pcg64& rng) noexcept { return simulate_heston_block(m, p, n, steps, anti, rng); });
}

/// European call/put under GBM (non-template convenience wrapper).
[[nodiscard]] McResult mc_european_gbm(const GbmParams& model, double strike, bool is_call,
                                       const McConfig& config, McWorkspace& workspace) noexcept;

/// European call/put under Heston (non-template convenience wrapper).
[[nodiscard]] McResult mc_european_heston(const HestonParams& model, double strike, bool is_call,
                                          const McConfig& config, McWorkspace& workspace) noexcept;

}  // namespace optengine
