#include "optengine/monte_carlo.hpp"

#include <array>
#include <atomic>
#include <thread>

namespace optengine {
namespace detail {

void run_blocks(std::size_t num_blocks, unsigned num_threads,
                void (*fn)(void* ctx, std::size_t block), void* ctx) noexcept {
  std::atomic<std::size_t> next_block{0};
  const auto worker = [&next_block, num_blocks, fn, ctx]() noexcept {
    for (;;) {
      const std::size_t block = next_block.fetch_add(1, std::memory_order_relaxed);
      if (block >= num_blocks) return;
      fn(ctx, block);
    }
  };

  unsigned threads = std::clamp(num_threads, 1U, kMcMaxThreads);
  if (static_cast<std::size_t>(threads) > num_blocks) threads = static_cast<unsigned>(num_blocks);
  if (threads <= 1) {
    worker();
    return;
  }
  // Fixed-capacity pool: jthreads join on scope exit, the caller works too.
  std::array<std::jthread, kMcMaxThreads - 1> pool;
  for (unsigned t = 0; t + 1 < threads; ++t) pool[t] = std::jthread(worker);
  worker();
}

McResult reduce_blocks(std::span<const double> sum, std::span<const double> sum_sq,
                       std::size_t num_samples, std::size_t num_paths, double discount) noexcept {
  double total = 0.0;
  double total_sq = 0.0;
  for (std::size_t b = 0; b < sum.size(); ++b) {
    total += sum[b];
    total_sq += sum_sq[b];
  }
  const double n = static_cast<double>(num_samples);
  const double mean = total / n;
  const double variance = std::max((total_sq - n * mean * mean) / (n - 1.0), 0.0);

  McResult result;
  result.price = discount * mean;
  result.std_error = discount * std::sqrt(variance / n);
  result.num_samples = num_samples;
  result.num_paths = num_paths;
  result.status = McStatus::kOk;
  return result;
}

}  // namespace detail

McResult mc_european_gbm(const GbmParams& model, double strike, bool is_call,
                         const McConfig& config, McWorkspace& workspace) noexcept {
  return is_call ? mc_price_gbm(model, EuropeanCall{strike}, config, workspace)
                 : mc_price_gbm(model, EuropeanPut{strike}, config, workspace);
}

McResult mc_european_heston(const HestonParams& model, double strike, bool is_call,
                            const McConfig& config, McWorkspace& workspace) noexcept {
  return is_call ? mc_price_heston(model, EuropeanCall{strike}, config, workspace)
                 : mc_price_heston(model, EuropeanPut{strike}, config, workspace);
}

}  // namespace optengine
