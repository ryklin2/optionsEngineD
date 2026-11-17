#include "optengine/bsm_scalar.hpp"

#include <cassert>
#include <cstddef>

namespace optengine {

// Keep the reference loop scalar even at -O3 -ffast-math -march=native, where
// the compiler could otherwise auto-vectorize it through libmvec.
#if defined(__clang__)
#define OPTENGINE_SCALAR_LOOP _Pragma("clang loop vectorize(disable) interleave(disable)")
#define OPTENGINE_NO_TREE_VECTORIZE
#elif defined(__GNUC__)
#define OPTENGINE_SCALAR_LOOP
#define OPTENGINE_NO_TREE_VECTORIZE __attribute__((optimize("no-tree-vectorize")))
#else
#define OPTENGINE_SCALAR_LOOP
#define OPTENGINE_NO_TREE_VECTORIZE
#endif

OPTENGINE_NO_TREE_VECTORIZE
void bsm_scalar_chain(ChainView in, GreeksView out) noexcept {
  const std::size_t n = in.size();
  assert(in.strike.size() == n && in.rate.size() == n && in.dividend.size() == n &&
         in.time_to_expiry.size() == n && in.volatility.size() == n && in.is_call.size() == n);
  assert(out.price.size() >= n && out.delta.size() >= n && out.gamma.size() >= n &&
         out.vega.size() >= n && out.theta.size() >= n && out.rho.size() >= n);

  OPTENGINE_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    const Greeks g = bsm_scalar_greeks({in.spot[i], in.strike[i], in.rate[i], in.dividend[i],
                                        in.time_to_expiry[i], in.volatility[i],
                                        in.is_call[i] != 0});
    out.price[i] = g.price;
    out.delta[i] = g.delta;
    out.gamma[i] = g.gamma;
    out.vega[i] = g.vega;
    out.theta[i] = g.theta;
    out.rho[i] = g.rho;
  }
}

}  // namespace optengine
