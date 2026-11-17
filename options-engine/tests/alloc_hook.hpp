// Test-only heap instrumentation.
//
// alloc_hook.cpp replaces the global operator new / delete family for the
// whole test binary and counts every allocation. Wrap the code under test in
// an AllocationCounter to prove that it performs no hidden heap allocation.
#pragma once

#include <cstdint>

namespace optengine::testing {

/// Total number of global operator new calls made by the process so far.
[[nodiscard]] std::uint64_t allocation_count() noexcept;

/// Counts allocations made (by any thread) since construction.
class AllocationCounter {
 public:
  AllocationCounter() noexcept : start_(allocation_count()) {}
  [[nodiscard]] std::uint64_t allocations() const noexcept { return allocation_count() - start_; }

 private:
  std::uint64_t start_;
};

}  // namespace optengine::testing
