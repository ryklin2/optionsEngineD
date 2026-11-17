#include "alloc_hook.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace {

std::atomic<std::uint64_t> g_allocations{0};

[[nodiscard]] void* counted_alloc(std::size_t size) noexcept {
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  return std::malloc(size == 0 ? 1 : size);
}

[[nodiscard]] void* counted_aligned_alloc(std::size_t size, std::size_t alignment) noexcept {
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (alignment < sizeof(void*)) alignment = sizeof(void*);
  // aligned_alloc requires the size to be a multiple of the alignment.
  const std::size_t rounded = (size + alignment - 1) / alignment * alignment;
  return std::aligned_alloc(alignment, rounded == 0 ? alignment : rounded);
}

}  // namespace

namespace optengine::testing {
std::uint64_t allocation_count() noexcept {
  return g_allocations.load(std::memory_order_relaxed);
}
}  // namespace optengine::testing

// ---- Replacement global allocation functions -------------------------------
void* operator new(std::size_t size) {
  if (void* p = counted_alloc(size)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
  if (void* p = counted_alloc(size)) return p;
  throw std::bad_alloc();
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept { return counted_alloc(size); }
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  return counted_alloc(size);
}
void* operator new(std::size_t size, std::align_val_t al) {
  if (void* p = counted_aligned_alloc(size, static_cast<std::size_t>(al))) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t al) {
  if (void* p = counted_aligned_alloc(size, static_cast<std::size_t>(al))) return p;
  throw std::bad_alloc();
}
void* operator new(std::size_t size, std::align_val_t al, const std::nothrow_t&) noexcept {
  return counted_aligned_alloc(size, static_cast<std::size_t>(al));
}
void* operator new[](std::size_t size, std::align_val_t al, const std::nothrow_t&) noexcept {
  return counted_aligned_alloc(size, static_cast<std::size_t>(al));
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
