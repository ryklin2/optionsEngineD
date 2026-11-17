cmake_minimum_required(VERSION 3.20)
project(optengine
  VERSION 1.0.0
  DESCRIPTION "High-throughput vectorized options pricing & Greeks engine"
  LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Language standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(OPTENGINE_NATIVE_ARCH      "Compile with -march=native (enables the AVX2 path on capable CPUs)" ON)
option(OPTENGINE_SANITIZE         "Enable ASan + UBSan in Debug builds" ON)
option(OPTENGINE_BUILD_TESTS      "Build the GoogleTest suite" ON)
option(OPTENGINE_BUILD_BENCHMARKS "Build the Google Benchmark suite" ON)

# ---------------------------------------------------------------------------
# Build-type flags
#   Debug   : -O0 -g -fsanitize=address,undefined   (test builds)
#   Release : -O3 -march=native -ffast-math         (benchmark builds)
# Sanitizer flags are global so third-party code is instrumented too (mixing
# instrumented and uninstrumented std containers yields ASan false positives).
# -march=native / -ffast-math are scoped to optengine targets only.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
if(OPTENGINE_SANITIZE)
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fsanitize=address,undefined -fno-omit-frame-pointer")
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address,undefined")
endif()

add_library(optengine_flags INTERFACE)
target_compile_options(optengine_flags INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion
  $<$<CONFIG:Release>:-ffast-math>)
if(OPTENGINE_NATIVE_ARCH)
  target_compile_options(optengine_flags INTERFACE -march=native)
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(optengine STATIC
  src/bsm_scalar.cpp
  src/bsm_simd.cpp
  src/iv_solver.cpp
  src/monte_carlo.cpp)
add_library(optengine::optengine ALIAS optengine)
target_include_directories(optengine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(optengine PUBLIC optengine_flags Threads::Threads)

# ---------------------------------------------------------------------------
# Dependencies (FetchContent)
# ---------------------------------------------------------------------------
include(FetchContent)

if(OPTENGINE_BUILD_TESTS)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  include(GoogleTest)

  add_executable(optengine_tests
    tests/alloc_hook.cpp
    tests/test_bsm.cpp
    tests/test_iv.cpp
    tests/test_mc.cpp)
  target_link_libraries(optengine_tests PRIVATE optengine GTest::gtest_main)
  gtest_discover_tests(optengine_tests DISCOVERY_TIMEOUT 60)
endif()

if(OPTENGINE_BUILD_BENCHMARKS)
  set(BENCHMARK_ENABLE_TESTING       OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS   OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL       OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR        OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_INSTALL_DOCS         OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.5
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(benchmark)

  add_executable(bench_pricing benchmarks/bench_pricing.cpp)
  target_link_libraries(bench_pricing PRIVATE optengine benchmark::benchmark)
endif()
