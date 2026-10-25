#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace r3d {

// Runtime SIMD dispatch information (Google Highway). Each kernel is compiled
// for several instruction sets and the best one supported by the CPU is
// selected at runtime.

// Name of the target the dispatcher currently selects, e.g. "AVX2".
[[nodiscard]] std::string simd_active_target_name();

// Targets that are both compiled into this binary and supported by the CPU,
// best first.
[[nodiscard]] std::vector<std::int64_t> simd_available_targets();

[[nodiscard]] std::string simd_target_name(std::int64_t target);

// Restricts dynamic dispatch to a single target while alive. Intended for
// tests that compare every SIMD target against the scalar reference. Not
// thread-safe; do not use while other threads run dispatched kernels.
class ScopedSimdTarget {
public:
  explicit ScopedSimdTarget(std::int64_t target);
  ScopedSimdTarget(const ScopedSimdTarget&) = delete;
  ScopedSimdTarget& operator=(const ScopedSimdTarget&) = delete;
  ~ScopedSimdTarget();
};

}  // namespace r3d
