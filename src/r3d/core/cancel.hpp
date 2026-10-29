#pragma once

#include <atomic>

#include "r3d/core/error.hpp"

namespace r3d {

// Thrown when a cancellation request stops a computation.
class Cancelled : public Error {
public:
  Cancelled() : Error("cancelled") {}
};

// Cancellation request, set from any thread (or a signal handler) and polled
// by the computation. Not copyable: computations hold a pointer to it.
class CancellationToken {
public:
  CancellationToken() = default;
  CancellationToken(const CancellationToken&) = delete;
  CancellationToken& operator=(const CancellationToken&) = delete;

  void request() noexcept { requested_.store(true, std::memory_order_relaxed); }
  void reset() noexcept { requested_.store(false, std::memory_order_relaxed); }
  [[nodiscard]] bool requested() const noexcept { return requested_.load(std::memory_order_relaxed); }
  void throw_if_requested() const {
    if (requested()) {
      throw Cancelled();
    }
  }

private:
  std::atomic<bool> requested_{false};
};

// The cancellation token of the calling thread (null if none). parallel_for
// polls the token of the thread that calls it before every chunk and throws
// Cancelled once it is requested; its bodies run with the same token, so
// nested loops stop as well.
[[nodiscard]] const CancellationToken* current_cancellation() noexcept;

// Throws Cancelled if the calling thread's token is requested.
void throw_if_cancelled();

// Makes `token` the calling thread's token for the lifetime of the scope (null: none).
class CancellationScope {
public:
  explicit CancellationScope(const CancellationToken* token) noexcept;
  CancellationScope(const CancellationScope&) = delete;
  CancellationScope& operator=(const CancellationScope&) = delete;
  ~CancellationScope();

private:
  const CancellationToken* previous_;
};

}  // namespace r3d
