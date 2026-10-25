#pragma once

#include <chrono>
#include <string>
#include <utility>

#include "r3d/core/log.hpp"

namespace r3d {

class Stopwatch {
public:
  Stopwatch() : start_(Clock::now()) {}
  void restart() { start_ = Clock::now(); }
  [[nodiscard]] double elapsed_seconds() const {
    return std::chrono::duration<double>(Clock::now() - start_).count();
  }

private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

// Logs the lifetime of a scope, e.g. one pipeline stage.
class ScopedTimer {
public:
  explicit ScopedTimer(std::string name, LogLevel level = LogLevel::debug)
      : name_(std::move(name)), level_(level) {}
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;
  ~ScopedTimer() { log(level_, "{} took {:.3f} s", name_, watch_.elapsed_seconds()); }

private:
  std::string name_;
  LogLevel level_;
  Stopwatch watch_;
};

}  // namespace r3d
