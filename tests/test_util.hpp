#pragma once

#include <atomic>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>

#include <doctest/doctest.h>

// CHECK that |a - b| <= tol; reports both values on failure.
#define R3D_CHECK_NEAR(a, b, tol) \
  CHECK_MESSAGE(std::abs((a) - (b)) <= (tol), #a " = ", (a), ", " #b " = ", (b))

namespace r3d::test {

// Random numbers that are identical on every standard library: std::mt19937's
// output is fixed by the standard, but the std::*_distribution algorithms are
// not (libstdc++, libc++ and MSVC produce different values from the same seed).
class Rng {
public:
  explicit Rng(unsigned seed) : gen_(seed) {}
  // Uniform in [lo, hi).
  double uniform(double lo, double hi) { return lo + (hi - lo) * ((gen_() + 0.5) / 4294967296.0); }
  // Uniform integer in [lo, hi] (modulo bias is irrelevant for tests).
  int uniform_int(int lo, int hi) {
    return lo + static_cast<int>(gen_() % static_cast<std::uint32_t>(hi - lo + 1));
  }
  // Normal distribution (Box-Muller).
  double normal(double mean, double sigma) {
    const double u1 = uniform(0.0, 1.0), u2 = uniform(0.0, 1.0);
    return mean + sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }

private:
  std::mt19937 gen_;
};

// Unique temporary directory, removed when the object goes out of scope.
class TempDir {
public:
  TempDir() {
    static std::atomic<int> counter{0};
    std::random_device rd;
    path_ = std::filesystem::temp_directory_path() /
            ("r3d_test_" + std::to_string(rd()) + "_" + std::to_string(counter.fetch_add(1)));
    std::filesystem::create_directories(path_);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

}  // namespace r3d::test
