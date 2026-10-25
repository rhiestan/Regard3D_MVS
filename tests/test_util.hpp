#pragma once

#include <atomic>
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
