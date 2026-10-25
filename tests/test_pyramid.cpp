#include <cmath>
#include <cstdint>
#include <random>

#include <doctest/doctest.h>

#include "r3d/core/pyramid.hpp"
#include "test_util.hpp"

TEST_CASE("Pyramid level sizes and intrinsics") {
  const r3d::ImageU8 img(101, 64, 3);
  const auto levels = r3d::build_pyramid(img, 10, 8);
  REQUIRE(levels.size() == 4);  // 101x64, 51x32, 26x16, 13x8
  CHECK(levels[1].width() == 51);
  CHECK(levels[3].width() == 13);
  CHECK(levels[3].height() == 8);
  CHECK(levels[3].channels() == 3);

  const r3d::PinholeIntrinsics k{101, 64, 80.0, 82.0, 50.25, 31.5};
  const r3d::PinholeIntrinsics k3 = r3d::pyramid_intrinsics(k, 3);
  CHECK(k3.width == 13);
  CHECK(k3.height == 8);
  CHECK(k3.fx == 10.0);
  CHECK(k3.cy == 31.5 / 8.0);

  CHECK(r3d::pyramid_levels_for(4000, 3000, 1e6) == 3);  // 4000x3000 -> 2000x1500 -> 1000x750
  CHECK(r3d::pyramid_levels_for(100, 100, 1e6) == 1);
}

TEST_CASE("Pyramid preserves positions: x_k = x_0 / 2^k") {
  const Eigen::Vector2d blob(123.37, 77.81);
  constexpr double kSigma = 6.0;
  r3d::ImageF32 img(256, 160, 1);
  for (int y = 0; y < img.height(); ++y) {
    for (int x = 0; x < img.width(); ++x) {
      img(x, y) = static_cast<float>(std::exp(-(Eigen::Vector2d(x, y) - blob).squaredNorm() / (2 * kSigma * kSigma)));
    }
  }
  const auto levels = r3d::build_pyramid(img, 4);
  REQUIRE(levels.size() == 4);
  for (int k = 0; k < 4; ++k) {
    const r3d::ImageF32& level = levels[static_cast<std::size_t>(k)];
    double sum = 0.0;
    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    for (int y = 0; y < level.height(); ++y) {
      for (int x = 0; x < level.width(); ++x) {
        sum += level(x, y);
        centroid += level(x, y) * Eigen::Vector2d(x, y);
      }
    }
    centroid /= sum;
    CAPTURE(k);
    R3D_CHECK_NEAR(centroid.x(), blob.x() / (1 << k), 1e-3);
    R3D_CHECK_NEAR(centroid.y(), blob.y() / (1 << k), 1e-3);
  }
}

TEST_CASE("8-bit pyramid is the exactly rounded float pyramid") {
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> dist(0, 255);
  r3d::ImageU8 u8(77, 41, 2);
  r3d::ImageF32 f32(77, 41, 2);
  for (int y = 0; y < u8.height(); ++y) {
    for (int x = 0; x < u8.width(); ++x) {
      for (int c = 0; c < 2; ++c) {
        u8(x, y, c) = static_cast<std::uint8_t>(dist(rng));
        f32(x, y, c) = u8(x, y, c);
      }
    }
  }
  const r3d::ImageU8 a = r3d::pyramid_down(u8);
  const r3d::ImageF32 b = r3d::pyramid_down(f32);
  REQUIRE(a.width() == 39);
  REQUIRE(a.height() == 21);
  int mismatches = 0;
  for (int y = 0; y < a.height(); ++y) {
    for (int x = 0; x < a.width(); ++x) {
      for (int c = 0; c < 2; ++c) {
        // Exact sums are multiples of 1/256, so rounding half up is unambiguous.
        mismatches += a(x, y, c) == static_cast<int>(std::floor(b(x, y, c) + 0.5F)) ? 0 : 1;
      }
    }
  }
  CHECK(mismatches == 0);

  r3d::ImageU16 flat(9, 9, 1);
  flat.fill(54321);
  const r3d::ImageU16 down = r3d::pyramid_down(flat);
  CHECK(down(0, 0) == 54321);
  CHECK(down(4, 4) == 54321);
}
