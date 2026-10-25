#include <cstdint>
#include <random>
#include <vector>

#include <doctest/doctest.h>

#include "r3d/core/color_convert.hpp"
#include "r3d/core/simd.hpp"

namespace {

std::vector<std::uint8_t> random_bytes(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<std::uint8_t> v(n);
  for (auto& b : v) b = static_cast<std::uint8_t>(dist(rng));
  return v;
}

}  // namespace

TEST_CASE("RGB/RGBA to gray: every SIMD target is bit-exact with scalar") {
  const std::vector<std::int64_t> targets = r3d::simd_available_targets();
  REQUIRE_FALSE(targets.empty());
  for (const std::int64_t target : targets) {
    const r3d::ScopedSimdTarget scoped(target);
    INFO("target ", r3d::simd_target_name(target));
    CHECK(r3d::simd_active_target_name() == r3d::simd_target_name(target));
    // Lengths cover empty input, tails and multiple full vectors.
    for (const std::size_t n : {0u, 1u, 7u, 15u, 16u, 17u, 31u, 64u, 100u, 1023u}) {
      const auto rgb = random_bytes(3 * n, static_cast<unsigned>(n));
      const auto rgba = random_bytes(4 * n, static_cast<unsigned>(n) + 1000);
      std::vector<std::uint8_t> expected(n), actual(n);

      r3d::scalar::rgb_to_gray_row(rgb.data(), expected.data(), n);
      r3d::rgb_to_gray_row(rgb.data(), actual.data(), n);
      CHECK(actual == expected);

      r3d::scalar::rgba_to_gray_row(rgba.data(), expected.data(), n);
      r3d::rgba_to_gray_row(rgba.data(), actual.data(), n);
      CHECK(actual == expected);
    }
  }
}

TEST_CASE("Gray conversion of known colours") {
  r3d::ImageU8 img(3, 1, 3);
  const std::uint8_t pixels[] = {255, 255, 255, 255, 0, 0, 0, 0, 0};
  for (int i = 0; i < 9; ++i) img.row(0)[i] = pixels[i];
  const r3d::ImageU8 gray = r3d::to_gray(img.view());
  CHECK(gray.channels() == 1);
  CHECK(gray(0, 0) == 255);  // weights sum to 256
  CHECK(gray(1, 0) == 77);   // (77 * 255 + 128) >> 8 = 77
  CHECK(gray(2, 0) == 0);
}

TEST_CASE("to_gray on a multi-row image and invalid channels") {
  r3d::ImageU8 rgba(70, 40, 4);
  for (int y = 0; y < 40; ++y)
    for (int x = 0; x < 70; ++x)
      for (int c = 0; c < 4; ++c) rgba(x, y, c) = static_cast<std::uint8_t>((x * 3 + y * 7 + c * 11) & 0xFF);
  const r3d::ImageU8 gray = r3d::to_gray(rgba.view());
  for (int y = 0; y < 40; ++y) {
    std::vector<std::uint8_t> expected(70);
    r3d::scalar::rgba_to_gray_row(rgba.row(y), expected.data(), 70);
    for (int x = 0; x < 70; ++x) CHECK(gray(x, y) == expected[static_cast<std::size_t>(x)]);
  }
  const r3d::ImageU8 two(4, 4, 2);
  CHECK_THROWS_AS(static_cast<void>(r3d::to_gray(two.view())), r3d::Error);
}
