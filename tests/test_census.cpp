#include <cstdint>
#include <random>

#include <doctest/doctest.h>

#include "r3d/core/image.hpp"
#include "r3d/core/simd.hpp"
#include "r3d/stereo/census.hpp"

namespace {

r3d::ImageU16 random_image(int w, int h, unsigned seed, int levels = 65536) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, levels - 1);
  r3d::ImageU16 img(w, h, 1);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      img(x, y) = static_cast<std::uint16_t>(dist(rng));
    }
  }
  return img;
}

}  // namespace

TEST_CASE("Census bits follow the documented order") {
  r3d::ImageU16 img(9, 7, 1);
  img.fill(100);
  CHECK(r3d::census_transform(img.view())(4, 3) == 0);  // no neighbour darker

  img(4, 3) = 200;  // bright centre: all 62 neighbours darker
  CHECK(r3d::census_transform(img.view())(4, 3) == (std::uint64_t{1} << 62) - 1);

  img.fill(100);
  img(0, 0) = 50;  // first neighbour in raster order -> bit 61
  CHECK(r3d::census_transform(img.view())(4, 3) == std::uint64_t{1} << 61);
  img(0, 0) = 100;
  img(8, 6) = 50;  // last neighbour -> bit 0
  CHECK(r3d::census_transform(img.view())(4, 3) == 1);
  img(8, 6) = 100;
  img(5, 3) = 50;  // right of the centre: neighbour index 31 -> bit 30
  CHECK(r3d::census_transform(img.view())(4, 3) == std::uint64_t{1} << 30);
}

TEST_CASE("Census borders are replicated") {
  // A horizontal ramp: at the left border the replicated column equals the
  // centre (not darker), so the bits of window columns 0..3 are clear.
  r3d::ImageU16 img(20, 10, 1);
  for (int y = 0; y < 10; ++y) {
    for (int x = 0; x < 20; ++x) {
      img(x, y) = static_cast<std::uint16_t>(10 * x);
    }
  }
  const r3d::Image<std::uint64_t> c = r3d::census_transform(img.view());
  CHECK(c(0, 5) == 0);
  // Interior: the 4 columns left of the centre (28 neighbours) are darker.
  std::uint64_t expected = 0;
  for (int j = 0; j < 7; ++j) {
    for (int i = 0; i < 9; ++i) {
      if (j == 3 && i == 4) continue;
      expected = (expected << 1) | (i < 4 ? 1U : 0U);
    }
  }
  CHECK(c(10, 5) == expected);
}

TEST_CASE("Census: every SIMD target is bit-exact with scalar") {
  for (const int w : {1, 5, 16, 31, 64, 77}) {
    const r3d::ImageU16 img = random_image(w, 13, static_cast<unsigned>(w), w % 2 == 0 ? 8 : 65536);
    const r3d::Image<std::uint64_t> expected = r3d::census_transform(img.view(), true);
    for (const std::int64_t target : r3d::simd_available_targets()) {
      const r3d::ScopedSimdTarget scoped(target);
      CAPTURE(r3d::simd_target_name(target));
      CAPTURE(w);
      const r3d::Image<std::uint64_t> actual = r3d::census_transform(img.view());
      bool equal = true;
      for (int y = 0; y < img.height(); ++y) {
        for (int x = 0; x < w; ++x) {
          equal = equal && actual(x, y) == expected(x, y);
        }
      }
      CHECK(equal);
    }
  }
}
