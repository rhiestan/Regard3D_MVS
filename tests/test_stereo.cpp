#include <cmath>
#include <cstdint>
#include <vector>

#include <doctest/doctest.h>

#include "r3d/core/image.hpp"
#include "r3d/core/simd.hpp"
#include "r3d/core/thread_pool.hpp"
#include "r3d/io/pfm.hpp"
#include "r3d/stereo/disparity_eval.hpp"
#include "r3d/stereo/sgm_kernels.hpp"
#include "r3d/stereo/stereo_matcher.hpp"
#include "test_util.hpp"

namespace {

// Smooth random texture: sum of random sinusoids, values in about [0, 65535].
struct Texture {
  std::vector<double> fx, fy, phase, amp;
  explicit Texture(unsigned seed) {
    r3d::test::Rng rng(seed);
    for (int i = 0; i < 24; ++i) {
      const double fx_mag = rng.uniform(0.05, 0.9);
      fx.push_back(fx_mag * rng.uniform(-1.0, 1.0));
      const double fy_mag = rng.uniform(0.05, 0.9);
      fy.push_back(fy_mag * rng.uniform(-1.0, 1.0));
      phase.push_back(rng.uniform(0.0, 6.283));
      amp.push_back(1.0 / (1.0 + i * 0.2));
    }
  }
  [[nodiscard]] double operator()(double x, double y) const {
    double v = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < fx.size(); ++i) {
      v += amp[i] * std::sin(fx[i] * x + fy[i] * y + phase[i]);
      norm += amp[i];
    }
    return 32768.0 + 30000.0 * v / norm * 2.0;
  }
};

std::uint16_t to_u16(double v) { return static_cast<std::uint16_t>(std::lround(std::clamp(v, 0.0, 65535.0))); }

// Rectified pair of a scene with disparity d(x, y) for the left image: the
// right image shows texture(x_right) with x_right = x_left - d.
template <typename Disp>
void render_pair(const Texture& tex, int w, int h, Disp&& disparity, r3d::ImageU16& left, r3d::ImageU16& right,
                 r3d::ImageF32& gt) {
  left = r3d::ImageU16(w, h, 1);
  right = r3d::ImageU16(w, h, 1);
  gt = r3d::ImageF32(w, h, 1);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      left(x, y) = to_u16(tex(x, y));
      gt(x, y) = static_cast<float>(disparity(static_cast<double>(x), static_cast<double>(y)));
      // Right pixel xr shows the surface point whose left position xl solves
      // xl - d(xl) = xr; a few fixed-point iterations suffice for slopes < 1.
      double xl = x;
      for (int it = 0; it < 20; ++it) {
        xl = x + disparity(xl, static_cast<double>(y));
      }
      right(x, y) = to_u16(tex(xl, y));
    }
  }
}

r3d::ImageU16 random_image(int w, int h, unsigned seed) {
  r3d::test::Rng rng(seed);
  r3d::ImageU16 img(w, h, 1);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) img(x, y) = static_cast<std::uint16_t>(rng.uniform_int(0, 65535));
  return img;
}

bool same_disparities(const r3d::ImageF32& a, const r3d::ImageF32& b) {
  if (a.width() != b.width() || a.height() != b.height()) return false;
  for (int y = 0; y < a.height(); ++y)
    for (int x = 0; x < a.width(); ++x) {
      const float u = a(x, y), v = b(x, y);
      if (!(u == v || (!std::isfinite(u) && !std::isfinite(v)))) return false;
    }
  return true;
}

}  // namespace

TEST_CASE("SGM cost volume equals the textbook implementation on every SIMD target") {
  // Correlated images (right = shifted left plus noise) so that paths matter.
  const Texture tex(3);
  r3d::ImageU16 left, right;
  r3d::ImageF32 gt;
  render_pair(tex, 41, 23, [](double x, double) { return 5.0 + 0.1 * x; }, left, right, gt);
  const r3d::ImageU16 noise = random_image(41, 23, 9);
  for (int y = 0; y < 23; ++y)
    for (int x = 0; x < 41; ++x) right(x, y) = static_cast<std::uint16_t>(right(x, y) ^ (noise(x, y) & 0x0F00));

  for (const int p2_adapt : {0, 12}) {
    for (const auto& [dmin, nd] : {std::pair{-3, 16}, std::pair{2, 48}}) {
      CAPTURE(p2_adapt);
      CAPTURE(dmin);
      CAPTURE(nd);
      r3d::StereoOptions options;
      options.p2_adapt = p2_adapt;
      options.p2 = 120;
      const std::vector<std::uint16_t> expected =
          r3d::detail::sgm_cost_volume_reference(left.view(), right.view(), dmin, nd, options);
      options.scalar_kernels = true;
      CHECK(r3d::detail::sgm_cost_volume(left.view(), right.view(), dmin, nd, options) == expected);
      options.scalar_kernels = false;
      for (const std::int64_t target : r3d::simd_available_targets()) {
        const r3d::ScopedSimdTarget scoped(target);
        CAPTURE(r3d::simd_target_name(target));
        CHECK(r3d::detail::sgm_cost_volume(left.view(), right.view(), dmin, nd, options) == expected);
      }
    }
  }
}

TEST_CASE("WTA row kernel: every SIMD target is bit-exact with scalar") {
  r3d::test::Rng rng(17);
  for (const std::size_t nd : {16u, 32u, 48u, 80u}) {
    const std::size_t w = 37;
    std::vector<std::uint16_t> sum(w * nd);
    for (auto& v : sum) v = static_cast<std::uint16_t>(rng.uniform_int(0, 40));  // small range: many ties
    std::vector<std::uint16_t> best(w), cost(w), second(w), rc(w + nd - 1, 0xFFFF), rb(w + nd - 1, 0xFFFF);
    r3d::sgm::scalar::wta_row(sum.data(), w, nd, best.data(), cost.data(), second.data(), rc.data(), rb.data());
    for (const std::int64_t target : r3d::simd_available_targets()) {
      const r3d::ScopedSimdTarget scoped(target);
      CAPTURE(r3d::simd_target_name(target));
      CAPTURE(nd);
      std::vector<std::uint16_t> best2(w), cost2(w), second2(w), rc2(w + nd - 1, 0xFFFF), rb2(w + nd - 1, 0xFFFF);
      r3d::sgm::wta_row(sum.data(), w, nd, best2.data(), cost2.data(), second2.data(), rc2.data(), rb2.data());
      CHECK(best2 == best);
      CHECK(cost2 == cost);
      CHECK(second2 == second);
      CHECK(rc2 == rc);
      CHECK(rb2 == rb);
    }
  }
}

TEST_CASE("Stereo matching recovers a slanted plane with sub-pixel accuracy") {
  const Texture tex(11);
  r3d::ImageU16 left, right;
  r3d::ImageF32 gt;
  const int w = 200, h = 120;
  render_pair(tex, w, h, [](double x, double y) { return 12.3 + 0.04 * x + 0.02 * y; }, left, right, gt);
  for (const auto method : {r3d::SubpixelMethod::parabola, r3d::SubpixelMethod::equiangular}) {
    r3d::StereoOptions options;
    options.subpixel = method;
    const r3d::DisparityMap result = r3d::match_stereo(left.view(), right.view(), {}, {}, 0, 32, options);
    CHECK(result.num_disparities == 32);
    // Skip the left border, where the right image has no match.
    r3d::ImageU8 mask(w, h, 1);
    for (int y = 0; y < h; ++y)
      for (int x = 30; x < w; ++x) mask(x, y) = 255;
    const r3d::DisparityErrorStats s = r3d::evaluate_disparity(result.disparity, gt, mask.view());
    CAPTURE(static_cast<int>(method));
    CHECK(s.density > 0.95);
    CHECK(s.bad1 < 0.01);
    CHECK(s.avg_error < 0.2);
  }
}

TEST_CASE("Stereo matching: tiles, threads and SIMD targets give consistent results") {
  const Texture tex(5);
  r3d::ImageU16 left, right;
  r3d::ImageF32 gt;
  const int w = 150, h = 90;
  // Two planes with a depth step: a discontinuity crossing the tile borders.
  render_pair(tex, w, h, [](double x, double) { return x < 70 ? 6.0 : 14.5; }, left, right, gt);
  r3d::StereoOptions options;
  options.tile_size = 48;
  options.tile_margin = 8;
  options.tile_blend = 8;

  options.scalar_kernels = true;
  const r3d::DisparityMap reference = r3d::match_stereo(left.view(), right.view(), {}, {}, 0, 32, options);
  options.scalar_kernels = false;
  for (const std::int64_t target : r3d::simd_available_targets()) {
    const r3d::ScopedSimdTarget scoped(target);
    CAPTURE(r3d::simd_target_name(target));
    CHECK(same_disparities(r3d::match_stereo(left.view(), right.view(), {}, {}, 0, 32, options).disparity,
                           reference.disparity));
  }

  r3d::set_default_thread_count(1);
  const r3d::DisparityMap single = r3d::match_stereo(left.view(), right.view(), {}, {}, 0, 32, options);
  r3d::set_default_thread_count(0);
  CHECK(same_disparities(single.disparity, reference.disparity));

  // Tiled and untiled results agree away from the step.
  options.tile_size = 1024;
  const r3d::DisparityMap whole = r3d::match_stereo(left.view(), right.view(), {}, {}, 0, 32, options);
  r3d::ImageU8 mask(w, h, 1);
  for (int y = 0; y < h; ++y)
    for (int x = 20; x < w; ++x) mask(x, y) = std::abs(x - 70) > 12 ? 255 : 0;
  const r3d::DisparityErrorStats tiled_stats = r3d::evaluate_disparity(reference.disparity, gt, mask.view());
  const r3d::DisparityErrorStats whole_stats = r3d::evaluate_disparity(whole.disparity, gt, mask.view());
  CHECK(tiled_stats.density > 0.95);
  CHECK(whole_stats.density > 0.95);
  CHECK(tiled_stats.bad1 < 0.01);
  CHECK(whole_stats.bad1 < 0.01);
}

TEST_CASE("Masks: invalid left pixels stay invalid, matches into invalid right pixels are rejected") {
  const Texture tex(8);
  r3d::ImageU16 left, right;
  r3d::ImageF32 gt;
  render_pair(tex, 120, 60, [](double, double) { return 10.0; }, left, right, gt);
  r3d::ImageU8 lmask(120, 60, 1), rmask(120, 60, 1);
  lmask.fill(255);
  rmask.fill(255);
  for (int y = 0; y < 60; ++y) {
    for (int x = 100; x < 120; ++x) lmask(x, y) = 0;
    for (int x = 0; x < 40; ++x) rmask(x, y) = 0;
  }
  const r3d::DisparityMap result = r3d::match_stereo(left.view(), right.view(), lmask.view(), rmask.view(), 0, 32);
  for (int y = 0; y < 60; ++y) {
    for (int x = 100; x < 120; ++x) CHECK_FALSE(std::isfinite(result.disparity(x, y)));
    for (int x = 0; x < 40; ++x) CHECK_FALSE(std::isfinite(result.disparity(x, y)));  // every x - d < 40
  }
  int good = 0;
  for (int y = 0; y < 60; ++y)
    for (int x = 55; x < 95; ++x) good += std::abs(result.disparity(x, y) - 10.0F) < 0.25F ? 1 : 0;
  CHECK(good > 0.95 * 60 * 40);
}

TEST_CASE("Masked left pixels do not disturb the left-right check of their neighbours") {
  // Masked pixels have zero matching costs; they must not win the right-image
  // WTA of the diagonal left-right check (regression: whole tiles were lost).
  const Texture tex(21);
  r3d::ImageU16 left, right;
  r3d::ImageF32 gt;
  render_pair(tex, 160, 60, [](double, double) { return 10.0; }, left, right, gt);
  // Noise, so that true matches have non-zero costs.
  r3d::test::Rng rng(4);
  for (int y = 0; y < 60; ++y)
    for (int x = 0; x < 160; ++x) {
      left(x, y) = to_u16(left(x, y) + rng.normal(0.0, 3000.0));
      right(x, y) = to_u16(right(x, y) + rng.normal(0.0, 3000.0));
    }
  // Masked top-right wedge (as left by rectification).
  r3d::ImageU8 lmask(160, 60, 1);
  int valid = 0;
  for (int y = 0; y < 60; ++y)
    for (int x = 0; x < 160; ++x)
      if (x < 70 + 2 * y) {
        lmask(x, y) = 255;
        valid += x >= 20 ? 1 : 0;
      }
  for (const bool full : {false, true}) {
    CAPTURE(full);
    r3d::StereoOptions options;
    options.lr_full_match = full;
    options.tile_size = 64;
    const r3d::DisparityMap result = r3d::match_stereo(left.view(), right.view(), lmask.view(), {}, 0, 256, options);
    int good = 0;
    for (int y = 0; y < 60; ++y)
      for (int x = 20; x < 160; ++x)
        good += lmask(x, y) != 0 && std::abs(result.disparity(x, y) - 10.0F) < 0.5F ? 1 : 0;
    CHECK(good > 0.75 * valid);  // 0.89-0.91 (without the fix: about 0.45)
  }
}

TEST_CASE("Speckle and median filters") {
  r3d::ImageF32 d(20, 10, 1);
  d.fill(5.0F);
  d(3, 3) = 20.0F;  // isolated outlier
  d(10, 5) = r3d::kInvalidDisparity;
  for (int x = 15; x < 20; ++x) d(x, 0) = 30.0F;  // small segment of 5 pixels
  r3d::ImageF32 s = d;
  r3d::remove_speckles(s, 6, 1.0F);
  CHECK_FALSE(std::isfinite(s(3, 3)));
  CHECK_FALSE(std::isfinite(s(17, 0)));
  CHECK(s(0, 0) == 5.0F);
  r3d::remove_speckles(s = d, 5, 1.0F);
  CHECK(s(17, 0) == 30.0F);

  const r3d::ImageF32 m = r3d::median_filter_3x3(d);
  CHECK(m(3, 3) == 5.0F);
  CHECK_FALSE(std::isfinite(m(10, 5)));
  CHECK(m(11, 5) == 5.0F);
}

TEST_CASE("Disparity evaluation, hole filling and PFM round trip") {
  r3d::ImageF32 gt(4, 1, 1), est(4, 1, 1);
  gt(0, 0) = 1.0F;
  gt(1, 0) = 2.0F;
  gt(2, 0) = 3.0F;
  gt(3, 0) = r3d::kInvalidDisparity;
  est(0, 0) = 1.2F;
  est(1, 0) = 5.0F;
  est(2, 0) = r3d::kInvalidDisparity;
  est(3, 0) = 9.0F;
  const r3d::DisparityErrorStats s = r3d::evaluate_disparity(est, gt, {});
  CHECK(s.num_evaluated == 3);
  CHECK(s.density == doctest::Approx(2.0 / 3.0));
  CHECK(s.bad2 == doctest::Approx(0.5));
  CHECK(s.avg_error == doctest::Approx((0.2 + 3.0) / 2).epsilon(1e-5));

  r3d::ImageF32 holes(6, 1, 1);
  holes.fill(r3d::kInvalidDisparity);
  holes(1, 0) = 4.0F;
  holes(4, 0) = 2.0F;
  const r3d::ImageF32 filled = r3d::fill_disparity_holes(holes);
  CHECK(filled(0, 0) == 4.0F);
  CHECK(filled(2, 0) == 2.0F);
  CHECK(filled(5, 0) == 2.0F);

  const r3d::test::TempDir dir;
  r3d::ImageF32 img(5, 3, 1);
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 5; ++x) img(x, y) = static_cast<float>(x * 10 + y) + 0.25F;
  img(2, 1) = r3d::kInvalidDisparity;
  r3d::write_pfm(dir.path() / "a.pfm", img.view());
  const r3d::ImageF32 back = r3d::read_pfm(dir.path() / "a.pfm");
  CHECK(same_disparities(back, img));
}
