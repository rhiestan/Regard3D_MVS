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

TEST_CASE("SGM cost volume with per-pixel label ranges equals the textbook implementation") {
  const Texture tex(13);
  r3d::ImageU16 left, right;
  r3d::ImageF32 gt;
  const int w = 37, h = 19, nd = 96;
  render_pair(tex, w, h, [](double x, double y) { return 20.0 + 0.5 * x - 0.3 * y; }, left, right, gt);
  r3d::test::Rng rng(31);
  // Random ranges: neighbours overlap partly, contain each other or are disjoint.
  r3d::LabelRanges ranges{r3d::Image<std::uint16_t>(w, h, 1), r3d::Image<std::uint16_t>(w, h, 1)};
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const int n = 16 * rng.uniform_int(1, 3);
      ranges.num(x, y) = static_cast<std::uint16_t>(n);
      ranges.lo(x, y) = static_cast<std::uint16_t>(rng.uniform_int(0, nd - n));
    }
  r3d::StereoOptions options;
  options.p2 = 150;
  options.p2_adapt = 8;
  const std::vector<std::uint16_t> expected =
      r3d::detail::sgm_cost_volume_reference(left.view(), right.view(), -2, nd, options, &ranges);
  options.scalar_kernels = true;
  CHECK(r3d::detail::sgm_cost_volume(left.view(), right.view(), -2, nd, options, &ranges) == expected);
  options.scalar_kernels = false;
  for (const std::int64_t target : r3d::simd_available_targets()) {
    const r3d::ScopedSimdTarget scoped(target);
    CAPTURE(r3d::simd_target_name(target));
    CHECK(r3d::detail::sgm_cost_volume(left.view(), right.view(), -2, nd, options, &ranges) == expected);
  }
  // Full ranges given explicitly give the same volume as no ranges.
  const r3d::LabelRanges full = r3d::detail::full_ranges(w, h, nd);
  CHECK(r3d::detail::sgm_cost_volume(left.view(), right.view(), -2, nd, options, &full) ==
        r3d::detail::sgm_cost_volume(left.view(), right.view(), -2, nd, options));
}

TEST_CASE("WTA row kernel: every SIMD target is bit-exact with scalar") {
  r3d::test::Rng rng(17);
  for (const int max_n : {1, 2, 3, 5}) {
    const std::size_t w = 37;
    std::vector<std::int32_t> lo(w);
    std::vector<std::uint16_t> num(w);
    std::vector<std::size_t> offset(w + 1, 0);
    std::ptrdiff_t shift = 0, top = 0;
    for (std::size_t x = 0; x < w; ++x) {
      num[x] = static_cast<std::uint16_t>(16 * rng.uniform_int(1, max_n));
      lo[x] = rng.uniform_int(0, 100);
      offset[x + 1] = offset[x] + num[x];
      shift = std::max<std::ptrdiff_t>(shift, lo[x] + num[x] - 1 - static_cast<std::ptrdiff_t>(x));
      top = std::max<std::ptrdiff_t>(top, static_cast<std::ptrdiff_t>(x) - lo[x]);
    }
    const r3d::sgm::RowLabels labels{lo.data(), num.data(), offset.data()};
    const auto nr = static_cast<std::size_t>(top + shift + 1);
    std::vector<std::uint16_t> sum(offset[w]);
    for (auto& v : sum) v = static_cast<std::uint16_t>(rng.uniform_int(0, 40));  // small range: many ties
    std::vector<std::uint16_t> best(w), cost(w), second(w), rc(nr, 0xFFFF), rb(nr, 0xFFFF);
    r3d::sgm::scalar::wta_row(sum.data(), w, labels, best.data(), cost.data(), second.data(), rc.data(), rb.data(),
                              shift);
    for (const std::int64_t target : r3d::simd_available_targets()) {
      const r3d::ScopedSimdTarget scoped(target);
      CAPTURE(r3d::simd_target_name(target));
      CAPTURE(max_n);
      std::vector<std::uint16_t> best2(w), cost2(w), second2(w), rc2(nr, 0xFFFF), rb2(nr, 0xFFFF);
      r3d::sgm::wta_row(sum.data(), w, labels, best2.data(), cost2.data(), second2.data(), rc2.data(), rb2.data(),
                        shift);
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

TEST_CASE("Left-right checks with different image widths and negative disparities") {
  // Rectified pairs have different widths; the right-to-left match runs on
  // the mirrored pair, whose disparities are offset by the width difference.
  const Texture tex(23);
  const int wl = 150, wr = 190, h = 60;
  const double d = -7.0;  // x_left - x_right
  r3d::ImageU16 left(wl, h, 1), right(wr, h, 1);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < wl; ++x) left(x, y) = to_u16(tex(x, y));
    for (int x = 0; x < wr; ++x) right(x, y) = to_u16(tex(x + d, y));
  }
  for (const int levels : {0, 1}) {
    for (const bool full : {false, true}) {
      CAPTURE(levels);
      CAPTURE(full);
      r3d::StereoOptions options;
      options.lr_full_match = full;
      options.levels = levels;
      const r3d::DisparityMap result = r3d::match_stereo(left.view(), right.view(), {}, {}, -20, 32, options);
      CHECK(result.levels == levels);
      int good = 0;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < wl; ++x) good += std::abs(result.disparity(x, y) - static_cast<float>(d)) < 0.25F ? 1 : 0;
      CHECK(good > 0.9 * wl * h);
    }
  }
}

TEST_CASE("Per-pixel label ranges from a coarser level") {
  // Coarse disparities 10 (left half) and 30 (right half), one invalid
  // coarse pixel, and an invalid region larger than the fill distance.
  const int wc = 40, hc = 20;
  r3d::ImageF32 coarse(wc, hc, 1);
  for (int y = 0; y < hc; ++y)
    for (int x = 0; x < wc; ++x) coarse(x, y) = x < wc / 2 ? 10.0F : 30.0F;
  coarse(5, 5) = r3d::kInvalidDisparity;
  for (int y = 0; y < hc; ++y)
    for (int x = 30; x < wc; ++x) coarse(x, y) = r3d::kInvalidDisparity;
  const int w = 2 * wc, h = 2 * hc, dmin = 4, nd = 96;
  r3d::ImageU8 valid(w, h, 1);
  valid.fill(255);
  valid(1, 1) = 0;
  r3d::StereoOptions options;
  options.range_radius = 1;
  options.range_margin = 3;
  options.range_fill = 4;
  // A right base image with a wide partner: no clipping.
  const r3d::LabelRanges r =
      r3d::detail::ranges_from_coarse(coarse, w, h, valid.view(), {}, 1000, 1, dmin, nd, options);
  auto covers = [&](int x, int y, int lo, int hi) {  // labels [lo, hi]
    return r.lo(x, y) <= lo && r.lo(x, y) + r.num(x, y) - 1 >= hi;
  };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      CAPTURE(x);
      CAPTURE(y);
      CHECK(r.num(x, y) % 16 == 0);
      CHECK(r.lo(x, y) >= 0);
      CHECK(r.lo(x, y) + r.num(x, y) <= nd);
    }
  // Far from the step: 2 * 10 - dmin +- margin in the smallest range.
  CHECK(covers(10, 30, 20 - dmin - 3, 20 - dmin + 3));
  CHECK(r.num(10, 30) == 16);
  // The invalid coarse pixel takes its neighbours' range.
  CHECK(covers(10, 10, 20 - dmin - 3, 20 - dmin + 3));
  CHECK(r.num(10, 10) == 16);
  // At the step both disparities are searched.
  CHECK(covers(39, 20, 20 - dmin - 3, 60 - dmin + 3));
  // Masked pixels get the smallest range.
  CHECK(r.num(1, 1) == 16);
  // Fill: a few pixels into the invalid region the right half's range is
  // used, beyond the fill distance the full range.
  CHECK(covers(62, 20, 60 - dmin - 3, 60 - dmin + 3));
  CHECK(r.num(62, 20) == 16);
  CHECK(r.lo(78, 20) == 0);
  CHECK(r.num(78, 20) == nd);
  // A left base image with a partner of the same width: pixel x matches
  // partner column x - d, so x = 10 (d = 20) has no possible match and the
  // full ranges of the invalid region are clipped to d <= x.
  const r3d::LabelRanges c =
      r3d::detail::ranges_from_coarse(coarse, w, h, valid.view(), {}, w, -1, dmin, nd, options);
  CHECK(c.num(10, 30) == 16);
  CHECK(c.lo(30, 30) <= 20 - dmin - 3);
  CHECK(c.lo(30, 30) + c.num(30, 30) - 1 >= 20 - dmin + 3);
  CHECK(c.lo(78, 20) + c.num(78, 20) - 1 < 78 - dmin + 16);
}

TEST_CASE("Hierarchical matching equals full-range matching on a wide disparity range") {
  // Two slanted planes with a large depth step: the foreground (x < 300)
  // has disparities around 150, the background around 46, of 256 labels.
  const Texture tex(29), hidden(31);
  const int w = 480, h = 300, step = 300;
  auto d_fg = [](double x, double y) { return 150.0 + 0.1 * y - 0.05 * x; };
  auto d_bg = [](double x, double) { return 40.0 + 0.02 * x; };
  r3d::ImageU16 left(w, h, 1), right(w, h, 1);
  r3d::ImageF32 gt(w, h, 1);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      left(x, y) = to_u16(tex(x, y));
      gt(x, y) = static_cast<float>(x < step ? d_fg(x, y) : d_bg(x, y));
      // Right pixel x shows the foreground point with x_left - d_fg = x if
      // it exists (it occludes), else the background point; background
      // points hidden in the left view (x_left < step) have their own texture.
      double xf = x;
      for (int it = 0; it < 30; ++it) xf = x + d_fg(xf, y);
      if (xf < step) {
        right(x, y) = to_u16(tex(xf, y));
        continue;
      }
      double xb = x;
      for (int it = 0; it < 30; ++it) xb = x + d_bg(xb, y);
      right(x, y) = to_u16(xb >= step ? tex(xb, y) : hidden(xb, y));
    }
  }
  // Score the matchable, non-occluded parts (the left border lacks matches).
  r3d::ImageU8 mask(w, h, 1);
  for (int y = 0; y < h; ++y)
    for (int x = 200; x < w; ++x) mask(x, y) = std::abs(x - step) > 12 ? 255 : 0;
  r3d::StereoOptions options;
  options.levels = 0;
  const r3d::DisparityMap full = r3d::match_stereo(left.view(), right.view(), {}, {}, 0, 256, options);
  options.levels = -1;
  const r3d::DisparityMap hier = r3d::match_stereo(left.view(), right.view(), {}, {}, 0, 256, options);
  CHECK(full.levels == 0);
  CHECK(hier.levels >= 1);
  CHECK(hier.cells * 3 < full.cells);
  const r3d::DisparityErrorStats sf = r3d::evaluate_disparity(full.disparity, gt, mask.view());
  const r3d::DisparityErrorStats sh = r3d::evaluate_disparity(hier.disparity, gt, mask.view());
  CHECK(sf.density > 0.95);
  CHECK(sf.bad1 < 0.01);
  CHECK(sh.density > 0.95);
  CHECK(sh.bad1 < 0.01);
  CHECK(sh.density >= sf.density - 0.01);
  CHECK(sh.avg_error <= sf.avg_error + 0.02);
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
