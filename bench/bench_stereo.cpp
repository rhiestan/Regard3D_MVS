// Throughput of the Census-SGM stages on synthetic images (plan section 6,
// Phases 2 and 3): Census transform, cost aggregation of one tile, and the
// complete matcher over the full range and hierarchically, single-threaded
// (per core) and with all threads.
//
// Usage: bench_stereo [-v] [-m] [width height disparities]
//   -v  debug log (per-stage timings of the matcher)
//   -m  complete matcher only
//   -H  hierarchical matcher only

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "r3d/core/image.hpp"
#include "r3d/core/log.hpp"
#include "r3d/core/simd.hpp"
#include "r3d/core/thread_pool.hpp"
#include "r3d/core/timer.hpp"
#include "r3d/stereo/census.hpp"
#include "r3d/stereo/sgm_kernels.hpp"
#include "r3d/stereo/stereo_matcher.hpp"

namespace {

// Multi-scale noise: bilinearly interpolated random grids with spacings of
// 1 .. 64 pixels and equal amplitudes, so that every pyramid level has texture.
r3d::ImageU16 random_image(int w, int h, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> sum(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0.0F);
  for (int s = 1; s <= 64; s *= 2) {
    const int gw = w / s + 2;
    const int gh = h / s + 2;
    std::vector<float> grid(static_cast<std::size_t>(gw) * static_cast<std::size_t>(gh));
    for (float& g : grid) {
      g = dist(rng);
    }
    for (int y = 0; y < h; ++y) {
      const int gy = y / s;
      const float fy = static_cast<float>(y % s) / static_cast<float>(s);
      for (int x = 0; x < w; ++x) {
        const int gx = x / s;
        const float fx = static_cast<float>(x % s) / static_cast<float>(s);
        auto at = [&](int i, int j) { return grid[static_cast<std::size_t>(j) * static_cast<std::size_t>(gw) +
                                                  static_cast<std::size_t>(i)]; };
        const float top = at(gx, gy) + fx * (at(gx + 1, gy) - at(gx, gy));
        const float bottom = at(gx, gy + 1) + fx * (at(gx + 1, gy + 1) - at(gx, gy + 1));
        sum[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)] +=
            top + fy * (bottom - top);
      }
    }
  }
  r3d::ImageU16 img(w, h, 1);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float v = sum[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)];
      img(x, y) = static_cast<std::uint16_t>(std::clamp(32768.0F + v * 8000.0F, 0.0F, 65535.0F));
    }
  }
  return img;
}

// Best of `runs` timings of f().
template <typename F>
double best_seconds(int runs, F&& f) {
  double best = 1e30;
  for (int i = 0; i < runs; ++i) {
    const r3d::Stopwatch watch;
    f();
    best = std::min(best, watch.elapsed_seconds());
  }
  return best;
}

// Uniform labels 0 .. num - 1 for a row of `width` pixels.
struct UniformRow {
  std::vector<std::int32_t> lo;
  std::vector<std::uint16_t> num;
  std::vector<std::size_t> offset;
  UniformRow(std::size_t width, std::size_t n) : lo(width, 0), num(width, static_cast<std::uint16_t>(n)), offset(width + 1) {
    for (std::size_t x = 0; x <= width; ++x) {
      offset[x] = x * n;
    }
  }
  [[nodiscard]] r3d::sgm::RowLabels labels() const { return {lo.data(), num.data(), offset.data()}; }
};

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> positional;
  bool matcher_only = false;
  bool hierarchical_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-v") {
      r3d::set_log_level(r3d::LogLevel::debug);
    } else if (a == "-H") {
      matcher_only = hierarchical_only = true;
    } else if (a == "-m") {
      matcher_only = true;
    } else {
      positional.push_back(a);
    }
  }
  const int w = positional.size() >= 3 ? std::atoi(positional[0].c_str()) : 1024;
  const int h = positional.size() >= 3 ? std::atoi(positional[1].c_str()) : 768;
  const int nd = positional.size() >= 3 ? std::atoi(positional[2].c_str()) : 128;
  const r3d::ImageU16 left = random_image(w, h, 1);
  // The right image shows the left one with a smooth disparity field
  // (nd / 8 .. nd * 5 / 8), so that the hierarchy has something to follow.
  r3d::ImageU16 right(w, h, 1);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int d = nd / 8 + (nd / 2) * (x + y) / (w + h);
      right(x, y) = left(std::min(w - 1, x + d), y);
    }
  }
  const double pixels = static_cast<double>(w) * h;
  const double cells = pixels * nd;
  std::printf("%dx%d, %d disparities, SIMD %s, %u threads\n", w, h, nd, r3d::simd_active_target_name().c_str(),
              r3d::default_thread_pool().num_threads());

  r3d::StereoOptions options;
  options.tile_size = 4096;  // one tile: pure aggregation throughput
  options.levels = 0;
  const unsigned threads = r3d::default_thread_pool().num_threads();
  r3d::set_default_thread_count(1);
  for (const bool scalar : {true, false}) {
    if (matcher_only) {
      break;
    }
    options.scalar_kernels = scalar;
    const char* name = scalar ? "scalar" : "simd  ";
    const double census = best_seconds(3, [&] { (void)r3d::census_transform(left.view(), scalar); });
    const double volume =
        best_seconds(2, [&] { (void)r3d::detail::sgm_cost_volume(left.view(), right.view(), 0, nd, options); });
    std::printf("  1 thread %s  census %7.1f Mpixel/s   cost + 8-path aggregation %7.1f Mcell/s\n", name,
                pixels / census * 1e-6, cells / volume * 1e-6);
  }
  if (!matcher_only) {
    // Cost rows alone (computed twice per pixel by the aggregation).
    const r3d::Image<std::uint64_t> cl = r3d::census_transform(left.view());
    std::vector<std::uint64_t> rev(static_cast<std::size_t>(w + nd));
    std::vector<std::uint16_t> cost(static_cast<std::size_t>(w) * static_cast<std::size_t>(nd));
    const UniformRow row(static_cast<std::size_t>(w), static_cast<std::size_t>(nd));
    for (const bool scalar : {true, false}) {
      const double t = best_seconds(3, [&] {
        for (int y = 0; y < h; ++y) {
          (scalar ? r3d::sgm::scalar::census_costs : r3d::sgm::census_costs)(
              cl.row(y), rev.data(), static_cast<std::size_t>(w), row.labels(), 0, cost.data());
        }
      });
      std::printf("  1 thread %s  cost rows %7.1f Mcell/s\n", scalar ? "scalar" : "simd  ", cells / t * 1e-6);
    }
  }
  if (!matcher_only) {
    // SGM row kernel alone on a short row whose buffers stay in L1/L2.
    const std::size_t rw = 32;
    const auto d = static_cast<std::size_t>(nd);
    const std::size_t stride = r3d::sgm::path_stride(d);
    std::vector<std::uint16_t> cost(rw * d, 7), p2(4 * rw, 200), prev(3 * rw * stride, 0), cur(3 * rw * stride, 0),
        prev_min(3 * rw, 0), cur_min(3 * rw, 0), scratch(r3d::sgm::kScratchBuffers * stride, 0), sum(rw * d, 0);
    const UniformRow row(rw, d);
    r3d::sgm::SgmRowArgs args;
    args.cost = cost.data();
    args.p2 = p2.data();
    args.prev = prev.data();
    args.prev_min = prev_min.data();
    args.cur = cur.data();
    args.cur_min = cur_min.data();
    args.scratch = scratch.data();
    args.sum = sum.data();
    args.width = rw;
    args.labels = row.labels();
    args.prev_labels = row.labels();
    args.has_prev = true;
    args.max_num = d;
    args.p1 = 28;
    const int reps = static_cast<int>(cells / static_cast<double>(rw * d));
    for (const bool scalar : {true, false}) {
      const double t = best_seconds(3, [&] {
        for (int i = 0; i < reps; ++i) {
          (scalar ? r3d::sgm::scalar::sgm_row : r3d::sgm::sgm_row)(args);
        }
      });
      std::printf("  1 thread %s  4-path row kernel %7.1f Mcell/s\n", scalar ? "scalar" : "simd  ", cells / t * 1e-6);
    }
  }
  // Complete matcher: full range (levels = 0) and hierarchical (automatic
  // levels). Mcell/s counts the cells of the full range in both cases.
  for (const int levels : {0, -1}) {
    if (hierarchical_only && levels == 0) {
      continue;
    }
    options = r3d::StereoOptions{};
    options.levels = levels;
    r3d::set_default_thread_count(1);
    r3d::DisparityMap result;
    const double t1 =
        best_seconds(2, [&] { result = r3d::match_stereo(left.view(), right.view(), {}, {}, 0, nd, options); });
    r3d::set_default_thread_count(threads);
    const double tn =
        best_seconds(3, [&] { (void)r3d::match_stereo(left.view(), right.view(), {}, {}, 0, nd, options); });
    std::printf("  %s (%d levels, %.1f cells per pixel): 1 thread %7.1f Mcell/s (%.3f s), %u threads %7.1f Mcell/s "
                "(%.3f s)\n",
                levels == 0 ? "full-range matcher  " : "hierarchical matcher", result.levels,
                static_cast<double>(result.cells) / pixels, cells / t1 * 1e-6, t1, threads, cells / tn * 1e-6, tn);
  }
  return 0;
}
