// Throughput of the Census-SGM stages on synthetic images (plan section 6,
// Phase 2): Census transform, cost aggregation of one tile, and the complete
// matcher, single-threaded (per core) and with all threads.
//
// Usage: bench_stereo [width height disparities]

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "r3d/core/image.hpp"
#include "r3d/core/simd.hpp"
#include "r3d/core/thread_pool.hpp"
#include "r3d/core/timer.hpp"
#include "r3d/stereo/census.hpp"
#include "r3d/stereo/sgm_kernels.hpp"
#include "r3d/stereo/stereo_matcher.hpp"

namespace {

r3d::ImageU16 random_image(int w, int h, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 65535);
  r3d::ImageU16 img(w, h, 1);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      img(x, y) = static_cast<std::uint16_t>(dist(rng));
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

}  // namespace

int main(int argc, char** argv) {
  const int w = argc > 3 ? std::atoi(argv[1]) : 1024;
  const int h = argc > 3 ? std::atoi(argv[2]) : 768;
  const int nd = argc > 3 ? std::atoi(argv[3]) : 128;
  const r3d::ImageU16 left = random_image(w, h, 1);
  const r3d::ImageU16 right = random_image(w, h, 2);
  const double pixels = static_cast<double>(w) * h;
  const double cells = pixels * nd;
  std::printf("%dx%d, %d disparities, SIMD %s, %u threads\n", w, h, nd, r3d::simd_active_target_name().c_str(),
              r3d::default_thread_pool().num_threads());

  r3d::StereoOptions options;
  options.tile_size = 4096;  // one tile: pure aggregation throughput
  const unsigned threads = r3d::default_thread_pool().num_threads();
  r3d::set_default_thread_count(1);
  for (const bool scalar : {true, false}) {
    options.scalar_kernels = scalar;
    const char* name = scalar ? "scalar" : "simd  ";
    const double census = best_seconds(3, [&] { (void)r3d::census_transform(left.view(), scalar); });
    const double volume =
        best_seconds(2, [&] { (void)r3d::detail::sgm_cost_volume(left.view(), right.view(), 0, nd, options); });
    std::printf("  1 thread %s  census %7.1f Mpixel/s   cost + 8-path aggregation %7.1f Mcell/s\n", name,
                pixels / census * 1e-6, cells / volume * 1e-6);
  }
  {
    // Cost rows alone (computed twice per pixel by the aggregation).
    const r3d::Image<std::uint64_t> cl = r3d::census_transform(left.view());
    std::vector<std::uint64_t> rev(static_cast<std::size_t>(w + nd));
    std::vector<std::uint16_t> cost(static_cast<std::size_t>(w) * static_cast<std::size_t>(nd));
    for (const bool scalar : {true, false}) {
      const double t = best_seconds(3, [&] {
        for (int y = 0; y < h; ++y) {
          (scalar ? r3d::sgm::scalar::census_costs : r3d::sgm::census_costs)(
              cl.row(y), rev.data(), static_cast<std::size_t>(w), static_cast<std::size_t>(nd), cost.data());
        }
      });
      std::printf("  1 thread %s  cost rows %7.1f Mcell/s\n", scalar ? "scalar" : "simd  ", cells / t * 1e-6);
    }
  }
  {
    // SGM row kernel alone on a short row whose buffers stay in L1/L2.
    const std::size_t rw = 32;
    const auto d = static_cast<std::size_t>(nd);
    const std::size_t stride = r3d::sgm::path_stride(d);
    std::vector<std::uint16_t> cost(rw * d, 7), p2(4 * rw, 200), prev(3 * rw * stride, 0), cur(3 * rw * stride, 0),
        prev_min(3 * rw, 0), cur_min(3 * rw, 0), scratch(3 * stride, 0), sum(rw * d, 0);
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
    args.num_disp = d;
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
  options = r3d::StereoOptions{};
  const double full1 = best_seconds(2, [&] { (void)r3d::match_stereo(left.view(), right.view(), {}, {}, 0, nd, options); });
  std::printf("  1 thread  complete matcher %7.1f Mcell/s (%.3f s)\n", cells / full1 * 1e-6, full1);
  r3d::set_default_thread_count(threads);
  const double full = best_seconds(3, [&] { (void)r3d::match_stereo(left.view(), right.view(), {}, {}, 0, nd, options); });
  std::printf("  %u threads complete matcher %7.1f Mcell/s (%.3f s)\n", threads, cells / full * 1e-6, full);
  return 0;
}
