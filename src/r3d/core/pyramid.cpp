#include "r3d/core/pyramid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "r3d/core/error.hpp"
#include "r3d/core/thread_pool.hpp"

namespace r3d {
namespace {

// Accumulator types: the horizontal pass sums to at most 16 * max(T) and
// the vertical pass to 256 * max(T).
template <typename T>
struct Accum {
  using Row = std::conditional_t<std::is_floating_point_v<T>, float, std::uint32_t>;
};

template <typename T>
T normalize(typename Accum<T>::Row sum) noexcept {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(sum * (1.0F / 256.0F));
  } else {
    return static_cast<T>((sum + 128U) >> 8U);
  }
}

}  // namespace

template <typename T>
Image<T> pyramid_down(const ImageView<const T>& src) {
  if (src.empty()) {
    throw_error("pyramid_down: empty image");
  }
  using A = typename Accum<T>::Row;
  const int w = src.width();
  const int h = src.height();
  const int c = src.channels();
  const int dw = (w + 1) / 2;
  const int dh = (h + 1) / 2;

  // Horizontal filter + decimation of every source row.
  Image<A> tmp = Image<A>::uninitialized(dw, h, c);
  parallel_for(0, static_cast<std::size_t>(h), 16, [&](std::size_t begin, std::size_t end) {
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      const T* in = src.row(y);
      A* out = tmp.row(y);
      for (int x = 0; x < dw; ++x) {
        const int sx = 2 * x;
        const int xm2 = std::max(sx - 2, 0) * c;
        const int xm1 = std::max(sx - 1, 0) * c;
        const int x0 = sx * c;
        const int xp1 = std::min(sx + 1, w - 1) * c;
        const int xp2 = std::min(sx + 2, w - 1) * c;
        for (int k = 0; k < c; ++k) {
          out[x * c + k] = static_cast<A>(in[xm2 + k]) + static_cast<A>(in[xp2 + k]) +
                           A{4} * (static_cast<A>(in[xm1 + k]) + static_cast<A>(in[xp1 + k])) +
                           A{6} * static_cast<A>(in[x0 + k]);
        }
      }
    }
  });

  // Vertical filter + decimation.
  Image<T> dst = Image<T>::uninitialized(dw, dh, c);
  const int row_len = dw * c;
  parallel_for(0, static_cast<std::size_t>(dh), 16, [&](std::size_t begin, std::size_t end) {
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      const int sy = 2 * y;
      const A* rm2 = tmp.row(std::max(sy - 2, 0));
      const A* rm1 = tmp.row(std::max(sy - 1, 0));
      const A* r0 = tmp.row(sy);
      const A* rp1 = tmp.row(std::min(sy + 1, h - 1));
      const A* rp2 = tmp.row(std::min(sy + 2, h - 1));
      T* out = dst.row(y);
      for (int i = 0; i < row_len; ++i) {
        out[i] = normalize<T>(rm2[i] + rp2[i] + A{4} * (rm1[i] + rp1[i]) + A{6} * r0[i]);
      }
    }
  });
  return dst;
}

template <typename T>
std::vector<Image<T>> build_pyramid(const ImageView<const T>& src, int num_levels, int min_size) {
  std::vector<Image<T>> levels;
  if (num_levels < 1 || src.empty()) {
    return levels;
  }
  Image<T> base = Image<T>::uninitialized(src.width(), src.height(), src.channels());
  for (int y = 0; y < src.height(); ++y) {
    std::memcpy(base.row(y), src.row(y), sizeof(T) * static_cast<std::size_t>(src.width()) * src.channels());
  }
  levels.push_back(std::move(base));
  while (static_cast<int>(levels.size()) < num_levels) {
    const Image<T>& prev = levels.back();
    if ((prev.width() + 1) / 2 < min_size || (prev.height() + 1) / 2 < min_size) {
      break;
    }
    levels.push_back(pyramid_down(prev.view()));
  }
  return levels;
}

int pyramid_levels_for(int width, int height, double max_top_pixels) {
  int levels = 1;
  while (static_cast<double>(width) * height > max_top_pixels && width > 1 && height > 1) {
    width = (width + 1) / 2;
    height = (height + 1) / 2;
    ++levels;
  }
  return levels;
}

PinholeIntrinsics pyramid_intrinsics(const PinholeIntrinsics& k, int level) {
  PinholeIntrinsics out = k;
  for (int i = 0; i < level; ++i) {
    out.width = (out.width + 1) / 2;
    out.height = (out.height + 1) / 2;
  }
  const double s = std::ldexp(1.0, -level);
  out.fx *= s;
  out.fy *= s;
  out.cx *= s;
  out.cy *= s;
  return out;
}

template Image<std::uint8_t> pyramid_down(const ImageView<const std::uint8_t>&);
template Image<std::uint16_t> pyramid_down(const ImageView<const std::uint16_t>&);
template Image<float> pyramid_down(const ImageView<const float>&);
template std::vector<Image<std::uint8_t>> build_pyramid(const ImageView<const std::uint8_t>&, int, int);
template std::vector<Image<std::uint16_t>> build_pyramid(const ImageView<const std::uint16_t>&, int, int);
template std::vector<Image<float>> build_pyramid(const ImageView<const float>&, int, int);

}  // namespace r3d
