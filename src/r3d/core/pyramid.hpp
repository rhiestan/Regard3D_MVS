#pragma once

#include <vector>

#include "r3d/core/camera.hpp"
#include "r3d/core/image.hpp"

namespace r3d {

// Gaussian pyramid by 2x decimation after a separable binomial filter
// [1 4 6 4 1] / 16 (border pixels replicated).
//
// Geometry: pixel i of level k+1 is centred on pixel 2i of level k, so with
// the pixel-centre convention a level-k coordinate is x_k = x_0 / 2^k and the
// level-k camera is K scaled by 2^-k (see pyramid_intrinsics). Level sizes are
// ceil(w / 2^k) x ceil(h / 2^k).

// One 2x reduction. 8/16-bit results are rounded exactly (integer arithmetic).
template <typename T>
[[nodiscard]] Image<T> pyramid_down(const ImageView<const T>& src);

template <typename T>
[[nodiscard]] Image<T> pyramid_down(const Image<T>& src) {
  return pyramid_down(src.view());
}

// Levels 0 (a copy of src) to num_levels - 1. Stops early when a level would
// be smaller than min_size pixels in either direction.
template <typename T>
[[nodiscard]] std::vector<Image<T>> build_pyramid(const ImageView<const T>& src, int num_levels, int min_size = 16);

template <typename T>
[[nodiscard]] std::vector<Image<T>> build_pyramid(const Image<T>& src, int num_levels, int min_size = 16) {
  return build_pyramid(src.view(), num_levels, min_size);
}

// Number of levels such that the top level has at most max_top_pixels pixels.
[[nodiscard]] int pyramid_levels_for(int width, int height, double max_top_pixels);

// Pinhole camera of pyramid level `level` for the level-0 camera `k`.
[[nodiscard]] PinholeIntrinsics pyramid_intrinsics(const PinholeIntrinsics& k, int level);

}  // namespace r3d
