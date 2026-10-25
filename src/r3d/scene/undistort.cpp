#include "r3d/scene/undistort.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <Eigen/LU>

#include "r3d/core/error.hpp"
#include "r3d/core/log.hpp"
#include "r3d/core/sampling.hpp"
#include "r3d/core/thread_pool.hpp"

namespace r3d {
namespace {

constexpr int kBorderSamples = 256;  // per image side

// Undistorted pixel (same K) of a distorted pixel, or false if the
// distortion cannot be inverted there.
bool undistort_checked(const CameraIntrinsics& cam, const Eigen::Vector2d& p, Eigen::Vector2d& u) {
  u = cam.undistort_pixel(p);
  return u.allFinite() && (cam.distort_pixel(u) - p).norm() < 1e-6;
}

template <typename T>
T to_pixel(float v) noexcept {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(v);
  } else {
    const float clamped = std::clamp(std::round(v), 0.0F, static_cast<float>(std::numeric_limits<T>::max()));
    return static_cast<T>(clamped);
  }
}

}  // namespace

PinholeIntrinsics undistorted_camera(const CameraIntrinsics& cam, const UndistortOptions& options) {
  PinholeIntrinsics out = cam.pinhole();
  if (options.fit == UndistortFit::same_size || !cam.has_distortion()) {
    return out;
  }

  // Undistorted positions of the source image border (pixel area boundary).
  const double x0 = -0.5;
  const double y0 = -0.5;
  const double x1 = cam.width - 0.5;
  const double y1 = cam.height - 0.5;
  constexpr double kInf = std::numeric_limits<double>::infinity();
  double all_xmin = kInf, all_xmax = -kInf, all_ymin = kInf, all_ymax = -kInf;
  double left = -kInf, right = kInf, top = -kInf, bottom = kInf;
  int failed = 0;
  for (int i = 0; i <= kBorderSamples; ++i) {
    const double t = static_cast<double>(i) / kBorderSamples;
    const double bx = x0 + t * (x1 - x0);
    const double by = y0 + t * (y1 - y0);
    const Eigen::Vector2d edge_points[4] = {{x0, by}, {x1, by}, {bx, y0}, {bx, y1}};
    for (int side = 0; side < 4; ++side) {
      Eigen::Vector2d u;
      if (!undistort_checked(cam, edge_points[side], u)) {
        ++failed;
        continue;
      }
      all_xmin = std::min(all_xmin, u.x());
      all_xmax = std::max(all_xmax, u.x());
      all_ymin = std::min(all_ymin, u.y());
      all_ymax = std::max(all_ymax, u.y());
      switch (side) {
        case 0:
          left = std::max(left, u.x());
          break;
        case 1:
          right = std::min(right, u.x());
          break;
        case 2:
          top = std::max(top, u.y());
          break;
        default:
          bottom = std::min(bottom, u.y());
          break;
      }
    }
  }
  if (failed > 0) {
    log_warn("undistortion: distortion is not invertible at {} border samples, keeping the original size", failed);
    return out;
  }

  double xmin = all_xmin, xmax = all_xmax, ymin = all_ymin, ymax = all_ymax;
  if (options.fit == UndistortFit::inner) {
    xmin = left;
    xmax = right;
    ymin = top;
    ymax = bottom;
  } else {
    const double g = std::max(options.max_growth, 1.0);
    xmin = std::max(xmin, cam.cx - g * (cam.cx - x0));
    xmax = std::min(xmax, cam.cx + g * (x1 - cam.cx));
    ymin = std::max(ymin, cam.cy - g * (cam.cy - y0));
    ymax = std::min(ymax, cam.cy + g * (y1 - cam.cy));
  }
  if (!(xmax > xmin) || !(ymax > ymin)) {
    throw_error("undistortion: degenerate image area [{}, {}] x [{}, {}]", xmin, xmax, ymin, ymax);
  }
  // keep_all rounds the size up (contains every source pixel), inner rounds
  // down (contains only valid pixels). The area starts at pixel boundary -0.5.
  constexpr double kEps = 1e-6;
  const double w = options.fit == UndistortFit::inner ? std::floor(xmax - xmin + kEps) : std::ceil(xmax - xmin - kEps);
  const double h = options.fit == UndistortFit::inner ? std::floor(ymax - ymin + kEps) : std::ceil(ymax - ymin - kEps);
  out.width = std::max(1, static_cast<int>(w));
  out.height = std::max(1, static_cast<int>(h));
  out.cx = cam.cx - xmin - 0.5;
  out.cy = cam.cy - ymin - 0.5;
  return out;
}

bool distortion_is_monotonic(const CameraIntrinsics& cam) {
  if (!cam.has_distortion()) {
    return true;
  }
  constexpr int kGrid = 32;
  for (int j = 0; j <= kGrid; ++j) {
    for (int i = 0; i <= kGrid; ++i) {
      const Eigen::Vector2d p(-0.5 + cam.width * static_cast<double>(i) / kGrid,
                              -0.5 + cam.height * static_cast<double>(j) / kGrid);
      Eigen::Vector2d u;
      if (!undistort_checked(cam, p, u)) {
        return false;
      }
      // Jacobian of distort_pixel at u by central differences.
      constexpr double h = 1e-3;
      Eigen::Matrix2d jac;
      jac.col(0) = (cam.distort_pixel(u + Eigen::Vector2d(h, 0)) - cam.distort_pixel(u - Eigen::Vector2d(h, 0))) / (2 * h);
      jac.col(1) = (cam.distort_pixel(u + Eigen::Vector2d(0, h)) - cam.distort_pixel(u - Eigen::Vector2d(0, h))) / (2 * h);
      if (!(jac.determinant() > 0.0)) {
        return false;
      }
    }
  }
  return true;
}

template <typename T>
Image<T> undistort_image(const ImageView<const T>& src, const CameraIntrinsics& cam, const PinholeIntrinsics& target,
                         ImageU8* valid) {
  if (src.width() != cam.width || src.height() != cam.height) {
    throw_error("undistort_image: image is {}x{}, camera is {}x{}", src.width(), src.height(), cam.width, cam.height);
  }
  const int channels = src.channels();
  Image<T> out = Image<T>::uninitialized(target.width, target.height, channels);
  if (valid != nullptr) {
    *valid = ImageU8::uninitialized(target.width, target.height, 1);
  }
  parallel_for(0, static_cast<std::size_t>(target.height), 8, [&](std::size_t begin, std::size_t end) {
    std::vector<float> value(static_cast<std::size_t>(channels));
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      T* row = out.row(y);
      std::uint8_t* mask = valid != nullptr ? valid->row(y) : nullptr;
      const double uy = (y - target.cy) / target.fy;
      for (int x = 0; x < target.width; ++x) {
        const Eigen::Vector2d d = cam.distort(Eigen::Vector2d((x - target.cx) / target.fx, uy));
        const double sx = cam.fx * d.x() + cam.cx;
        const double sy = cam.fy * d.y() + cam.cy;
        T* px = row + static_cast<std::ptrdiff_t>(x) * channels;
        const bool inside = inside_image(src.width(), src.height(), sx, sy);
        if (inside) {
          sample_bicubic(src, sx, sy, value.data());
          for (int c = 0; c < channels; ++c) {
            px[c] = to_pixel<T>(value[static_cast<std::size_t>(c)]);
          }
        } else {
          std::fill(px, px + channels, T{});
        }
        if (mask != nullptr) {
          mask[x] = inside ? 255 : 0;
        }
      }
    }
  });
  return out;
}

template Image<std::uint8_t> undistort_image(const ImageView<const std::uint8_t>&, const CameraIntrinsics&,
                                             const PinholeIntrinsics&, ImageU8*);
template Image<std::uint16_t> undistort_image(const ImageView<const std::uint16_t>&, const CameraIntrinsics&,
                                              const PinholeIntrinsics&, ImageU8*);
template Image<float> undistort_image(const ImageView<const float>&, const CameraIntrinsics&,
                                      const PinholeIntrinsics&, ImageU8*);

}  // namespace r3d
