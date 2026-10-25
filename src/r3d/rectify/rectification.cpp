#include "r3d/rectify/rectification.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <type_traits>
#include <vector>

#include <Eigen/Geometry>

#include "r3d/core/error.hpp"
#include "r3d/core/pyramid.hpp"
#include "r3d/core/sampling.hpp"
#include "r3d/core/thread_pool.hpp"

namespace r3d {
namespace {

constexpr int kBorderSamples = 64;  // per image side
constexpr double kInf = std::numeric_limits<double>::infinity();

struct Box {
  double umin = kInf, umax = -kInf, vmin = kInf, vmax = -kInf;
  void extend(double u, double v) {
    umin = std::min(umin, u);
    umax = std::max(umax, u);
    vmin = std::min(vmin, v);
    vmax = std::max(vmax, v);
  }
  [[nodiscard]] bool empty() const { return !(umax > umin) || !(vmax > vmin); }
};

// Ray (camera coordinates, z = 1) of a distorted pixel, or false if the
// distortion cannot be inverted there.
bool pixel_ray(const CameraIntrinsics& cam, const Eigen::Vector2d& p, Eigen::Vector3d& ray) {
  const Eigen::Vector2d u = cam.undistort_pixel(p);
  if (!u.allFinite() || (cam.distort_pixel(u) - p).norm() > 1e-6) {
    return false;
  }
  ray = Eigen::Vector3d((u.x() - cam.cx) / cam.fx, (u.y() - cam.cy) / cam.fy, 1.0);
  return true;
}

// Rectified image-plane position (focal f, principal point 0) of a source
// camera ray; false if the ray does not point into the rectified half space.
bool rectified_position(const Eigen::Matrix3d& m, const Eigen::Vector3d& ray, double f, Eigen::Vector2d& uv) {
  const Eigen::Vector3d r = m * ray;
  if (!(r.z() > 1e-9 * r.norm())) {
    return false;
  }
  uv = Eigen::Vector2d(f * r.x() / r.z(), f * r.y() / r.z());
  return true;
}

// Extent of the image border of (cam, pose) in the rectified image plane of
// the rotation `rect`, clipped to `window` (the region around the rectified
// principal point that the rectified image may cover at most).
bool rectified_extent(const CameraIntrinsics& cam, const Pose& pose, const Eigen::Matrix3d& rect, double f,
                      const Box& window, Box& box, std::string& reason) {
  const Eigen::Matrix3d m = rect * pose.rotation.transpose();  // source camera -> rectified camera
  const double x0 = -0.5;
  const double y0 = -0.5;
  const double x1 = cam.width - 0.5;
  const double y1 = cam.height - 0.5;
  int failed = 0;
  for (int i = 0; i <= kBorderSamples; ++i) {
    const double t = static_cast<double>(i) / kBorderSamples;
    const double bx = x0 + t * (x1 - x0);
    const double by = y0 + t * (y1 - y0);
    for (const Eigen::Vector2d& p : {Eigen::Vector2d(x0, by), Eigen::Vector2d(x1, by), Eigen::Vector2d(bx, y0),
                                     Eigen::Vector2d(bx, y1)}) {
      Eigen::Vector3d ray;
      if (!pixel_ray(cam, p, ray)) {
        ++failed;
        continue;
      }
      Eigen::Vector2d uv;
      if (!rectified_position(m, ray, f, uv)) {
        // Beyond the rectified image plane: the image extends to infinity in
        // the direction of the ray.
        const Eigen::Vector3d r = m * ray;
        uv = Eigen::Vector2d(r.x() > 0 ? kInf : -kInf, r.y() > 0 ? kInf : -kInf);
      }
      box.extend(std::clamp(uv.x(), window.umin, window.umax), std::clamp(uv.y(), window.vmin, window.vmax));
    }
  }
  if (failed > kBorderSamples) {
    reason = "lens distortion not invertible along the image border";
    return false;
  }
  if (box.empty()) {
    reason = "image outside the rectified view";
    return false;
  }
  return true;
}

// Crop window of a source camera: max_growth times half its larger dimension
// around the rectified image of its principal point.
bool crop_window(const CameraIntrinsics& cam, const Pose& pose, const Eigen::Matrix3d& rect, double f, double growth,
                 Box& window) {
  Eigen::Vector2d c;
  if (!rectified_position(rect * pose.rotation.transpose(), Eigen::Vector3d::UnitZ(), f, c)) {
    return false;
  }
  const double half = growth * 0.5 * std::max(cam.width, cam.height) * f / cam.fx;
  window = Box{c.x() - half, c.x() + half, c.y() - half, c.y() + half};
  return true;
}

// Fraction of the source image that lands inside the rectified area `area`.
double coverage(const CameraIntrinsics& cam, const Pose& pose, const Eigen::Matrix3d& rect, double f, const Box& area) {
  const Eigen::Matrix3d m = rect * pose.rotation.transpose();
  constexpr int kGrid = 24;
  int inside = 0;
  for (int j = 0; j < kGrid; ++j) {
    for (int i = 0; i < kGrid; ++i) {
      const Eigen::Vector2d p((i + 0.5) * cam.width / kGrid - 0.5, (j + 0.5) * cam.height / kGrid - 0.5);
      Eigen::Vector3d ray;
      Eigen::Vector2d uv;
      if (pixel_ray(cam, p, ray) && rectified_position(m, ray, f, uv) && uv.x() >= area.umin &&
          uv.x() <= area.umax && uv.y() >= area.vmin && uv.y() <= area.vmax) {
        ++inside;
      }
    }
  }
  return static_cast<double>(inside) / (kGrid * kGrid);
}

// Pixel grid covering [lo, hi] of the image plane: size and principal point
// (pixel centres at lo + 0.5, lo + 1.5, ...).
void pixel_grid(double lo, double hi, int& size, double& c) {
  constexpr double kEps = 1e-6;
  size = std::max(1, static_cast<int>(std::ceil(hi - lo - kEps)));
  c = -lo - 0.5;
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

CameraIntrinsics pyramid_camera(const CameraIntrinsics& cam, int level) {
  CameraIntrinsics out = cam;
  const double s = std::ldexp(1.0, -level);
  out.fx *= s;
  out.fy *= s;
  out.cx *= s;
  out.cy *= s;
  for (int k = 0; k < level; ++k) {
    out.width = (out.width + 1) / 2;
    out.height = (out.height + 1) / 2;
  }
  return out;
}

RectifiedPair compute_rectification(const CameraIntrinsics& cam_left, const Pose& pose_left,
                                    const CameraIntrinsics& cam_right, const Pose& pose_right, const DepthRange& depth,
                                    const RectificationOptions& options) {
  RectifiedPair pair;
  if (!cam_left.is_perspective() || !cam_right.is_perspective()) {
    pair.reason = "not a perspective camera";
    return pair;
  }
  if (!depth.valid()) {
    pair.reason = "invalid depth range";
    return pair;
  }
  const Eigen::Vector3d baseline = pose_right.center - pose_left.center;
  pair.baseline = baseline.norm();
  if (!(pair.baseline > 0.0)) {
    pair.reason = "zero baseline";
    return pair;
  }

  // Rectified axes: x along the baseline, z close to the mean viewing direction.
  const Eigen::Vector3d x_axis = baseline / pair.baseline;
  const Eigen::Vector3d mean_view = pose_left.view_direction() + pose_right.view_direction();
  Eigen::Vector3d y_axis = mean_view.cross(x_axis);
  if (y_axis.norm() < 1e-6 * mean_view.norm() || mean_view.norm() < 1e-6) {
    pair.reason = "baseline along the viewing direction";
    return pair;
  }
  y_axis.normalize();
  const Eigen::Vector3d z_axis = x_axis.cross(y_axis);
  Eigen::Matrix3d rect;
  rect.row(0) = x_axis.transpose();
  rect.row(1) = y_axis.transpose();
  rect.row(2) = z_axis.transpose();
  const double f = options.resolution * cam_left.fx;

  Box window_left, window_right;
  if (!crop_window(cam_left, pose_left, rect, f, options.max_growth, window_left) ||
      !crop_window(cam_right, pose_right, rect, f, options.max_growth, window_right)) {
    pair.reason = "optical axis more than 90 degrees off the rectified direction";
    return pair;
  }
  Box box_left, box_right;
  if (!rectified_extent(cam_left, pose_left, rect, f, window_left, box_left, pair.reason) ||
      !rectified_extent(cam_right, pose_right, rect, f, window_right, box_right, pair.reason)) {
    return pair;
  }

  // Image-plane disparity f * b / z over the depth range, sampled on rays
  // across the left image (z of the rectified camera differs from the depth
  // of the base camera).
  double dmin = kInf;
  double dmax = -kInf;
  constexpr int kGrid = 8;
  for (int j = 0; j <= kGrid; ++j) {
    for (int i = 0; i <= kGrid; ++i) {
      const Eigen::Vector2d p(-0.5 + cam_left.width * static_cast<double>(i) / kGrid,
                              -0.5 + cam_left.height * static_cast<double>(j) / kGrid);
      Eigen::Vector3d ray;
      if (!pixel_ray(cam_left, p, ray)) {
        continue;
      }
      for (const double z : {depth.min, depth.max}) {
        const Eigen::Vector3d x_world = pose_left.camera_to_world(ray * z);
        const Eigen::Vector3d xl = rect * (x_world - pose_left.center);
        const Eigen::Vector3d xr = rect * (x_world - pose_right.center);
        if (xl.z() <= 0.0 || xr.z() <= 0.0) {
          continue;
        }
        const double d = f * (xl.x() / xl.z() - xr.x() / xr.z());
        dmin = std::min(dmin, d);
        dmax = std::max(dmax, d);
      }
    }
  }
  if (!(dmax >= dmin)) {
    pair.reason = "depth range behind the rectified cameras";
    return pair;
  }

  // Crop: rows seen by both images; left columns that can have a match;
  // right columns that can be the match of a left column.
  const double vmin = std::max(box_left.vmin, box_right.vmin);
  const double vmax = std::min(box_left.vmax, box_right.vmax);
  const double ul_min = std::max(box_left.umin, box_right.umin + dmin);
  const double ul_max = std::min(box_left.umax, box_right.umax + dmax);
  const double ur_min = std::max(box_right.umin, ul_min - dmax);
  const double ur_max = std::min(box_right.umax, ul_max - dmin);
  if (!(vmax > vmin) || !(ul_max > ul_min) || !(ur_max > ur_min)) {
    pair.reason = "rectified images do not overlap";
    return pair;
  }

  PinholeIntrinsics left{0, 0, f, f, 0.0, 0.0};
  PinholeIntrinsics right = left;
  pixel_grid(vmin, vmax, left.height, left.cy);
  right.height = left.height;
  right.cy = left.cy;
  pixel_grid(ul_min, ul_max, left.width, left.cx);
  pixel_grid(ur_min, ur_max, right.width, right.cx);

  const double coverage_left = coverage(cam_left, pose_left, rect, f, Box{ul_min, ul_max, vmin, vmax});
  const double coverage_right = coverage(cam_right, pose_right, rect, f, Box{ur_min, ur_max, vmin, vmax});
  if (coverage_left < options.min_coverage || coverage_right < options.min_coverage) {
    pair.reason = std::format("rectified images cover only {:.0f}% / {:.0f}% of the views (epipole near the image)",
                              100 * coverage_left, 100 * coverage_right);
    return pair;
  }

  pair.left = PinholeCamera{left, Pose{rect, pose_left.center}};
  pair.right = PinholeCamera{right, Pose{rect, pose_right.center}};
  const double offset = left.cx - right.cx;
  pair.disparity_min = static_cast<int>(std::floor(dmin + offset));
  const int dmax_index = static_cast<int>(std::ceil(dmax + offset));
  const int align = std::max(1, options.disparity_align);
  pair.num_disparities = (dmax_index - pair.disparity_min + 1 + align - 1) / align * align;
  if (pair.num_disparities > options.max_disparities) {
    pair.reason = std::format("disparity range too large ({} labels)", pair.num_disparities);
    return pair;
  }
  pair.valid = true;
  return pair;
}

template <typename T>
Image<T> rectify_image(const ImageView<const T>& src, const CameraIntrinsics& cam, const Pose& pose,
                       const PinholeCamera& target, ImageU8* valid) {
  if (src.width() != cam.width || src.height() != cam.height) {
    throw_error("rectify_image: image is {}x{}, camera is {}x{}", src.width(), src.height(), cam.width, cam.height);
  }
  if ((pose.center - target.pose.center).norm() > 1e-9 * (1.0 + pose.center.norm())) {
    throw_error("rectify_image: the target camera must have the same centre as the source");
  }
  const PinholeIntrinsics& k = target.intrinsics;

  // Anti-aliasing: sample a pyramid level when the target has less than
  // 1/1.5 of the source resolution.
  int level = 0;
  while (cam.fx / k.fx > 1.5 * std::ldexp(1.0, level) && (cam.width >> (level + 1)) >= 16 &&
         (cam.height >> (level + 1)) >= 16) {
    ++level;
  }
  std::vector<Image<T>> pyramid;
  ImageView<const T> source = src;
  CameraIntrinsics source_cam = cam;
  if (level > 0) {
    pyramid = build_pyramid(src, level + 1, 1);
    source = pyramid[static_cast<std::size_t>(level)].view();
    source_cam = pyramid_camera(cam, level);
  }

  const Eigen::Matrix3d m = pose.rotation * target.pose.rotation.transpose();  // target -> source camera
  const int channels = src.channels();
  Image<T> out = Image<T>::uninitialized(k.width, k.height, channels);
  if (valid != nullptr) {
    *valid = ImageU8::uninitialized(k.width, k.height, 1);
  }
  parallel_for(0, static_cast<std::size_t>(k.height), 8, [&](std::size_t begin, std::size_t end) {
    std::vector<float> value(static_cast<std::size_t>(channels));
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      T* row = out.row(y);
      std::uint8_t* mask = valid != nullptr ? valid->row(y) : nullptr;
      const double ry = (y - k.cy) / k.fy;
      for (int x = 0; x < k.width; ++x) {
        const Eigen::Vector3d r = m * Eigen::Vector3d((x - k.cx) / k.fx, ry, 1.0);
        bool inside = r.z() > 1e-9;
        Eigen::Vector2d s = Eigen::Vector2d::Zero();
        if (inside) {
          s = source_cam.project(r);
          inside = inside_image(source.width(), source.height(), s.x(), s.y());
        }
        T* px = row + static_cast<std::ptrdiff_t>(x) * channels;
        if (inside) {
          sample_bicubic(source, s.x(), s.y(), value.data());
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

template Image<std::uint8_t> rectify_image(const ImageView<const std::uint8_t>&, const CameraIntrinsics&, const Pose&,
                                           const PinholeCamera&, ImageU8*);
template Image<std::uint16_t> rectify_image(const ImageView<const std::uint16_t>&, const CameraIntrinsics&,
                                            const Pose&, const PinholeCamera&, ImageU8*);
template Image<float> rectify_image(const ImageView<const float>&, const CameraIntrinsics&, const Pose&,
                                    const PinholeCamera&, ImageU8*);

}  // namespace r3d
