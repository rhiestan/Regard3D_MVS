#pragma once

#include <string>

#include <Eigen/Core>

#include "r3d/core/camera.hpp"
#include "r3d/core/camera_model.hpp"
#include "r3d/core/image.hpp"
#include "r3d/select/view_selection.hpp"

namespace r3d {

// Calibrated stereo rectification (Fusiello, Trucco & Verri 2000): both
// cameras keep their centres and are rotated to a common orientation whose
// x axis is the baseline, so that epipolar lines become image rows. The
// common viewing direction is the mean of the two optical axes, which keeps
// the rotation of each image small.
//
// Near the epipole the rectified images stretch without bound (forward
// motion). Each rectified image is therefore cropped to `max_growth` times
// the source half-size around the rectified image of the principal point;
// the pair is rejected if that keeps less than `min_coverage` of either
// image (the lost periphery has little parallax anyway).
struct RectificationOptions {
  double resolution = 1.0;      // rectified focal length relative to the left camera
  double max_growth = 1.5;      // crop per side, relative to half the larger source dimension
  double min_coverage = 0.4;    // minimum fraction of each source image inside its rectified image
  // Label ranges larger than this times the wider rectified image are rejected
  // (relative, so that the same pairs are kept at every resolution). Only a
  // guard against degenerate depth ranges: the hierarchical matcher narrows
  // wide ranges per pixel, and even pairs of 79x the width added points on ETH3D.
  double max_disparity_ratio = 100.0;
  int disparity_align = 16;     // the label count is rounded up to a multiple of this
};

// Geometry of a rectified pair. Both cameras have the same rotation, focal
// length (fx = fy), cy and height; cx and width differ because each image is
// cropped to its own useful area. The left camera is always the base view
// and the right camera lies on its +x axis, so the disparity
//   d = x_left - x_right = f * baseline / z + (cx_left - cx_right)
// grows towards the cameras (z: depth along the rectified optical axis).
struct RectifiedPair {
  PinholeCamera left;
  PinholeCamera right;
  double baseline = 0.0;
  int disparity_min = 0;    // labels d = disparity_min .. disparity_min + num_disparities - 1
  int num_disparities = 0;  // multiple of RectificationOptions::disparity_align
  bool valid = false;
  std::string reason;       // why the pair is not valid

  [[nodiscard]] double focal() const noexcept { return left.intrinsics.fx; }
  [[nodiscard]] double disparity_offset() const noexcept { return left.intrinsics.cx - right.intrinsics.cx; }
  [[nodiscard]] double depth_from_disparity(double d) const noexcept {
    return focal() * baseline / (d - disparity_offset());
  }
  [[nodiscard]] double disparity_from_depth(double z) const noexcept {
    return focal() * baseline / z + disparity_offset();
  }
  // World point of left pixel (x, y) with disparity d (requires d > disparity_offset()).
  [[nodiscard]] Eigen::Vector3d triangulate(double x, double y, double d) const {
    return left.unproject(Eigen::Vector2d(x, y), depth_from_disparity(d));
  }
};

// Rectification of the base view (cam_left, pose_left) with the partner view
// (cam_right, pose_right). `depth` is the depth range of the scene in the
// base view; it sets the disparity range and the crop of both images. A
// pair whose rectified images would be unreasonably large (forward motion,
// epipole inside or near the image) is returned with valid = false.
[[nodiscard]] RectifiedPair compute_rectification(const CameraIntrinsics& cam_left, const Pose& pose_left,
                                                  const CameraIntrinsics& cam_right, const Pose& pose_right,
                                                  const DepthRange& depth, const RectificationOptions& options = {});

// Resamples `src`, an image of camera `cam` with pose `pose`, into the
// rectified camera `target` (which must have the same centre). Bicubic
// interpolation, after reducing the source with the pyramid filter when the
// target resolution is much lower. Pixels without a source are 0; if `valid`
// is not null it receives a mask (255 valid, 0 invalid). Rows run in parallel.
template <typename T>
[[nodiscard]] Image<T> rectify_image(const ImageView<const T>& src, const CameraIntrinsics& cam, const Pose& pose,
                                     const PinholeCamera& target, ImageU8* valid = nullptr);

template <typename T>
[[nodiscard]] Image<T> rectify_image(const Image<T>& src, const CameraIntrinsics& cam, const Pose& pose,
                                     const PinholeCamera& target, ImageU8* valid = nullptr) {
  return rectify_image(src.view(), cam, pose, target, valid);
}

// Intrinsics of pyramid level `level` of an image of `cam` (see pyramid.hpp:
// K is scaled by 2^-level; the distortion of normalised coordinates is unchanged).
[[nodiscard]] CameraIntrinsics pyramid_camera(const CameraIntrinsics& cam, int level);

}  // namespace r3d
