#pragma once

#include <array>
#include <optional>
#include <string_view>

#include <Eigen/Core>

#include "r3d/core/camera.hpp"

namespace r3d {

// Camera models of OpenMVG (names as in sfm_data). COLMAP models are mapped
// onto these by the COLMAP importer.
enum class CameraModel {
  pinhole,
  radial_k1,  // k1
  radial_k3,  // k1, k2, k3
  brown_t2,   // k1, k2, k3, t1, t2 (t1, t2 = OpenCV p1, p2)
  fisheye,    // k1..k4, equidistant: theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8)
  spherical,  // equirectangular panorama; read but not usable for dense matching
};

// OpenMVG's registered polymorphic name, e.g. "pinhole_radial_k3".
[[nodiscard]] std::string_view openmvg_camera_model_name(CameraModel model) noexcept;
[[nodiscard]] std::optional<CameraModel> parse_openmvg_camera_model(std::string_view name) noexcept;
// Number of distortion parameters stored in CameraIntrinsics::params.
[[nodiscard]] int distortion_parameter_count(CameraModel model) noexcept;

// Calibrated intrinsics as estimated by SfM, including lens distortion.
// Pixel convention as PinholeIntrinsics: (0, 0) is the centre of the top-left pixel.
//
// Normalised coordinates: u = (x/z, y/z) of the undistorted ray; distorted
// normalised coordinates d = distort(u); pixel = f * d + c.
struct CameraIntrinsics {
  CameraModel model = CameraModel::pinhole;
  int width = 0;
  int height = 0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  std::array<double, 5> params{};  // see CameraModel for the order

  [[nodiscard]] bool is_perspective() const noexcept { return model != CameraModel::spherical; }
  [[nodiscard]] bool has_distortion() const noexcept;

  // Lens distortion in normalised coordinates (OpenMVG add_disto / remove_disto).
  [[nodiscard]] Eigen::Vector2d distort(const Eigen::Vector2d& u) const;
  // Inverse of distort(), by damped Newton iteration; accurate to ~1e-12.
  [[nodiscard]] Eigen::Vector2d undistort(const Eigen::Vector2d& d) const;

  // Distorted pixel <-> undistorted pixel of the pinhole camera with the same K.
  [[nodiscard]] Eigen::Vector2d undistort_pixel(const Eigen::Vector2d& pixel) const;
  [[nodiscard]] Eigen::Vector2d distort_pixel(const Eigen::Vector2d& pixel) const;

  // Camera coordinates -> distorted pixel. Requires x_cam.z() > 0.
  [[nodiscard]] Eigen::Vector2d project(const Eigen::Vector3d& x_cam) const;

  // The distortion-free pinhole camera with the same K and image size.
  [[nodiscard]] PinholeIntrinsics pinhole() const { return {width, height, fx, fy, cx, cy}; }
};

}  // namespace r3d
