#pragma once

#include <Eigen/Core>

namespace r3d {

// Camera pose in OpenMVG convention: rotation R maps world to camera axes and
// C is the camera centre in world coordinates, so X_cam = R * (X_world - C).
struct Pose {
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d center = Eigen::Vector3d::Zero();

  [[nodiscard]] Eigen::Vector3d world_to_camera(const Eigen::Vector3d& x_world) const {
    return rotation * (x_world - center);
  }
  [[nodiscard]] Eigen::Vector3d camera_to_world(const Eigen::Vector3d& x_cam) const {
    return rotation.transpose() * x_cam + center;
  }
  // Translation t of the [R | t] form, t = -R * C.
  [[nodiscard]] Eigen::Vector3d translation() const { return -rotation * center; }
  // Unit viewing direction (camera +z axis) in world coordinates.
  [[nodiscard]] Eigen::Vector3d view_direction() const { return rotation.row(2).transpose(); }
};

// Undistorted pinhole intrinsics. Pixel convention: integer coordinates are
// pixel centres, (0, 0) is the centre of the top-left pixel (same as OpenMVG).
struct PinholeIntrinsics {
  int width = 0;
  int height = 0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;

  [[nodiscard]] Eigen::Matrix3d K() const {
    Eigen::Matrix3d k;
    k << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    return k;
  }

  // Camera coordinates -> pixel. Requires x_cam.z() > 0.
  [[nodiscard]] Eigen::Vector2d project(const Eigen::Vector3d& x_cam) const {
    const double inv_z = 1.0 / x_cam.z();
    return {fx * x_cam.x() * inv_z + cx, fy * x_cam.y() * inv_z + cy};
  }

  // Pixel + depth (distance along the optical axis, not along the ray) -> camera coordinates.
  [[nodiscard]] Eigen::Vector3d unproject(const Eigen::Vector2d& pixel, double depth) const {
    return {(pixel.x() - cx) / fx * depth, (pixel.y() - cy) / fy * depth, depth};
  }

  [[nodiscard]] bool contains(const Eigen::Vector2d& pixel, double border = 0.0) const {
    return pixel.x() >= -0.5 + border && pixel.y() >= -0.5 + border &&
           pixel.x() < width - 0.5 - border && pixel.y() < height - 0.5 - border;
  }
};

// A calibrated, undistorted view: intrinsics plus pose.
struct PinholeCamera {
  PinholeIntrinsics intrinsics;
  Pose pose;

  [[nodiscard]] Eigen::Vector2d project(const Eigen::Vector3d& x_world) const {
    return intrinsics.project(pose.world_to_camera(x_world));
  }
  [[nodiscard]] double depth(const Eigen::Vector3d& x_world) const {
    return pose.world_to_camera(x_world).z();
  }
  [[nodiscard]] Eigen::Vector3d unproject(const Eigen::Vector2d& pixel, double depth) const {
    return pose.camera_to_world(intrinsics.unproject(pixel, depth));
  }
  // 3x4 projection matrix P = K [R | t].
  [[nodiscard]] Eigen::Matrix<double, 3, 4> projection_matrix() const {
    Eigen::Matrix<double, 3, 4> rt;
    rt.leftCols<3>() = pose.rotation;
    rt.col(3) = pose.translation();
    return intrinsics.K() * rt;
  }
};

}  // namespace r3d
