#include "r3d/core/camera_model.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/LU>

#include "r3d/core/error.hpp"

namespace r3d {
namespace {

struct ModelInfo {
  CameraModel model;
  std::string_view name;
  int num_params;
};

constexpr std::array<ModelInfo, 6> kModels{{
    {CameraModel::pinhole, "pinhole", 0},
    {CameraModel::radial_k1, "pinhole_radial_k1", 1},
    {CameraModel::radial_k3, "pinhole_radial_k3", 3},
    {CameraModel::brown_t2, "pinhole_brown_t2", 5},
    {CameraModel::fisheye, "fisheye", 4},
    {CameraModel::spherical, "spherical", 0},
}};

const ModelInfo& info(CameraModel model) {
  for (const ModelInfo& m : kModels) {
    if (m.model == model) {
      return m;
    }
  }
  return kModels[0];
}

// Jacobian of distort() by central differences. Only used for Newton steps;
// the accuracy of undistort() is set by the residual, not by the Jacobian.
Eigen::Matrix2d distortion_jacobian(const CameraIntrinsics& cam, const Eigen::Vector2d& u) {
  Eigen::Matrix2d jac;
  for (int k = 0; k < 2; ++k) {
    const double h = 1e-7 * std::max(1.0, std::abs(u[k]));
    Eigen::Vector2d up = u;
    Eigen::Vector2d um = u;
    up[k] += h;
    um[k] -= h;
    jac.col(k) = (cam.distort(up) - cam.distort(um)) / (2.0 * h);
  }
  return jac;
}

}  // namespace

std::string_view openmvg_camera_model_name(CameraModel model) noexcept { return info(model).name; }

std::optional<CameraModel> parse_openmvg_camera_model(std::string_view name) noexcept {
  for (const ModelInfo& m : kModels) {
    if (m.name == name) {
      return m.model;
    }
  }
  return std::nullopt;
}

int distortion_parameter_count(CameraModel model) noexcept { return info(model).num_params; }

bool CameraIntrinsics::has_distortion() const noexcept {
  const int n = distortion_parameter_count(model);
  for (int i = 0; i < n; ++i) {
    if (params[static_cast<std::size_t>(i)] != 0.0) {
      return true;
    }
  }
  return false;
}

// Formulas as in OpenMVG (cameras/Camera_Pinhole_*.hpp, add_disto).
Eigen::Vector2d CameraIntrinsics::distort(const Eigen::Vector2d& u) const {
  const double r2 = u.squaredNorm();
  switch (model) {
    case CameraModel::pinhole:
    case CameraModel::spherical:
      return u;
    case CameraModel::radial_k1:
      return u * (1.0 + params[0] * r2);
    case CameraModel::radial_k3:
      return u * (1.0 + r2 * (params[0] + r2 * (params[1] + r2 * params[2])));
    case CameraModel::brown_t2: {
      const double k1 = params[0];
      const double k2 = params[1];
      const double k3 = params[2];
      const double t1 = params[3];
      const double t2 = params[4];
      const double radial = r2 * (k1 + r2 * (k2 + r2 * k3));
      const double tx = t2 * (r2 + 2.0 * u.x() * u.x()) + 2.0 * t1 * u.x() * u.y();
      const double ty = t1 * (r2 + 2.0 * u.y() * u.y()) + 2.0 * t2 * u.x() * u.y();
      return {u.x() + u.x() * radial + tx, u.y() + u.y() * radial + ty};
    }
    case CameraModel::fisheye: {
      const double r = std::sqrt(r2);
      if (r <= 1e-8) {
        return u;
      }
      const double theta = std::atan(r);
      const double t2 = theta * theta;
      const double theta_d = theta * (1.0 + t2 * (params[0] + t2 * (params[1] + t2 * (params[2] + t2 * params[3]))));
      return u * (theta_d / r);
    }
  }
  return u;
}

Eigen::Vector2d CameraIntrinsics::undistort(const Eigen::Vector2d& d) const {
  if (!has_distortion()) {
    return d;
  }
  Eigen::Vector2d u = d;
  Eigen::Vector2d residual = distort(u) - d;
  double error = residual.squaredNorm();
  constexpr double kTolerance = 1e-26;  // |residual| < 1e-13 in normalised units
  for (int iter = 0; iter < 50 && error > kTolerance; ++iter) {
    const Eigen::Vector2d step = distortion_jacobian(*this, u).partialPivLu().solve(residual);
    // Damping: halve the step until the residual decreases (guards against
    // overshooting where the distortion polynomial is strongly curved).
    double scale = 1.0;
    bool improved = false;
    for (int k = 0; k < 30; ++k) {
      const Eigen::Vector2d candidate = u - scale * step;
      const Eigen::Vector2d candidate_residual = distort(candidate) - d;
      const double candidate_error = candidate_residual.squaredNorm();
      if (std::isfinite(candidate_error) && candidate_error < error) {
        u = candidate;
        residual = candidate_residual;
        error = candidate_error;
        improved = true;
        break;
      }
      scale *= 0.5;
    }
    if (!improved) {
      break;  // converged to machine precision, or no inverse exists here
    }
  }
  return u;
}

Eigen::Vector2d CameraIntrinsics::undistort_pixel(const Eigen::Vector2d& pixel) const {
  const Eigen::Vector2d d((pixel.x() - cx) / fx, (pixel.y() - cy) / fy);
  const Eigen::Vector2d u = undistort(d);
  return {fx * u.x() + cx, fy * u.y() + cy};
}

Eigen::Vector2d CameraIntrinsics::distort_pixel(const Eigen::Vector2d& pixel) const {
  const Eigen::Vector2d u((pixel.x() - cx) / fx, (pixel.y() - cy) / fy);
  const Eigen::Vector2d d = distort(u);
  return {fx * d.x() + cx, fy * d.y() + cy};
}

Eigen::Vector2d CameraIntrinsics::project(const Eigen::Vector3d& x_cam) const {
  if (!is_perspective()) {
    throw_error("project: camera model '{}' is not perspective", openmvg_camera_model_name(model));
  }
  const Eigen::Vector2d d = distort(x_cam.head<2>() / x_cam.z());
  return {fx * d.x() + cx, fy * d.y() + cy};
}

}  // namespace r3d
