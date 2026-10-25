#include <cmath>

#include <Eigen/Geometry>
#include <doctest/doctest.h>

#include "r3d/core/camera.hpp"
#include "test_util.hpp"

namespace {

r3d::PinholeCamera make_camera() {
  r3d::PinholeCamera cam;
  cam.intrinsics = {.width = 640, .height = 480, .fx = 500.0, .fy = 510.0, .cx = 319.5, .cy = 239.5};
  cam.pose.rotation =
      Eigen::AngleAxisd(0.3, Eigen::Vector3d(1.0, 2.0, -0.5).normalized()).toRotationMatrix();
  cam.pose.center = Eigen::Vector3d(1.0, -2.0, 0.5);
  return cam;
}

}  // namespace

TEST_CASE("Pose transforms are inverse to each other") {
  const r3d::PinholeCamera cam = make_camera();
  const Eigen::Vector3d x(3.0, 1.0, 7.0);
  const Eigen::Vector3d back = cam.pose.camera_to_world(cam.pose.world_to_camera(x));
  CHECK((back - x).norm() < 1e-12);
  CHECK(cam.pose.world_to_camera(cam.pose.center).norm() < 1e-12);
  CHECK((cam.pose.rotation * cam.pose.view_direction() - Eigen::Vector3d::UnitZ()).norm() < 1e-12);
}

TEST_CASE("Projection and unprojection round trip") {
  const r3d::PinholeCamera cam = make_camera();
  const Eigen::Vector2d pixel(100.25, 400.75);
  const double depth = 12.5;
  const Eigen::Vector3d x = cam.unproject(pixel, depth);
  R3D_CHECK_NEAR(cam.depth(x), depth, 1e-9);
  CHECK((cam.project(x) - pixel).norm() < 1e-9);

  // Projection matrix agrees with the explicit pipeline.
  const Eigen::Vector3d h = cam.projection_matrix() * x.homogeneous();
  CHECK((h.hnormalized() - pixel).norm() < 1e-9);
}

TEST_CASE("Principal point projects from the optical axis") {
  const r3d::PinholeCamera cam = make_camera();
  const Eigen::Vector3d on_axis = cam.pose.center + 5.0 * cam.pose.view_direction();
  const Eigen::Vector2d p = cam.project(on_axis);
  R3D_CHECK_NEAR(p.x(), 319.5, 1e-9);
  R3D_CHECK_NEAR(p.y(), 239.5, 1e-9);
  CHECK(cam.intrinsics.contains(p));
  CHECK_FALSE(cam.intrinsics.contains(Eigen::Vector2d(-0.6, 10.0)));
  CHECK(cam.intrinsics.contains(Eigen::Vector2d(639.4, 479.4)));
}
