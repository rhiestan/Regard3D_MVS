#include <algorithm>
#include <cmath>
#include <random>

#include <Eigen/Geometry>
#include <doctest/doctest.h>

#include "r3d/core/camera_model.hpp"
#include "r3d/core/image.hpp"
#include "r3d/rectify/rectification.hpp"
#include "r3d/stereo/pair_cloud.hpp"
#include "test_util.hpp"

using r3d::CameraIntrinsics;
using r3d::CameraModel;
using r3d::Pose;

namespace {

Eigen::Matrix3d rotation(double rx_deg, double ry_deg, double rz_deg) {
  constexpr double kDeg = 3.14159265358979323846 / 180.0;
  return (Eigen::AngleAxisd(rz_deg * kDeg, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(ry_deg * kDeg, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(rx_deg * kDeg, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

CameraIntrinsics camera(CameraModel model) {
  CameraIntrinsics cam;
  cam.model = model;
  cam.width = 320;
  cam.height = 240;
  cam.fx = cam.fy = 280.0;
  cam.cx = 161.2;
  cam.cy = 118.7;
  if (model == CameraModel::radial_k3) {
    cam.params = {-0.2, 0.06, -0.008};
  } else if (model == CameraModel::brown_t2) {
    cam.params = {0.1, -0.04, 0.0, 0.001, -0.002};
  }
  return cam;
}

struct StereoRig {
  CameraIntrinsics left_cam;
  CameraIntrinsics right_cam;
  Pose left_pose;
  Pose right_pose;
};

// Base camera looking roughly along +z, partner 0.4 to its right (or left),
// slightly rotated and displaced.
StereoRig rig(CameraModel model, double side = 1.0) {
  StereoRig r;
  r.left_cam = camera(model);
  r.right_cam = camera(model);
  r.right_cam.fx = r.right_cam.fy = 300.0;
  r.left_pose.rotation = rotation(2.0, -3.0, 1.0);
  r.left_pose.center = Eigen::Vector3d(0.1, -0.2, 0.0);
  r.right_pose.rotation = rotation(-1.0, 4.0 * side, -2.0);
  r.right_pose.center = r.left_pose.center + r.left_pose.rotation.transpose() * Eigen::Vector3d(0.4 * side, 0.03, 0.05);
  return r;
}

}  // namespace

TEST_CASE("Rectified cameras have aligned rows and the documented disparity") {
  for (const double side : {1.0, -1.0}) {
    for (const CameraModel model : {CameraModel::pinhole, CameraModel::radial_k3, CameraModel::brown_t2}) {
      CAPTURE(side);
      CAPTURE(r3d::openmvg_camera_model_name(model));
      const StereoRig r = rig(model, side);
      const r3d::DepthRange depth{2.0, 10.0};
      const r3d::RectifiedPair pair =
          r3d::compute_rectification(r.left_cam, r.left_pose, r.right_cam, r.right_pose, depth);
      REQUIRE_MESSAGE(pair.valid, pair.reason);
      CHECK(pair.num_disparities % 16 == 0);
      CHECK(pair.left.intrinsics.height == pair.right.intrinsics.height);
      CHECK(pair.left.intrinsics.cy == pair.right.intrinsics.cy);
      CHECK(pair.left.intrinsics.fx == doctest::Approx(r.left_cam.fx));
      // Rectified size stays close to the source.
      CHECK(pair.left.intrinsics.width < 2 * r.left_cam.width);
      CHECK(pair.left.intrinsics.height < 2 * r.left_cam.height);

      std::mt19937 rng(7);
      std::uniform_real_distribution<double> ux(0.0, r.left_cam.width - 1.0);
      std::uniform_real_distribution<double> uy(0.0, r.left_cam.height - 1.0);
      std::uniform_real_distribution<double> uz(depth.min, depth.max);
      int seen = 0;
      for (int i = 0; i < 500; ++i) {
        // A point seen by the base view inside the depth range.
        const Eigen::Vector2d u = r.left_cam.undistort_pixel(Eigen::Vector2d(ux(rng), uy(rng)));
        const Eigen::Vector3d ray((u.x() - r.left_cam.cx) / r.left_cam.fx, (u.y() - r.left_cam.cy) / r.left_cam.fy, 1.0);
        const Eigen::Vector3d x = r.left_pose.camera_to_world(ray * uz(rng));
        const Eigen::Vector2d pl = pair.left.project(x);
        const Eigen::Vector2d pr = pair.right.project(x);
        R3D_CHECK_NEAR(pl.y(), pr.y(), 1e-9);
        const double d = pl.x() - pr.x();
        R3D_CHECK_NEAR(d, pair.disparity_from_depth(pair.left.depth(x)), 1e-9);
        CHECK(d >= pair.disparity_min);
        CHECK(d <= pair.disparity_min + pair.num_disparities - 1);
        R3D_CHECK_NEAR((pair.triangulate(pl.x(), pl.y(), d) - x).norm(), 0.0, 1e-9);
        if (pair.right.intrinsics.contains(pr)) {
          ++seen;
          CHECK(pair.left.intrinsics.contains(pl, -1.0));
        }
      }
      CHECK(seen > 100);
    }
  }
}

TEST_CASE("Forward motion and zero baseline are rejected") {
  StereoRig r = rig(CameraModel::pinhole);
  r.right_pose = r.left_pose;
  r.right_pose.center += r.left_pose.view_direction() * 0.5;  // straight ahead
  const r3d::RectifiedPair forward =
      r3d::compute_rectification(r.left_cam, r.left_pose, r.right_cam, r.right_pose, r3d::DepthRange{2.0, 10.0});
  CHECK_FALSE(forward.valid);
  CHECK_FALSE(forward.reason.empty());

  r.right_pose = r.left_pose;
  CHECK_FALSE(
      r3d::compute_rectification(r.left_cam, r.left_pose, r.right_cam, r.right_pose, r3d::DepthRange{2.0, 10.0}).valid);
  const StereoRig ok = rig(CameraModel::pinhole);
  CHECK_FALSE(
      r3d::compute_rectification(ok.left_cam, ok.left_pose, ok.right_cam, ok.right_pose, r3d::DepthRange{}).valid);
}

TEST_CASE("rectify_image moves image content to its rectified position") {
  for (const CameraModel model : {CameraModel::pinhole, CameraModel::radial_k3, CameraModel::brown_t2}) {
    CAPTURE(r3d::openmvg_camera_model_name(model));
    const StereoRig r = rig(model);
    const r3d::RectifiedPair pair =
        r3d::compute_rectification(r.left_cam, r.left_pose, r.right_cam, r.right_pose, r3d::DepthRange{2.0, 10.0});
    REQUIRE(pair.valid);
    for (const bool right : {false, true}) {
      const CameraIntrinsics& cam = right ? r.right_cam : r.left_cam;
      const Pose& pose = right ? r.right_pose : r.left_pose;
      const r3d::PinholeCamera& target = right ? pair.right : pair.left;
      // Source image: a Gaussian blob centred on a known rectified pixel.
      const Eigen::Vector2d centre(0.45 * target.intrinsics.width + 0.3, 0.55 * target.intrinsics.height - 0.2);
      const Eigen::Matrix3d m = target.pose.rotation * pose.rotation.transpose();
      r3d::ImageF32 src(cam.width, cam.height, 1);
      for (int y = 0; y < cam.height; ++y) {
        for (int x = 0; x < cam.width; ++x) {
          const Eigen::Vector2d u = cam.undistort(Eigen::Vector2d((x - cam.cx) / cam.fx, (y - cam.cy) / cam.fy));
          const Eigen::Vector3d ray = m * Eigen::Vector3d(u.x(), u.y(), 1.0);
          const Eigen::Vector2d p = target.intrinsics.project(ray);
          src(x, y) = static_cast<float>(1000.0 * std::exp(-(p - centre).squaredNorm() / (2.0 * 2.5 * 2.5)));
        }
      }
      r3d::ImageU8 valid;
      const r3d::ImageF32 rect = r3d::rectify_image(src, cam, pose, target, &valid);
      CHECK(rect.width() == target.intrinsics.width);
      double sw = 0.0, sx = 0.0, sy = 0.0;
      for (int y = 0; y < rect.height(); ++y) {
        for (int x = 0; x < rect.width(); ++x) {
          if ((Eigen::Vector2d(x, y) - centre).norm() < 12.0) {
            CHECK(valid(x, y) == 255);
            sw += rect(x, y);
            sx += rect(x, y) * x;
            sy += rect(x, y) * y;
          }
        }
      }
      R3D_CHECK_NEAR(sx / sw, centre.x(), 0.05);
      R3D_CHECK_NEAR(sy / sw, centre.y(), 0.05);
    }
  }
}

TEST_CASE("Disparities triangulate to the scene points") {
  const StereoRig r = rig(CameraModel::radial_k3);
  const r3d::RectifiedPair pair =
      r3d::compute_rectification(r.left_cam, r.left_pose, r.right_cam, r.right_pose, r3d::DepthRange{2.0, 10.0});
  REQUIRE(pair.valid);
  // A plane 5 units in front of the base camera: exact disparity per rectified pixel.
  const Eigen::Vector3d normal = r.left_pose.view_direction();
  const Eigen::Vector3d on_plane = r.left_pose.center + 5.0 * normal;
  r3d::DisparityMap disp;
  disp.disparity = r3d::ImageF32(pair.left.intrinsics.width, pair.left.intrinsics.height, 1);
  disp.disparity.fill(r3d::kInvalidDisparity);
  for (int y = 0; y < disp.disparity.height(); y += 7) {
    for (int x = 0; x < disp.disparity.width(); x += 5) {
      const Eigen::Vector3d ray = pair.left.unproject(Eigen::Vector2d(x, y), 1.0) - pair.left.pose.center;
      const double t = (on_plane - pair.left.pose.center).dot(normal) / ray.dot(normal);
      const Eigen::Vector3d x_world = pair.left.pose.center + t * ray;
      const Eigen::Vector2d pr = pair.right.project(x_world);
      disp.disparity(x, y) = static_cast<float>(x - pr.x());
    }
  }
  r3d::ImageU8 color(disp.disparity.width(), disp.disparity.height(), 3);
  color.fill(77);
  const r3d::PointCloud cloud = r3d::disparity_to_point_cloud(pair, disp, color.view());
  REQUIRE(cloud.size() > 1000);
  CHECK(cloud.colors.size() == cloud.size());
  REQUIRE(cloud.find_scalar("confidence") != nullptr);
  double worst = 0.0;
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    worst = std::max(worst, std::abs((cloud.world_position(i) - on_plane).dot(normal)));
  }
  // Float disparities and positions: about 1e-5 relative precision.
  CHECK(worst < 1e-3);
}

TEST_CASE("pyramid_camera scales K by 2^-level") {
  const CameraIntrinsics cam = camera(CameraModel::radial_k3);
  const CameraIntrinsics l2 = r3d::pyramid_camera(cam, 2);
  CHECK(l2.fx == doctest::Approx(cam.fx / 4));
  CHECK(l2.cx == doctest::Approx(cam.cx / 4));
  CHECK(l2.width == 80);
  CHECK(l2.height == 60);
  CHECK(l2.params == cam.params);
  // Pixel x at level 0 is pixel x / 4 at level 2.
  const Eigen::Vector3d x_cam(0.3, -0.2, 2.0);
  const Eigen::Vector2d p0 = cam.project(x_cam);
  const Eigen::Vector2d p2 = l2.project(x_cam);
  R3D_CHECK_NEAR(p2.x(), p0.x() / 4, 1e-12);
  R3D_CHECK_NEAR(p2.y(), p0.y() / 4, 1e-12);
}
