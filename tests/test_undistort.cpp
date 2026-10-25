#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <doctest/doctest.h>

#include "r3d/core/camera_model.hpp"
#include "r3d/core/error.hpp"
#include "r3d/core/image.hpp"
#include "r3d/scene/undistort.hpp"
#include "test_util.hpp"

using r3d::CameraIntrinsics;
using r3d::CameraModel;
using r3d::PinholeIntrinsics;

namespace {

CameraIntrinsics barrel_camera() {
  CameraIntrinsics cam;
  cam.model = CameraModel::radial_k3;
  cam.width = 400;
  cam.height = 300;
  cam.fx = cam.fy = 300.0;
  cam.cx = 201.3;
  cam.cy = 148.6;
  cam.params = {-0.25, 0.08, -0.01};
  return cam;
}

CameraIntrinsics brown_camera() {
  CameraIntrinsics cam = barrel_camera();
  cam.model = CameraModel::brown_t2;
  cam.params = {0.12, -0.05, 0.01, 0.002, -0.0015};  // pincushion + tangential
  return cam;
}

r3d::UndistortOptions fit(r3d::UndistortFit mode) {
  r3d::UndistortOptions options;
  options.fit = mode;
  return options;
}

// Renders an image of `cam` whose content is the function `f` of the
// undistorted pixel position in the `target` pinhole camera.
template <typename F>
r3d::ImageF32 render_distorted(const CameraIntrinsics& cam, const PinholeIntrinsics& target, F&& f) {
  r3d::ImageF32 img(cam.width, cam.height, 1);
  for (int y = 0; y < cam.height; ++y) {
    for (int x = 0; x < cam.width; ++x) {
      const Eigen::Vector2d u = cam.undistort(Eigen::Vector2d((x - cam.cx) / cam.fx, (y - cam.cy) / cam.fy));
      img(x, y) = static_cast<float>(f(Eigen::Vector2d(target.fx * u.x() + target.cx, target.fy * u.y() + target.cy)));
    }
  }
  return img;
}

}  // namespace

TEST_CASE("Undistorted camera covers the source image as requested") {
  for (const CameraIntrinsics& cam : {barrel_camera(), brown_camera()}) {
    CAPTURE(r3d::openmvg_camera_model_name(cam.model));
    REQUIRE(r3d::distortion_is_monotonic(cam));

    const PinholeIntrinsics same = r3d::undistorted_camera(cam, fit(r3d::UndistortFit::same_size));
    CHECK(same.width == cam.width);
    CHECK(same.cx == cam.cx);

    // keep_all: every source border point lands inside the output image.
    const PinholeIntrinsics all = r3d::undistorted_camera(cam);
    CHECK(all.fx == cam.fx);
    for (int i = 0; i <= 100; ++i) {
      const double t = i / 100.0;
      for (const Eigen::Vector2d& p :
           {Eigen::Vector2d(-0.5, -0.5 + t * cam.height), Eigen::Vector2d(cam.width - 0.5, -0.5 + t * cam.height),
            Eigen::Vector2d(-0.5 + t * cam.width, -0.5), Eigen::Vector2d(-0.5 + t * cam.width, cam.height - 0.5)}) {
        const Eigen::Vector2d u = cam.undistort_pixel(p) - Eigen::Vector2d(cam.cx, cam.cy) + Eigen::Vector2d(all.cx, all.cy);
        CHECK(u.x() >= -0.5 - 1e-6);
        CHECK(u.y() >= -0.5 - 1e-6);
        CHECK(u.x() <= all.width - 0.5 + 1e-6);
        CHECK(u.y() <= all.height - 0.5 + 1e-6);
      }
    }

    // inner: every output pixel samples inside the source image.
    const PinholeIntrinsics inner = r3d::undistorted_camera(cam, fit(r3d::UndistortFit::inner));
    CHECK(inner.width <= all.width);
    CHECK(inner.height <= all.height);
    r3d::ImageU8 valid;
    const r3d::ImageF32 src(cam.width, cam.height, 1);
    static_cast<void>(r3d::undistort_image(src, cam, inner, &valid));
    int invalid = 0;
    for (int y = 0; y < valid.height(); ++y) {
      for (int x = 0; x < valid.width(); ++x) {
        invalid += valid(x, y) == 0 ? 1 : 0;
      }
    }
    CHECK(invalid == 0);
  }
}

TEST_CASE("Undistortion moves features to their pinhole positions (< 0.05 px)") {
  for (const CameraIntrinsics& cam : {barrel_camera(), brown_camera()}) {
    CAPTURE(r3d::openmvg_camera_model_name(cam.model));
    const PinholeIntrinsics target = r3d::undistorted_camera(cam, fit(r3d::UndistortFit::inner));

    // Gaussian blobs (sigma 1.5 px in the undistorted image) on a grid.
    constexpr double kSigma = 1.5;
    std::vector<Eigen::Vector2d> blobs;
    for (int j = 0; j < 5; ++j) {
      for (int i = 0; i < 7; ++i) {
        blobs.emplace_back(30.0 + i * (target.width - 60.0) / 6.0 + 0.37 * i, 30.0 + j * (target.height - 60.0) / 4.0 + 0.21 * j);
      }
    }
    const r3d::ImageF32 src = render_distorted(cam, target, [&](const Eigen::Vector2d& p) {
      double v = 0.0;
      for (const Eigen::Vector2d& b : blobs) {
        v += std::exp(-(p - b).squaredNorm() / (2.0 * kSigma * kSigma));
      }
      return v;
    });
    r3d::ImageU8 valid;
    const r3d::ImageF32 out = r3d::undistort_image(src, cam, target, &valid);

    double max_error = 0.0;
    for (const Eigen::Vector2d& b : blobs) {
      double sum = 0.0;
      Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
      const int r = static_cast<int>(std::ceil(4 * kSigma));
      for (int y = static_cast<int>(std::lround(b.y())) - r; y <= static_cast<int>(std::lround(b.y())) + r; ++y) {
        for (int x = static_cast<int>(std::lround(b.x())) - r; x <= static_cast<int>(std::lround(b.x())) + r; ++x) {
          REQUIRE(valid(x, y) == 255);
          sum += out(x, y);
          centroid += out(x, y) * Eigen::Vector2d(x, y);
        }
      }
      max_error = std::max(max_error, (centroid / sum - b).norm());
    }
    CHECK(max_error < 0.05);
    MESSAGE("max blob centroid error ", max_error, " px");
  }
}

TEST_CASE("Undistorted image matches the analytic pinhole image") {
  const CameraIntrinsics cam = barrel_camera();
  const PinholeIntrinsics target = r3d::undistorted_camera(cam, fit(r3d::UndistortFit::inner));
  const auto pattern = [](const Eigen::Vector2d& p) {
    return 0.5 + 0.5 * std::sin(2.0 * std::numbers::pi * p.x() / 17.0) * std::cos(2.0 * std::numbers::pi * p.y() / 23.0);
  };
  const r3d::ImageF32 src = render_distorted(cam, target, pattern);
  const r3d::ImageF32 out = r3d::undistort_image(src, cam, target);
  double sum_sq = 0.0;
  int n = 0;
  for (int y = 2; y < target.height - 2; ++y) {
    for (int x = 2; x < target.width - 2; ++x) {
      const double e = out(x, y) - pattern(Eigen::Vector2d(x, y));
      sum_sq += e * e;
      ++n;
    }
  }
  CHECK(std::sqrt(sum_sq / n) < 0.01);
}

TEST_CASE("Undistortion of 8-bit colour images, masks and errors") {
  const CameraIntrinsics cam = barrel_camera();
  r3d::ImageU8 src(cam.width, cam.height, 3);
  for (int y = 0; y < cam.height; ++y) {
    for (int x = 0; x < cam.width; ++x) {
      src(x, y, 0) = 200;
      src(x, y, 1) = static_cast<std::uint8_t>(x % 256);
      src(x, y, 2) = 7;
    }
  }
  const PinholeIntrinsics target = r3d::undistorted_camera(cam);
  CHECK(target.width > cam.width);  // barrel distortion: the undistorted image is larger
  r3d::ImageU8 valid;
  const r3d::ImageU8 out = r3d::undistort_image(src, cam, target, &valid);
  CHECK(out.channels() == 3);
  const auto cx = static_cast<int>(std::lround(target.cx));
  const auto cy = static_cast<int>(std::lround(target.cy));
  CHECK(valid(cx, cy) == 255);
  CHECK(out(cx, cy, 0) == 200);
  CHECK(out(cx, cy, 2) == 7);
  CHECK(valid(0, 0) == 0);  // corners of the barrel-undistorted image are empty
  CHECK(out(0, 0, 0) == 0);

  const r3d::ImageU8 wrong_size(10, 10, 3);
  CHECK_THROWS_AS(static_cast<void>(r3d::undistort_image(wrong_size, cam, target)), r3d::Error);
}

TEST_CASE("Folding distortion is detected") {
  CameraIntrinsics cam = barrel_camera();
  cam.params = {-0.9, 0.0, 0.0};  // r_d = r (1 - 0.9 r^2) folds at r = 0.61 < image corner radius
  CHECK_FALSE(r3d::distortion_is_monotonic(cam));
  CHECK(r3d::distortion_is_monotonic(barrel_camera()));
  // Without a valid inverse at the border the original geometry is kept.
  const PinholeIntrinsics target = r3d::undistorted_camera(cam);
  CHECK(target.width == cam.width);
}
