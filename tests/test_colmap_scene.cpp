#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <doctest/doctest.h>

#include "r3d/core/error.hpp"
#include "r3d/io/colmap_scene.hpp"
#include "test_util.hpp"

namespace fs = std::filesystem;

namespace {

// Projection exactly as COLMAP defines it (src/colmap/sensor/models.h), in
// COLMAP pixel coordinates (origin at the corner of the top-left pixel).
Eigen::Vector2d colmap_project(const std::string& model, const std::vector<double>& p, const Eigen::Quaterniond& q,
                               const Eigen::Vector3d& t, const Eigen::Vector3d& x) {
  const Eigen::Vector3d xc = q.toRotationMatrix() * x + t;
  const double u = xc.x() / xc.z();
  const double v = xc.y() / xc.z();
  if (model == "OPENCV") {
    const double k1 = p[4], k2 = p[5], p1 = p[6], p2 = p[7];
    const double u2 = u * u, v2 = v * v, uv = u * v, r2 = u2 + v2;
    const double radial = k1 * r2 + k2 * r2 * r2;
    const double du = u * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2);
    const double dv = v * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2);
    return {p[0] * (u + du) + p[2], p[1] * (v + dv) + p[3]};
  }
  // SIMPLE_RADIAL: f, cx, cy, k
  const double radial = p[3] * (u * u + v * v);
  return {p[0] * u * (1.0 + radial) + p[1], p[0] * v * (1.0 + radial) + p[2]};
}

struct Image {
  std::uint32_t id;
  std::uint32_t camera;
  Eigen::Quaterniond q;
  Eigen::Vector3d t;
  std::string name;
};

}  // namespace

TEST_CASE("COLMAP text model import") {
  const r3d::test::TempDir dir;
  const fs::path model = dir.path() / "sparse" / "0";
  const fs::path images_dir = dir.path() / "images";
  fs::create_directories(model);
  fs::create_directories(images_dir / "sub");

  const std::vector<double> opencv = {610.0, 605.0, 320.5, 240.25, -0.08, 0.02, 0.001, -0.0007};
  const std::vector<double> simple_radial = {500.0, 300.0, 200.0, -0.05};
  {
    std::ofstream f(model / "cameras.txt");
    f << "# Camera list with one line of data per camera:\n";
    f << "3 OPENCV 640 480";
    for (const double v : opencv) f << ' ' << v;
    f << "\n1 SIMPLE_RADIAL 600 400";
    for (const double v : simple_radial) f << ' ' << v;
    f << "\n7 FOV 640 480 500 500 320 240 0.9\n";
  }
  const std::vector<Image> images = {
      {5, 3, Eigen::Quaterniond(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY())), Eigen::Vector3d(0.3, 0, 0),
       "sub/b image.jpg"},
      {2, 1, Eigen::Quaterniond(Eigen::AngleAxisd(-0.15, Eigen::Vector3d::UnitY())), Eigen::Vector3d(-0.4, 0.1, 0.2),
       "a.jpg"},
      {9, 7, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero(), "c.jpg"},  // FOV camera: skipped
  };
  const std::vector<Eigen::Vector3d> points = {{0.1, 0.2, 5.0}, {-0.5, 0.3, 6.0}, {0.4, -0.2, 4.5}};
  {
    std::ofstream f(model / "images.txt");
    f.precision(17);
    f << "# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    for (const Image& img : images) {
      f << img.id << ' ' << img.q.w() << ' ' << img.q.x() << ' ' << img.q.y() << ' ' << img.q.z() << ' ' << img.t.x()
        << ' ' << img.t.y() << ' ' << img.t.z() << ' ' << img.camera << ' ' << img.name << '\n';
      // POINTS2D in reverse point order (point i at index 2 - i) plus one untriangulated point.
      for (int i = 2; i >= 0; --i) {
        const std::vector<double>& p = img.camera == 3 ? opencv : simple_radial;
        const Eigen::Vector2d x =
            img.camera == 7 ? Eigen::Vector2d(1, 1)
                            : colmap_project(img.camera == 3 ? "OPENCV" : "SIMPLE_RADIAL", p, img.q, img.t, points[i]);
        f << x.x() << ' ' << x.y() << ' ' << 100 + i << ' ';
      }
      f << "10 10 -1\n";
    }
    std::ofstream(images_dir / "a.jpg") << "x";
    std::ofstream(images_dir / "sub" / "b image.jpg") << "x";
    std::ofstream(images_dir / "c.jpg") << "x";
  }
  {
    std::ofstream f(model / "points3D.txt");
    f.precision(17);
    f << "# 3D point list\n";
    for (int i = 2; i >= 0; --i) {
      f << 100 + i << ' ' << points[i].x() << ' ' << points[i].y() << ' ' << points[i].z() << " 128 128 128 0.5";
      for (const Image& img : images) {
        f << ' ' << img.id << ' ' << 2 - i;
      }
      f << '\n';
    }
    f << "200 0 0 1 0 0 0 0 9 3\n";  // only seen by the skipped image
  }

  r3d::ColmapImportReport report;
  const r3d::Scene scene = r3d::read_colmap_scene(model, {}, &report);
  CHECK(report.num_cameras == 3);
  CHECK(report.num_images == 3);
  CHECK(report.num_images_unsupported == 1);
  CHECK(report.num_points == 4);
  CHECK(report.num_points_dropped == 1);

  REQUIRE(scene.cameras.size() == 2);
  CHECK(scene.cameras[0].model == r3d::CameraModel::radial_k1);  // camera id 1
  CHECK(scene.cameras[0].cx == 300.0 - 0.5);
  CHECK(scene.cameras[1].model == r3d::CameraModel::brown_t2);  // camera id 3
  CHECK(scene.cameras[1].fy == 605.0);
  CHECK(scene.cameras[1].params[3] == 0.001);

  REQUIRE(scene.views.size() == 2);
  CHECK(scene.views[0].source_id == 2);
  CHECK(scene.views[1].source_id == 5);
  CHECK(scene.views[1].image_path == (images_dir / "sub" / "b image.jpg").lexically_normal());
  const Eigen::Vector3d center = -(images[1].q.toRotationMatrix().transpose() * images[1].t);
  CHECK((scene.views[0].pose.center - center).norm() < 1e-12);

  REQUIRE(scene.num_points() == 3);
  CHECK(scene.points[0] == points[0]);  // sorted by point id
  CHECK(scene.track(0).size() == 2);

  // Our camera models + the pixel shift reproduce COLMAP's projections.
  const r3d::ReprojectionStats stats = r3d::compute_reprojection_stats(scene);
  CHECK(stats.num_observations == 6);
  CHECK(stats.max < 1e-3);

  r3d::ColmapReadOptions options;
  options.image_dir = dir.path() / "nowhere";
  CHECK_THROWS_AS(static_cast<void>(r3d::read_colmap_scene(model, options)), r3d::IoError);
  CHECK_THROWS_AS(static_cast<void>(r3d::read_colmap_scene(dir.path() / "missing")), r3d::IoError);
  {
    std::ofstream(model / "points3D.txt") << "1 0 0 1 0 0 0 0 2 99\n";  // POINT2D_IDX out of range
  }
  CHECK_THROWS_AS(static_cast<void>(r3d::read_colmap_scene(model)), r3d::IoError);
}
