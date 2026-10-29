#include <cstdint>
#include <fstream>
#include <string>
#include <utility>

#include <doctest/doctest.h>

#include <Eigen/Geometry>

#include "r3d/io/image_io.hpp"
#include "r3d/io/mesher_export.hpp"
#include "test_util.hpp"

namespace {

r3d::PointCloud fused_cloud() {
  r3d::PointCloud cloud;
  for (int i = 0; i < 10; ++i) {
    cloud.positions.emplace_back(static_cast<float>(i), 0.0F, 1.0F);
    cloud.normals.emplace_back(0.0F, 0.0F, -1.0F);
    cloud.colors.push_back({10, 20, 30});
    cloud.visibility.push_back({1, 2});
  }
  cloud.scalar("value");
  cloud.scalar("confidence");
  for (int i = 0; i < 10; ++i) {
    cloud.scalar("value")[static_cast<std::size_t>(i)] = 0.001F * static_cast<float>(i + 1);
    cloud.scalar("confidence")[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
  }
  return cloud;
}

// Reads the two lines of a texrecon .cam file.
struct CamFile {
  Eigen::Vector3d t;
  Eigen::Matrix3d r;
  double flen = 0.0, d0 = 0.0, d1 = 0.0, paspect = 0.0, ppx = 0.0, ppy = 0.0;
};

CamFile read_cam(const std::filesystem::path& path) {
  std::ifstream in(path);
  CamFile c;
  in >> c.t.x() >> c.t.y() >> c.t.z();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      in >> c.r(i, j);
    }
  }
  in >> c.flen >> c.d0 >> c.d1 >> c.paspect >> c.ppx >> c.ppy;
  REQUIRE(in);
  return c;
}

// Pixel of x_world as MVE computes it (CameraInfo::fill_calibration), in the
// pixel corner convention: pixel i covers [i, i + 1).
Eigen::Vector2d mve_project(const CamFile& c, int width, int height, const Eigen::Vector3d& x_world) {
  const double image_aspect = static_cast<double>(width) / height * c.paspect;
  double ax = 0.0, ay = 0.0;
  if (image_aspect < 1.0) {
    ax = c.flen * height / c.paspect;
    ay = c.flen * height;
  } else {
    ax = c.flen * width;
    ay = c.flen * width * c.paspect;
  }
  const Eigen::Vector3d x = c.r * x_world + c.t;
  return {ax * x.x() / x.z() + width * c.ppx, ay * x.y() / x.z() + height * c.ppy};
}

r3d::Pose test_pose() {
  r3d::Pose pose;
  pose.rotation = Eigen::AngleAxisd(0.3, Eigen::Vector3d(1.0, -2.0, 0.5).normalized()).toRotationMatrix();
  pose.center = Eigen::Vector3d(0.4, -1.2, -3.0);
  return pose;
}

}  // namespace

TEST_CASE("Cloud layouts for PoissonRecon and FSSR") {
  const r3d::PointCloud cloud = fused_cloud();
  SUBCASE("full keeps everything") {
    const r3d::PointCloud out = r3d::cloud_for_mesher(cloud, {});
    CHECK(out.scalars.size() == 2);
    CHECK(out.has_visibility());
  }
  SUBCASE("poisson: positions, normals, colours; normal length from the confidence") {
    r3d::CloudLayoutOptions options;
    options.layout = r3d::CloudLayout::poisson;
    r3d::PointCloud out = r3d::cloud_for_mesher(cloud, options);
    CHECK(out.scalars.empty());
    CHECK(!out.has_visibility());
    CHECK(out.has_colors());
    R3D_CHECK_NEAR(out.normals[0].norm(), 0.25F, 1e-6F);  // default: full length at 4 fused depths
    R3D_CHECK_NEAR(out.normals[9].norm(), 1.0F, 1e-6F);
    options.poisson_full_confidence = 0.0;
    out = r3d::cloud_for_mesher(cloud, options);
    R3D_CHECK_NEAR(out.normals[0].norm(), 1.0F, 1e-6F);
  }
  SUBCASE("fssr: scale and confidence in (0, 1]") {
    r3d::CloudLayoutOptions options;
    options.layout = r3d::CloudLayout::fssr;
    options.fssr_scale = 3.0;
    options.fssr_full_confidence = 5.0;
    const r3d::PointCloud out = r3d::cloud_for_mesher(cloud, options);
    REQUIRE(out.find_scalar("value") != nullptr);
    REQUIRE(out.find_scalar("confidence") != nullptr);
    R3D_CHECK_NEAR((*out.find_scalar("value"))[1], 0.006F, 1e-7F);
    R3D_CHECK_NEAR((*out.find_scalar("confidence"))[1], 0.4F, 1e-6F);
    R3D_CHECK_NEAR((*out.find_scalar("confidence"))[8], 1.0F, 1e-6F);
    CHECK(!out.has_visibility());
  }
  SUBCASE("missing attributes are reported") {
    r3d::PointCloud bare = cloud;
    bare.scalars.clear();
    r3d::CloudLayoutOptions options;
    options.layout = r3d::CloudLayout::fssr;
    CHECK_THROWS(static_cast<void>(r3d::cloud_for_mesher(bare, options)));
    bare.normals.clear();
    options.layout = r3d::CloudLayout::poisson;
    CHECK_THROWS(static_cast<void>(r3d::cloud_for_mesher(bare, options)));
  }
  CHECK(r3d::parse_cloud_layout("fssr") == r3d::CloudLayout::fssr);
  CHECK(!r3d::parse_cloud_layout("delaunay").has_value());
}

TEST_CASE("texrecon cameras project like ours (pixel corner convention)") {
  const r3d::test::TempDir dir;
  const r3d::Pose pose = test_pose();
  for (const auto& [w, h] : {std::pair{640, 480}, std::pair{480, 640}}) {
    const r3d::PinholeIntrinsics k{w, h, 520.0, 523.0, 0.47 * w, 0.52 * h};
    const std::filesystem::path path = dir.path() / "view.cam";
    r3d::write_texrecon_camera(path, k, pose);
    const CamFile c = read_cam(path);
    CHECK(c.d0 == 0.0);
    const r3d::PinholeCamera ours{k, pose};
    for (const Eigen::Vector3d& x : {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.5, -0.3, 0.8)}) {
      const Eigen::Vector2d expected = ours.project(x) + Eigen::Vector2d(0.5, 0.5);
      const Eigen::Vector2d got = mve_project(c, w, h, x);
      R3D_CHECK_NEAR(got.x(), expected.x(), 1e-9);
      R3D_CHECK_NEAR(got.y(), expected.y(), 1e-9);
    }
  }
}

TEST_CASE("texrecon scene folder: copied and undistorted images with cameras") {
  const r3d::test::TempDir dir;
  r3d::ImageU8 img(64, 48, 3);
  for (int y = 0; y < img.height(); ++y) {
    for (int x = 0; x < img.width(); ++x) {
      for (int c = 0; c < 3; ++c) {
        img(x, y, c) = static_cast<std::uint8_t>((x * 4 + y * 2 + c * 60) % 256);
      }
    }
  }
  r3d::write_jpeg(dir.path() / "a.JPG", img.view(), 95);
  r3d::write_png(dir.path() / "b.png", img.view());
  r3d::Scene scene;
  r3d::CameraIntrinsics pinhole;
  pinhole.width = 64;
  pinhole.height = 48;
  pinhole.fx = pinhole.fy = 60.0;
  pinhole.cx = 31.5;
  pinhole.cy = 23.5;
  r3d::CameraIntrinsics radial = pinhole;
  radial.model = r3d::CameraModel::radial_k1;
  radial.params[0] = -0.05;
  scene.cameras = {pinhole, radial};
  scene.views.push_back({7, dir.path() / "a.JPG", 64, 48, 0, test_pose()});
  scene.views.push_back({8, dir.path() / "b.png", 64, 48, 1, test_pose()});
  scene.views.push_back({9, dir.path() / "a.JPG", 64, 48, 1, test_pose()});  // same name as view 7
  const std::filesystem::path out = dir.path() / "texrecon";
  r3d::TexreconExportReport report;
  r3d::write_texrecon_scene(scene, out, {}, &report);
  CHECK(report.images == 3);
  CHECK(report.copied == 1);
  for (const char* name : {"a_7", "b", "a_9"}) {
    CHECK(std::filesystem::exists(out / (std::string(name) + ".cam")));
    REQUIRE(std::filesystem::exists(out / (std::string(name) + ".jpg")));
    const r3d::ImageFileInfo info = r3d::read_image_info(out / (std::string(name) + ".jpg"));
    CHECK(info.width == 64);
    CHECK(info.height == 48);
  }
  CHECK(std::filesystem::file_size(out / "a_7.jpg") == std::filesystem::file_size(dir.path() / "a.JPG"));
}
