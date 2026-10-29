#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <vector>

#include <Eigen/Geometry>
#include <doctest/doctest.h>

#include "r3d/io/image_io.hpp"
#include "r3d/io/ply.hpp"
#include "r3d/pipeline/dense.hpp"
#include "test_util.hpp"

namespace {

struct Plane {
  Eigen::Vector3d normal;
  Eigen::Vector3d point;
  [[nodiscard]] double distance(const Eigen::Vector3d& x) const { return (x - point).dot(normal); }
};

r3d::Pose look_at(const Eigen::Vector3d& center, const Eigen::Vector3d& target) {
  const Eigen::Vector3d z = (target - center).normalized();
  const Eigen::Vector3d x = Eigen::Vector3d::UnitY().cross(z).normalized();
  r3d::Pose pose;
  pose.rotation.row(0) = x.transpose();
  pose.rotation.row(1) = z.cross(x).transpose();
  pose.rotation.row(2) = z.transpose();
  pose.center = center;
  return pose;
}

// Smooth texture on the plane (wavelengths of about 8 to 25 pixels), 4x4 supersampled.
r3d::ImageU8 render(const r3d::PinholeIntrinsics& k, const r3d::Pose& pose, const Plane& plane) {
  const Eigen::Vector3d u = plane.normal.unitOrthogonal();
  const Eigen::Vector3d v = plane.normal.cross(u);
  const std::array<Eigen::Vector3d, 5> waves{Eigen::Vector3d(31.0, 7.0, 0.3), Eigen::Vector3d(-11.0, 23.0, 1.1),
                                             Eigen::Vector3d(5.0, -13.0, 2.0), Eigen::Vector3d(43.0, 29.0, 0.7),
                                             Eigen::Vector3d(-37.0, 47.0, 2.9)};
  r3d::ImageU8 img(k.width, k.height, 1);
  for (int y = 0; y < k.height; ++y) {
    for (int x = 0; x < k.width; ++x) {
      double acc = 0.0;
      for (int sy = 0; sy < 4; ++sy) {
        for (int sx = 0; sx < 4; ++sx) {
          const Eigen::Vector2d px(x - 0.375 + 0.25 * sx, y - 0.375 + 0.25 * sy);
          const Eigen::Vector3d dir = pose.camera_to_world(k.unproject(px, 1.0)) - pose.center;
          const double t = (plane.point - pose.center).dot(plane.normal) / dir.dot(plane.normal);
          const Eigen::Vector3d xw = pose.center + t * dir;
          for (const Eigen::Vector3d& w : waves) {
            acc += std::sin(w.x() * (xw - plane.point).dot(u) + w.y() * (xw - plane.point).dot(v) + w.z());
          }
        }
      }
      img(x, y) = static_cast<std::uint8_t>(std::lround(128.0 + 25.0 * acc / 16.0));
    }
  }
  return img;
}

// Four cameras side by side looking at a slanted textured plane 5 units away,
// images written to `dir`, sparse points on the plane.
r3d::Scene plane_scene(const std::filesystem::path& dir, Plane& plane) {
  plane.normal = Eigen::Vector3d(0.2, -0.1, -1.0).normalized();
  plane.point = Eigen::Vector3d(0.0, 0.0, 5.0);
  r3d::Scene scene;
  r3d::CameraIntrinsics cam;
  cam.width = 320;
  cam.height = 240;
  cam.fx = cam.fy = 300.0;
  cam.cx = 159.5;
  cam.cy = 119.5;
  scene.cameras.push_back(cam);
  for (int i = 0; i < 4; ++i) {
    r3d::SceneView view;
    view.source_id = static_cast<std::uint32_t>(10 + i);
    view.width = cam.width;
    view.height = cam.height;
    view.pose = look_at(Eigen::Vector3d(-0.6 + 0.4 * i, 0.0, 0.0), plane.point);
    view.image_path = dir / std::format("view{}.png", i);
    r3d::write_png(view.image_path, render(cam.pinhole(), view.pose, plane).view(), 1);
    scene.views.push_back(view);
  }
  const Eigen::Vector3d u = plane.normal.unitOrthogonal();
  const Eigen::Vector3d v = plane.normal.cross(u);
  std::vector<r3d::TrackObservation> track;
  for (int a = -6; a <= 6; ++a) {
    for (int b = -6; b <= 6; ++b) {
      const Eigen::Vector3d x = plane.point + 0.25 * a * u + 0.25 * b * v;
      track.clear();
      for (std::uint32_t vi = 0; vi < scene.views.size(); ++vi) {
        const Eigen::Vector2d px = cam.project(scene.views[vi].pose.world_to_camera(x));
        if (cam.pinhole().contains(px)) {
          track.push_back({.view = vi, .x = static_cast<float>(px.x()), .y = static_cast<float>(px.y())});
        }
      }
      if (track.size() >= 2) {
        scene.add_point(x, track);
      }
    }
  }
  return scene;
}

}  // namespace

TEST_CASE("Dense presets") {
  for (const r3d::DensePreset p :
       {r3d::DensePreset::fast, r3d::DensePreset::normal, r3d::DensePreset::high, r3d::DensePreset::ultra}) {
    CHECK(r3d::parse_dense_preset(r3d::to_string(p)) == p);
  }
  CHECK_FALSE(r3d::parse_dense_preset("medium").has_value());
  // Larger presets never use smaller depth maps.
  const auto size = [](r3d::DensePreset p) {
    const int s = r3d::dense_preset(p).depth.max_image_size;
    return s > 0 ? s : 1 << 30;
  };
  CHECK(size(r3d::DensePreset::fast) < size(r3d::DensePreset::normal));
  CHECK(size(r3d::DensePreset::normal) < size(r3d::DensePreset::high));
  CHECK(size(r3d::DensePreset::high) <= size(r3d::DensePreset::ultra));
}

TEST_CASE("depth_map_resolution limits the longer side") {
  r3d::Scene scene;
  r3d::CameraIntrinsics cam;
  cam.width = 4000;
  cam.height = 3000;
  cam.fx = cam.fy = 3000.0;
  cam.cx = 1999.5;
  cam.cy = 1499.5;
  scene.cameras.push_back(cam);
  scene.views.emplace_back();
  r3d::DepthMapOptions o;
  CHECK(r3d::depth_map_resolution(scene, 0, o) == 1.0);
  o.max_image_size = 1000;
  CHECK(r3d::depth_map_resolution(scene, 0, o) == doctest::Approx(0.25));
  o.resolution = 0.1;
  CHECK(r3d::depth_map_resolution(scene, 0, o) == doctest::Approx(0.1));
}

TEST_CASE("densify: a plane from four views, with progress and cancellation") {
  const r3d::test::TempDir dir;
  Plane plane;
  const r3d::Scene scene = plane_scene(dir.path(), plane);
  REQUIRE(scene.num_points() > 50);
  r3d::DenseOptions options = r3d::dense_preset(r3d::DensePreset::normal);
  options.fusion.with_views = true;

  SUBCASE("points on the plane, progress in order up to 1") {
    std::vector<r3d::DenseProgress> steps;
    r3d::DenseCallbacks callbacks;
    callbacks.progress = [&](const r3d::DenseProgress& p) { steps.push_back(p); };
    std::size_t points = 0;
    std::vector<double> distance;
    bool views_are_source_ids = true;
    r3d::DenseReport report;
    r3d::densify(
        scene, options,
        [&](r3d::PointCloud&& part) {
          points += part.size();
          for (std::size_t i = 0; i < part.size(); ++i) {
            distance.push_back(std::abs(plane.distance(part.positions[i].cast<double>())));
            for (const std::uint32_t v : part.visibility[i]) {
              views_are_source_ids = views_are_source_ids && v >= 10 && v < 14;
            }
          }
        },
        callbacks, &report);
    CHECK(report.views == 4);
    CHECK(report.pairs_matched > 0);
    CHECK(report.fusion.output_points == points);
    CHECK(points > 20000);
    // GSD 0.017: median 0.13 GSD (0.04% of the distance), a few outliers at the image borders.
    std::sort(distance.begin(), distance.end());
    CHECK(distance[distance.size() / 2] < 0.005);
    CHECK(distance[distance.size() * 99 / 100] < 0.04);
    CHECK(distance.back() < 0.25);
    CHECK(views_are_source_ids);
    REQUIRE(!steps.empty());
    for (std::size_t i = 1; i < steps.size(); ++i) {
      CHECK(steps[i].fraction >= steps[i - 1].fraction);
      CHECK(steps[i].stage >= steps[i - 1].stage);
    }
    CHECK(steps.back().fraction == 1.0);
    CHECK(steps.back().stage == r3d::DenseStage::fusion);
  }

  SUBCASE("densify_to_ply writes the layout") {
    const std::filesystem::path out = dir.path() / "cloud.ply";
    r3d::CloudLayoutOptions layout;
    layout.layout = r3d::CloudLayout::poisson;
    r3d::densify_to_ply(scene, options, out, layout);
    const r3d::PointCloud cloud = r3d::read_ply(out);
    CHECK(cloud.size() > 20000);
    CHECK(cloud.has_normals());
    CHECK(cloud.scalars.empty());
  }

  SUBCASE("cancellation in every stage stops with Cancelled and removes the file") {
    for (const r3d::DenseStage stage : {r3d::DenseStage::depth_maps, r3d::DenseStage::refinement,
                                        r3d::DenseStage::consistency, r3d::DenseStage::visibility,
                                        r3d::DenseStage::fusion}) {
      CAPTURE(r3d::to_string(stage));
      r3d::CancellationToken token;
      r3d::DenseCallbacks callbacks;
      callbacks.cancel = &token;
      r3d::DenseStage last = r3d::DenseStage::depth_maps;
      callbacks.progress = [&](const r3d::DenseProgress& p) {
        last = p.stage;
        if (p.stage == stage && p.done > 0) {
          token.request();
        }
      };
      const std::filesystem::path out = dir.path() / "cancelled.ply";
      CHECK_THROWS_AS(r3d::densify_to_ply(scene, options, out, {}, callbacks), r3d::Cancelled);
      CHECK(last == stage);
      CHECK_FALSE(std::filesystem::exists(out));
    }
  }

  SUBCASE("a work directory caches the depth maps for reuse") {
    options.work_dir = dir.path() / "work";
    r3d::DenseReport first;
    r3d::densify(scene, options, [](r3d::PointCloud&&) {}, {}, &first);
    CHECK(first.cached_views == 0);
    options.reuse_refined = true;
    r3d::DenseReport second;
    r3d::densify(scene, options, [](r3d::PointCloud&&) {}, {}, &second);
    CHECK(second.cached_views == 4);
    CHECK(second.pairs_matched == 0);
    CHECK(second.fusion.output_points == first.fusion.output_points);
  }
}
