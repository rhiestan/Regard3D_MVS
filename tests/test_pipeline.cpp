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

// Smooth texture on the plane (wavelengths of about 6 to 19 pixels), 2x2 supersampled.
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
      for (int sy = 0; sy < 2; ++sy) {
        for (int sx = 0; sx < 2; ++sx) {
          const Eigen::Vector2d px(x - 0.25 + 0.5 * sx, y - 0.25 + 0.5 * sy);
          const Eigen::Vector3d dir = pose.camera_to_world(k.unproject(px, 1.0)) - pose.center;
          const double t = (plane.point - pose.center).dot(plane.normal) / dir.dot(plane.normal);
          const Eigen::Vector3d xw = pose.center + t * dir;
          for (const Eigen::Vector3d& w : waves) {
            acc += std::sin(w.x() * (xw - plane.point).dot(u) + w.y() * (xw - plane.point).dot(v) + w.z());
          }
        }
      }
      img(x, y) = static_cast<std::uint8_t>(std::lround(128.0 + 25.0 * acc / 4.0));
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
  cam.width = 240;
  cam.height = 180;
  cam.fx = cam.fy = 225.0;
  cam.cx = 119.5;
  cam.cy = 89.5;
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
  // Only `normal` refines (D53).
  CHECK_FALSE(r3d::dense_preset(r3d::DensePreset::fast).refine);
  CHECK(r3d::dense_preset(r3d::DensePreset::normal).refine);
  CHECK_FALSE(r3d::dense_preset(r3d::DensePreset::high).refine);
  CHECK_FALSE(r3d::dense_preset(r3d::DensePreset::ultra).refine);
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

TEST_CASE("densify: a plane from four views, with progress, reuse and cancellation") {
  // One full run; the reuse and cancellation checks start from its cached depth maps, which keeps
  // the test short in sanitizer builds.
  const r3d::test::TempDir dir;
  Plane plane;
  const r3d::Scene scene = plane_scene(dir.path(), plane);
  REQUIRE(scene.num_points() > 50);
  r3d::DenseOptions options = r3d::dense_preset(r3d::DensePreset::normal);
  options.fusion.with_views = true;
  options.work_dir = dir.path() / "work";

  std::vector<r3d::DenseProgress> steps;
  r3d::DenseCallbacks callbacks;
  callbacks.progress = [&](const r3d::DenseProgress& p) { steps.push_back(p); };
  std::size_t points = 0;
  std::vector<double> distance;
  bool views_are_source_ids = true;
  r3d::DenseReport first;
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
      callbacks, &first);
  CHECK(first.views == 4);
  CHECK(first.cached_views == 0);
  CHECK(first.pairs_matched > 0);
  CHECK(first.fusion.output_points == points);
  CHECK(points > 20000);
  // GSD 0.022: median 0.1 GSD (0.04% of the distance), a few outliers at the image borders.
  REQUIRE(!distance.empty());
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

  // The refined maps from the work directory, written in the poisson layout.
  r3d::DenseOptions reuse = options;
  reuse.reuse_refined = true;
  const std::filesystem::path out = dir.path() / "cloud.ply";
  r3d::CloudLayoutOptions layout;
  layout.layout = r3d::CloudLayout::poisson;
  r3d::DenseReport second;
  r3d::densify_to_ply(scene, reuse, out, layout, {}, &second);
  CHECK(second.cached_views == 4);
  CHECK(second.pairs_matched == 0);
  CHECK(second.fusion.output_points == points);
  const r3d::PointCloud cloud = r3d::read_ply(out);
  CHECK(cloud.size() == points);
  CHECK(cloud.has_normals());
  CHECK(cloud.scalars.empty());

  // Cancellation in every stage stops with Cancelled and removes the file. Matching runs without
  // the cache, refinement from the cached matched maps, the later stages from the refined maps.
  for (const r3d::DenseStage stage : {r3d::DenseStage::depth_maps, r3d::DenseStage::refinement,
                                      r3d::DenseStage::consistency, r3d::DenseStage::visibility,
                                      r3d::DenseStage::fusion}) {
    CAPTURE(r3d::to_string(stage));
    r3d::DenseOptions o = options;
    if (stage == r3d::DenseStage::depth_maps) {
      o.work_dir.clear();
    } else if (stage == r3d::DenseStage::refinement) {
      o.reuse = true;
    } else {
      o.reuse_refined = true;
    }
    r3d::CancellationToken token;
    r3d::DenseCallbacks cancel;
    cancel.cancel = &token;
    r3d::DenseStage last = r3d::DenseStage::depth_maps;
    cancel.progress = [&](const r3d::DenseProgress& p) {
      last = p.stage;
      if (p.stage == stage && p.done > 0) {
        token.request();
      }
    };
    const std::filesystem::path cancelled = dir.path() / "cancelled.ply";
    CHECK_THROWS_AS(r3d::densify_to_ply(scene, o, cancelled, {}, cancel), r3d::Cancelled);
    CHECK(last == stage);
    CHECK_FALSE(std::filesystem::exists(cancelled));
  }
}
