#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#include <Eigen/Geometry>
#include <doctest/doctest.h>

#include "r3d/io/ply.hpp"
#include "r3d/io/scene_export.hpp"
#include "r3d/select/view_selection.hpp"
#include "test_util.hpp"

namespace {

// Camera at `center` looking at `target` (OpenMVG convention: rows of R are
// the camera axes in world coordinates, +z forward, +y down).
r3d::Pose look_at(const Eigen::Vector3d& center, const Eigen::Vector3d& target) {
  const Eigen::Vector3d z = (target - center).normalized();
  Eigen::Vector3d x = Eigen::Vector3d::UnitY().cross(z);
  if (x.norm() < 1e-9) {
    x = Eigen::Vector3d::UnitX();
  }
  x.normalize();
  const Eigen::Vector3d y = z.cross(x);
  r3d::Pose pose;
  pose.rotation.row(0) = x.transpose();
  pose.rotation.row(1) = y.transpose();
  pose.rotation.row(2) = z.transpose();
  pose.center = center;
  return pose;
}

// Adds random points in a ball and their observations in every view that sees them.
void add_points(r3d::Scene& scene, const Eigen::Vector3d& center, double radius, int count, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<r3d::TrackObservation> track;
  for (int i = 0; i < count;) {
    const Eigen::Vector3d p(u(rng), u(rng), u(rng));
    if (p.norm() > 1.0) {
      continue;
    }
    ++i;
    const Eigen::Vector3d x = center + radius * p;
    track.clear();
    for (std::uint32_t v = 0; v < scene.views.size(); ++v) {
      const r3d::SceneView& view = scene.views[v];
      const Eigen::Vector3d xc = view.pose.world_to_camera(x);
      if (xc.z() <= 0.0) {
        continue;
      }
      const Eigen::Vector2d px = scene.cameras[view.camera].project(xc);
      if (scene.cameras[view.camera].pinhole().contains(px)) {
        track.push_back({v, static_cast<float>(px.x()), static_cast<float>(px.y())});
      }
    }
    if (track.size() >= 2) {
      scene.add_point(x, track);
    }
  }
}

r3d::Scene make_scene(const std::vector<r3d::Pose>& poses) {
  r3d::Scene scene;
  r3d::CameraIntrinsics cam;
  cam.width = 800;
  cam.height = 600;
  cam.fx = cam.fy = 700.0;
  cam.cx = 399.5;
  cam.cy = 299.5;
  scene.cameras.push_back(cam);
  for (std::size_t i = 0; i < poses.size(); ++i) {
    r3d::SceneView view;
    view.source_id = static_cast<std::uint32_t>(100 + i);
    view.width = cam.width;
    view.height = cam.height;
    view.pose = poses[i];
    scene.views.push_back(view);
  }
  return scene;
}

// 24 cameras on a ring of radius 10 (15 degrees apart) around the origin,
// plus one camera at 3x the distance behind camera 0.
r3d::Scene ring_scene() {
  std::vector<r3d::Pose> poses;
  for (int i = 0; i < 24; ++i) {
    const double a = 2.0 * std::numbers::pi * i / 24.0;
    poses.push_back(look_at(Eigen::Vector3d(10.0 * std::cos(a), 0.3 * std::sin(3 * a), 10.0 * std::sin(a)),
                            Eigen::Vector3d::Zero()));
  }
  poses.push_back(look_at(Eigen::Vector3d(30.0, 0.0, 1.0), Eigen::Vector3d::Zero()));
  r3d::Scene scene = make_scene(poses);
  add_points(scene, Eigen::Vector3d::Zero(), 2.0, 4000, 1);
  return scene;
}

int ring_distance(std::uint32_t a, std::uint32_t b) {
  const int d = std::abs(static_cast<int>(a) - static_cast<int>(b)) % 24;
  return std::min(d, 24 - d);
}

}  // namespace

TEST_CASE("View selection on a camera ring picks near neighbours on both sides") {
  const r3d::Scene scene = ring_scene();
  REQUIRE(scene.num_points() > 1000);
  const r3d::ViewSelection selection = r3d::select_views(scene);
  REQUIRE(selection.partners.size() == scene.views.size());

  for (std::uint32_t v = 0; v < 24; ++v) {
    CAPTURE(v);
    const auto& partners = selection.partners[v];
    REQUIRE(partners.size() == 4);
    bool left = false;
    bool right = false;
    for (const r3d::StereoPartner& p : partners) {
      CHECK(p.view != 24);  // 3x the distance: GSD ratio 3 > max_scale_ratio
      CHECK(ring_distance(v, p.view) <= 3);
      CHECK(p.mean_angle_deg < 50.0);
      CHECK(p.score > 0.0);
      CHECK_FALSE(p.epipole_in_image);
      left = left || p.view == (v + 23) % 24 || p.view == (v + 22) % 24;
      right = right || p.view == (v + 1) % 24 || p.view == (v + 2) % 24;
      // Points lie at distance 10 +- 2 from every ring camera.
      CHECK(p.depth.valid());
      CHECK(p.depth.min >= 8.0 / 1.1 - 1e-9);
      CHECK(p.depth.max <= 12.0 * 1.1 + 1e-9);
    }
    CHECK(left);
    CHECK(right);
    // Partners are ordered by selection; the first one scores highest.
    for (const r3d::StereoPartner& p : partners) {
      CHECK(p.score <= partners.front().score);
    }
    CHECK(selection.view_depth[v].min < 8.5);
    CHECK(selection.view_depth[v].max > 11.5);
  }
  CHECK(selection.partners[24].empty());  // the far camera has no partner within the GSD limit

  const auto pairs = selection.stereo_pairs();
  CHECK(std::is_sorted(pairs.begin(), pairs.end()));
  CHECK(std::adjacent_find(pairs.begin(), pairs.end()) == pairs.end());
  for (const auto& [a, b] : pairs) {
    CHECK(a < b);
  }

  // Deterministic and independent of the thread count / hash order.
  const r3d::ViewSelection again = r3d::select_views(scene);
  for (std::size_t v = 0; v < scene.views.size(); ++v) {
    REQUIRE(again.partners[v].size() == selection.partners[v].size());
    for (std::size_t k = 0; k < selection.partners[v].size(); ++k) {
      CHECK(again.partners[v][k].view == selection.partners[v][k].view);
      CHECK(again.partners[v][k].score == selection.partners[v][k].score);
    }
  }

  r3d::ViewSelectionOptions options;
  options.max_partners = 2;
  CHECK(r3d::select_views(scene, options).partners[5].size() == 2);
}

TEST_CASE("Forward motion is flagged") {
  // Camera 1 moves straight ahead of camera 0: the epipole is the image centre.
  r3d::Scene scene = make_scene({look_at(Eigen::Vector3d(0, 0, -10), Eigen::Vector3d::Zero()),
                                 look_at(Eigen::Vector3d(0, 0, -7), Eigen::Vector3d(0, 0, 1))});
  add_points(scene, Eigen::Vector3d::Zero(), 3.0, 3000, 2);
  r3d::ViewSelectionOptions options;
  options.min_relative_score = 0.0;
  const r3d::ViewSelection selection = r3d::select_views(scene, options);
  REQUIRE(selection.partners[0].size() == 1);
  CHECK(selection.partners[0][0].view == 1);
  CHECK(selection.partners[0][0].epipole_in_image);
}

TEST_CASE("Sparse cloud and camera frustum export") {
  const r3d::Scene scene = ring_scene();
  const r3d::PointCloud cloud = r3d::sparse_point_cloud(scene, false);
  REQUIRE(cloud.size() == scene.num_points());
  CHECK((cloud.world_position(7) - scene.points[7]).norm() < 1e-5);
  CHECK(cloud.visibility[7].size() == scene.track(7).size());
  CHECK(cloud.visibility[7][0] == scene.views[scene.track(7)[0].view].source_id);
  REQUIRE(cloud.find_scalar("track_length") != nullptr);
  CHECK((*cloud.find_scalar("track_length"))[7] == static_cast<float>(scene.track(7).size()));

  const r3d::test::TempDir dir;
  r3d::write_camera_frustums_ply(dir.path() / "cameras.ply", scene, 0.5);
  const r3d::PointCloud frustums = r3d::read_ply(dir.path() / "cameras.ply");  // faces are skipped
  REQUIRE(frustums.size() == 5 * scene.views.size());
  CHECK(frustums.has_colors());
  for (std::size_t v = 0; v < scene.views.size(); ++v) {
    CHECK((frustums.world_position(5 * v) - scene.views[v].pose.center).norm() < 1e-5);  // read back as float
    // The image rectangle's longer side (width) is 0.5 long.
    R3D_CHECK_NEAR((frustums.world_position(5 * v + 2) - frustums.world_position(5 * v + 1)).norm(), 0.5, 1e-5);
  }
}
