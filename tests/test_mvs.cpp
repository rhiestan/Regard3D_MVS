#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <doctest/doctest.h>

#include "r3d/core/camera_model.hpp"
#include "r3d/mvs/consistency.hpp"
#include "r3d/mvs/depth_map.hpp"
#include "r3d/mvs/depth_map_store.hpp"
#include "r3d/mvs/fusion.hpp"
#include "r3d/mvs/multi_stereo.hpp"
#include "r3d/mvs/refinement.hpp"
#include "r3d/mvs/visibility.hpp"
#include "r3d/rectify/rectification.hpp"
#include "test_util.hpp"

namespace {

Eigen::Matrix3d rotation(double rx_deg, double ry_deg, double rz_deg) {
  constexpr double kDeg = 3.14159265358979323846 / 180.0;
  return (Eigen::AngleAxisd(rz_deg * kDeg, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(ry_deg * kDeg, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(rx_deg * kDeg, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

r3d::CameraIntrinsics pinhole_camera() {
  r3d::CameraIntrinsics cam;
  cam.width = 320;
  cam.height = 240;
  cam.fx = cam.fy = 280.0;
  cam.cx = 161.2;
  cam.cy = 118.7;
  return cam;
}

// Plane n . x = n . p0, slanted against the base view.
struct Plane {
  Eigen::Vector3d normal;
  Eigen::Vector3d point;
  // Ray parameter t of centre + t * dir on the plane.
  [[nodiscard]] double intersect(const Eigen::Vector3d& centre, const Eigen::Vector3d& dir) const {
    return (point - centre).dot(normal) / dir.dot(normal);
  }
};

Plane test_plane(const r3d::Pose& base) {
  Plane p;
  p.normal = (base.view_direction() + base.rotation.transpose() * Eigen::Vector3d(0.3, -0.2, 0.0)).normalized();
  p.point = base.center + 5.0 * base.view_direction();
  return p;
}

// Exact disparities of the plane for every rectified left pixel.
r3d::DisparityMap plane_disparity(const r3d::RectifiedPair& pair, const Plane& plane, float shift = 0.0F) {
  r3d::DisparityMap disp;
  disp.disparity = r3d::ImageF32(pair.left.intrinsics.width, pair.left.intrinsics.height, 1);
  disp.confidence = r3d::ImageF32(pair.left.intrinsics.width, pair.left.intrinsics.height, 1);
  disp.confidence.fill(0.5F);
  for (int y = 0; y < disp.disparity.height(); ++y) {
    for (int x = 0; x < disp.disparity.width(); ++x) {
      const Eigen::Vector3d dir = pair.left.unproject(Eigen::Vector2d(x, y), 1.0) - pair.left.pose.center;
      const Eigen::Vector3d xw = pair.left.pose.center + plane.intersect(pair.left.pose.center, dir) * dir;
      disp.disparity(x, y) = static_cast<float>(x - pair.right.project(xw).x()) + shift;
    }
  }
  return disp;
}

// Exact depth map of the plane in camera `cam` (pair mask: bit 0, partner `partner`).
r3d::DepthMap plane_depth_map(const r3d::PinholeCamera& cam, const Plane& plane, std::uint32_t view,
                              std::uint32_t partner) {
  r3d::DepthMap map;
  map.view = view;
  map.camera = cam;
  map.allocate(cam.intrinsics.width, cam.intrinsics.height);
  map.partners = {partner};
  for (int y = 0; y < map.height(); ++y) {
    for (int x = 0; x < map.width(); ++x) {
      const Eigen::Vector3d dir = cam.unproject(Eigen::Vector2d(x, y), 1.0) - cam.pose.center;
      map.depth(x, y) = static_cast<float>(plane.intersect(cam.pose.center, dir));
      map.pair_mask(x, y) = 1;
      map.confidence(x, y) = 0.5F;
      map.color(x, y, 0) = 200;
      map.color(x, y, 1) = 100;
      map.color(x, y, 2) = 50;
    }
  }
  return map;
}

struct MultiRig {
  r3d::CameraIntrinsics cam = pinhole_camera();
  r3d::Pose base;
  std::vector<r3d::Pose> partners;
  MultiRig() {
    base.rotation = rotation(2.0, -3.0, 1.0);
    base.center = Eigen::Vector3d(0.1, -0.2, 0.0);
    const std::vector<Eigen::Vector3d> offsets{{0.8, 0.05, 0.1}, {-0.7, 0.1, 0.0}, {0.1, 0.75, -0.1}};
    for (std::size_t i = 0; i < offsets.size(); ++i) {
      r3d::Pose p;
      p.rotation = rotation(-1.0 + static_cast<double>(i), 3.0 - 2.0 * static_cast<double>(i), -2.0);
      p.center = base.center + base.rotation.transpose() * offsets[i];
      partners.push_back(p);
    }
  }
  [[nodiscard]] r3d::PinholeCamera grid() const { return {cam.pinhole(), base}; }
};

}  // namespace

TEST_CASE("scale_intrinsics keeps pixel centres consistent") {
  const r3d::PinholeIntrinsics k = pinhole_camera().pinhole();
  const r3d::PinholeIntrinsics h = r3d::scale_intrinsics(k, 0.5);
  CHECK(h.width == 160);
  CHECK(h.height == 120);
  const Eigen::Vector3d x(0.3, -0.2, 2.0);
  const Eigen::Vector2d p = k.project(x);
  const Eigen::Vector2d q = h.project(x);
  CHECK(q.x() == doctest::Approx((p.x() + 0.5) * 0.5 - 0.5));
  CHECK(q.y() == doctest::Approx((p.y() + 0.5) * 0.5 - 0.5));
}

TEST_CASE("combine_pairs triangulates a slanted plane from several pairs") {
  const MultiRig rig;
  const Plane plane = test_plane(rig.base);
  std::vector<r3d::RectifiedPair> pairs;
  std::vector<r3d::DisparityMap> disps;
  for (const r3d::Pose& p : rig.partners) {
    pairs.push_back(r3d::compute_rectification(rig.cam, rig.base, rig.cam, p, r3d::DepthRange{2.0, 12.0}));
    REQUIRE(pairs.back().valid);
  }
  for (const r3d::RectifiedPair& p : pairs) {
    disps.push_back(plane_disparity(p, plane));
  }
  // The third pair is wrong by 6 pixels everywhere: outvoted where the other two see the plane.
  disps[2] = plane_disparity(pairs[2], plane, 6.0F);
  std::vector<r3d::PairDisparity> inputs;
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    inputs.push_back({&pairs[i], &disps[i], static_cast<std::uint32_t>(10 + i)});
  }
  const r3d::DepthMap map = r3d::combine_pairs(rig.grid(), inputs);
  CHECK(map.partners == std::vector<std::uint32_t>{10, 11, 12});
  std::size_t valid = 0, two_pairs = 0;
  double worst = 0.0;
  for (int y = 0; y < map.height(); ++y) {
    for (int x = 0; x < map.width(); ++x) {
      if (!map.valid(x, y)) {
        continue;
      }
      ++valid;
      if (map.pairs(x, y) < 2) {
        continue;  // seen by one pair only (possibly the wrong one): accepted by design
      }
      CHECK((map.pair_mask(x, y) & 4) == 0);
      ++two_pairs;
      const Eigen::Vector3d dir = map.camera.unproject(Eigen::Vector2d(x, y), 1.0) - map.camera.pose.center;
      worst = std::max(worst, std::abs(map.depth(x, y) / plane.intersect(map.camera.pose.center, dir) - 1.0));
    }
  }
  const double total = static_cast<double>(map.width()) * map.height();
  CHECK(static_cast<double>(valid) > 0.5 * total);
  CHECK(static_cast<double>(two_pairs) > 0.8 * static_cast<double>(valid));
  // Bilinear (and at the border nearest) sampling of a projectively varying disparity.
  CHECK(worst < 1e-3);
}

TEST_CASE("combine_pairs rejects ties, missing pairs and narrow triangulation angles") {
  const MultiRig rig;
  const Plane plane = test_plane(rig.base);
  const r3d::RectifiedPair a = r3d::compute_rectification(rig.cam, rig.base, rig.cam, rig.partners[0], {2.0, 12.0});
  const r3d::RectifiedPair b = r3d::compute_rectification(rig.cam, rig.base, rig.cam, rig.partners[1], {2.0, 12.0});
  REQUIRE(a.valid);
  REQUIRE(b.valid);
  const r3d::DisparityMap da = plane_disparity(a, plane);
  const r3d::DisparityMap db = plane_disparity(b, plane, 5.0F);
  const std::vector<r3d::PairDisparity> two{{&a, &da, 1}, {&b, &db, 2}};

  // One pair against one: no depth (only pixels seen by a single pair remain).
  const std::vector<r3d::PairDisparity> one{{&a, &da, 1}};
  const std::size_t single = r3d::combine_pairs(rig.grid(), one).num_valid();
  const r3d::DepthMap tie = r3d::combine_pairs(rig.grid(), two);
  CHECK(single > 0);
  CHECK(tie.num_valid() < single / 2);
  for (int y = 0; y < tie.height(); ++y) {
    for (int x = 0; x < tie.width(); ++x) {
      if (tie.valid(x, y)) {
        CHECK(tie.pairs(x, y) == 1);
      }
    }
  }

  // A single pair is enough by default, not with min_pairs = 2.
  r3d::MultiStereoOptions opt;
  opt.min_pairs = 2;
  CHECK(r3d::combine_pairs(rig.grid(), one, opt).num_valid() == 0);

  // A short baseline gives triangulation angles below 3 degrees at 5 units.
  r3d::Pose close = rig.partners[0];
  close.center = rig.base.center + rig.base.rotation.transpose() * Eigen::Vector3d(0.15, 0.0, 0.0);
  const r3d::RectifiedPair c = r3d::compute_rectification(rig.cam, rig.base, rig.cam, close, {2.0, 12.0});
  REQUIRE(c.valid);
  const r3d::DisparityMap dc = plane_disparity(c, plane);
  const std::vector<r3d::PairDisparity> narrow{{&c, &dc, 3}};
  CHECK(r3d::combine_pairs(rig.grid(), narrow).num_valid() == 0);
  opt = {};
  opt.min_angle_deg = 1.0;
  CHECK(r3d::combine_pairs(rig.grid(), narrow, opt).num_valid() > 0);
}

TEST_CASE("sample_disparity interpolates within surfaces and not across jumps") {
  r3d::ImageF32 d(4, 2, 1);
  d.fill(10.0F);
  d(2, 0) = 10.5F;
  d(3, 0) = 30.0F;
  d(3, 1) = r3d::kInvalidDisparity;
  CHECK(r3d::sample_disparity(d, 1.5, 0.0) == doctest::Approx(10.25));
  CHECK(r3d::sample_disparity(d, 2.4, 0.3) == doctest::Approx(10.5));  // jump to 30: nearest pixel
  CHECK(r3d::sample_disparity(d, 2.6, 0.6) == r3d::kInvalidDisparity);  // nearest pixel invalid
  CHECK(r3d::sample_disparity(d, -0.6, 0.0) == r3d::kInvalidDisparity);
}

TEST_CASE("Geometric consistency keeps agreeing depths and removes the rest") {
  const MultiRig rig;
  const Plane plane = test_plane(rig.base);
  const r3d::PinholeCamera cam0 = rig.grid();
  const r3d::PinholeCamera cam1{rig.cam.pinhole(), rig.partners[0]};
  const r3d::PinholeCamera cam2{rig.cam.pinhole(), rig.partners[1]};
  std::vector<r3d::DepthMap> maps{plane_depth_map(cam0, plane, 0, 1), plane_depth_map(cam1, plane, 1, 0),
                                  plane_depth_map(cam2, plane, 2, 0)};
  // Map 0: a block 5% too far (outlier) and a pixel without depth.
  for (int y = 100; y < 120; ++y) {
    for (int x = 150; x < 170; ++x) {
      maps[0].depth(x, y) *= 1.05F;
    }
  }
  const std::vector<std::vector<std::size_t>> neighbours{{1, 2}, {0, 2}, {0, 1}};

  SUBCASE("independent support") {
    // Map 0 and map 1 only see each other's pair; map 2 (pair with view 0) is
    // independent evidence for both.
    r3d::ConsistencyOptions opt;
    opt.edge_erosion = 0;
    opt.min_views = 1;
    opt.independent_support = true;
    r3d::ConsistencyStats st;
    r3d::filter_depth_maps(maps, neighbours, opt, &st);
    CHECK(st.input == st.rejected + st.output);
    for (int y = 100; y < 120; ++y) {
      for (int x = 150; x < 170; ++x) {
        CHECK_FALSE(maps[0].valid(x, y));
      }
    }
    CHECK(maps[0].valid(60, 60));
    CHECK(maps[0].support(60, 60) == 1);  // from map 2 only
  }
  SUBCASE("mirrored pairs alone are not enough") {
    r3d::ConsistencyOptions opt;
    opt.edge_erosion = 0;
    opt.min_views = 1;
    opt.independent_support = true;
    std::vector<std::vector<std::size_t>> only_pair{{1}, {0}, {}};
    r3d::filter_depth_maps(maps, only_pair, opt);
    CHECK(maps[0].num_valid() == 0);
    CHECK(maps[1].num_valid() == 0);
  }
  SUBCASE("any support") {
    r3d::ConsistencyOptions opt;
    opt.edge_erosion = 0;
    opt.min_views = 1;
    opt.independent_support = false;
    std::vector<std::vector<std::size_t>> only_pair{{1}, {0}, {}};
    r3d::filter_depth_maps(maps, only_pair, opt);
    CHECK(maps[0].valid(60, 60));
    CHECK(maps[0].support(60, 60) == 1);
  }
  SUBCASE("defaults: two pairs or one agreeing view") {
    // Without neighbours, a depth survives only where two pairs agree on it.
    r3d::ConsistencyOptions opt;
    opt.edge_erosion = 0;
    maps[2].partners = {0, 1};
    maps[2].pair_mask(60, 60) = 3;
    std::vector<std::vector<std::size_t>> none{{}, {}, {}};
    r3d::filter_depth_maps(maps, none, opt);
    CHECK(maps[2].num_valid() == 1);
    CHECK(maps[2].valid(60, 60));
    CHECK(maps[2].support(60, 60) == 0);
  }
}

TEST_CASE("erode_depth_edges removes both sides of a depth jump") {
  r3d::DepthMap map;
  map.allocate(10, 5);
  map.depth.fill(2.0F);
  for (int y = 0; y < 5; ++y) {
    for (int x = 5; x < 10; ++x) {
      map.depth(x, y) = 3.0F;
    }
  }
  CHECK(r3d::erode_depth_edges(map, 1, 0.05) == 10);
  CHECK_FALSE(map.valid(4, 2));
  CHECK_FALSE(map.valid(5, 2));
  CHECK(map.valid(3, 2));
  CHECK(map.valid(6, 2));
  CHECK(r3d::erode_depth_edges(map, 2, 0.05) == 0);  // no jump between valid neighbours left
}

TEST_CASE("erode_depth_edges optionally erodes hole borders") {
  r3d::DepthMap map;
  map.allocate(9, 9);
  map.depth.fill(2.0F);
  map.depth(4, 4) = 0.0F;
  CHECK(r3d::erode_depth_edges(map, 1, 0.05) == 0);
  CHECK(r3d::erode_depth_edges(map, 1, 0.05, true) == 4);  // the 4-neighbours of the hole
  CHECK_FALSE(map.valid(3, 4));
  CHECK(map.valid(3, 3));
  CHECK(map.valid(0, 0));  // the image border is not a hole
}

TEST_CASE("Voxel fusion: points on the surface, oriented normals, one point per voxel") {
  const MultiRig rig;
  const Plane plane = test_plane(rig.base);
  const std::vector<r3d::DepthMap> maps{plane_depth_map(rig.grid(), plane, 0, 1),
                                        plane_depth_map({rig.cam.pinhole(), rig.partners[0]}, plane, 1, 0)};
  r3d::FusionOptions opt;
  opt.voxel_size = 0.05;
  r3d::FusionStats st;
  const Eigen::Vector3d origin(0.1, 0.2, 0.3);
  const r3d::PointCloud cloud = r3d::fuse_depth_maps(maps, origin, opt, &st);
  CHECK(st.input_points == maps[0].num_valid() + maps[1].num_valid());
  REQUIRE(cloud.size() == st.output_points);
  REQUIRE(cloud.has_normals());
  REQUIRE(cloud.has_colors());
  REQUIRE(cloud.find_scalar("value") != nullptr);
  CHECK(st.median_gsd > 0.0);
  // Both cameras are on the same side of the plane: normals point to that side.
  const Eigen::Vector3d facing =
      (rig.base.center - plane.point).dot(plane.normal) > 0.0 ? plane.normal : Eigen::Vector3d(-plane.normal);
  double worst = 0.0, worst_normal = 1.0;
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    worst = std::max(worst, std::abs((cloud.world_position(i) - plane.point).dot(plane.normal)));
    worst_normal = std::min(worst_normal, cloud.normals[i].cast<double>().dot(facing));
    CHECK(cloud.colors[i][0] == 200);
  }
  CHECK(worst < 1e-4);
  CHECK(worst_normal > 0.999);
  // About 15 pixels per 5 cm voxel (GSD about 2 cm, two views).
  CHECK(cloud.size() < st.input_points / 4);
}

TEST_CASE("Depth map file round trip") {
  const MultiRig rig;
  r3d::DepthMap map = plane_depth_map(rig.grid(), test_plane(rig.base), 7, 3);
  map.partners = {3, 9};
  map.pair_mask(5, 6) = 3;
  map.support(7, 8) = 4;
  map.depth(1, 2) = 0.0F;
  const r3d::test::TempDir dir;
  r3d::write_depth_map(dir.path() / "d.r3dd", map);
  const r3d::DepthMap r = r3d::read_depth_map(dir.path() / "d.r3dd");
  CHECK(r.view == 7);
  CHECK(r.partners == map.partners);
  CHECK(r.camera.intrinsics.cx == map.camera.intrinsics.cx);
  CHECK(r.camera.pose.rotation.isApprox(map.camera.pose.rotation));
  CHECK(r.camera.pose.center.isApprox(map.camera.pose.center));
  REQUIRE(r.width() == map.width());
  for (int y = 0; y < map.height(); ++y) {
    for (int x = 0; x < map.width(); ++x) {
      CHECK(r.depth(x, y) == map.depth(x, y));
      CHECK(r.pair_mask(x, y) == map.pair_mask(x, y));
      CHECK(r.support(x, y) == map.support(x, y));
      CHECK(r.color(x, y, 2) == map.color(x, y, 2));
    }
  }
  CHECK(r.pairs(5, 6) == 2);
}

namespace {

// Smooth texture on a plane (wavelengths of about 4 to 25 pixels in the test
// cameras), rendered with 4x4 supersampling.
r3d::ImageF32 render_plane(const r3d::PinholeCamera& cam, const Plane& plane, double phase) {
  const Eigen::Vector3d u = plane.normal.unitOrthogonal();
  const Eigen::Vector3d v = plane.normal.cross(u);
  const std::array<Eigen::Vector3d, 5> waves{Eigen::Vector3d(31.0, 7.0, 0.3), Eigen::Vector3d(-11.0, 23.0, 1.1),
                                             Eigen::Vector3d(5.0, -13.0, 2.0), Eigen::Vector3d(43.0, 29.0, 0.7),
                                             Eigen::Vector3d(-37.0, 47.0, 2.9)};
  r3d::ImageF32 img(cam.intrinsics.width, cam.intrinsics.height, 1);
  for (int y = 0; y < img.height(); ++y) {
    for (int x = 0; x < img.width(); ++x) {
      double acc = 0.0;
      for (int sy = 0; sy < 4; ++sy) {
        for (int sx = 0; sx < 4; ++sx) {
          const Eigen::Vector2d px(x - 0.375 + 0.25 * sx, y - 0.375 + 0.25 * sy);
          const Eigen::Vector3d dir = cam.unproject(px, 1.0) - cam.pose.center;
          const Eigen::Vector3d xw = cam.pose.center + plane.intersect(cam.pose.center, dir) * dir;
          const double pu = (xw - plane.point).dot(u);
          const double pv = (xw - plane.point).dot(v);
          for (const Eigen::Vector3d& wv : waves) {
            acc += std::sin(wv.x() * pu + wv.y() * pv + wv.z() + phase);
          }
        }
      }
      img(x, y) = static_cast<float>(128.0 + 20.0 * acc / 16.0);
    }
  }
  return img;
}

double relative_depth_rms(const r3d::DepthMap& map, const r3d::DepthMap& truth, int border) {
  double sum = 0.0;
  int n = 0;
  for (int y = border; y < map.height() - border; ++y) {
    for (int x = border; x < map.width() - border; ++x) {
      const double e = (map.depth(x, y) - truth.depth(x, y)) / truth.depth(x, y);
      sum += e * e;
      ++n;
    }
  }
  return std::sqrt(sum / n);
}

}  // namespace

TEST_CASE("Refinement fits slanted planes from stair-stepped depths") {
  const MultiRig rig;
  Plane plane;
  // The base view is the central 160 x 120 crop of the rig's grid, which
  // keeps the test fast in sanitizer builds.
  r3d::PinholeCamera cam0 = rig.grid();
  cam0.intrinsics.width = 160;
  cam0.intrinsics.height = 120;
  cam0.intrinsics.cx -= 80.0;
  cam0.intrinsics.cy -= 60.0;
  plane.normal = (rig.base.view_direction() + rig.base.rotation.transpose() * Eigen::Vector3d(0.8, 0.4, 0.0))
                     .normalized();  // 42 degrees
  plane.point = rig.base.center + 5.0 * rig.base.view_direction();
  const r3d::DepthMap truth = plane_depth_map(cam0, plane, 0, 1);
  // Rendered once: doctest runs this setup again for every subcase.
  static const std::vector<r3d::RefinementImage> images = [&] {
    std::vector<r3d::RefinementImage> result;
    result.push_back(r3d::make_refinement_image(cam0, render_plane(cam0, plane, 0.0)));
    for (const r3d::Pose& p : rig.partners) {
      const r3d::PinholeCamera cam{rig.cam.pinhole(), p};
      result.push_back(r3d::make_refinement_image(cam, render_plane(cam, plane, 0.0)));
    }
    return result;
  }();
  const std::vector<const r3d::RefinementImage*> views{&images[1], &images[2], &images[3]};

  // SGM-like input: constant depth per 4x4 block (fronto-parallel steps) plus a
  // smooth error of up to 1% (about 0.4 px of disparity).
  r3d::DepthMap map = truth;
  for (int y = 0; y < map.height(); ++y) {
    for (int x = 0; x < map.width(); ++x) {
      const float block = truth.depth(std::min(x / 4 * 4 + 2, map.width() - 1), std::min(y / 4 * 4 + 2, map.height() - 1));
      map.depth(x, y) = block * static_cast<float>(1.0 + 0.01 * std::sin(x / 9.0) * std::cos(y / 13.0));
    }
  }
  const int border = 12;
  const double before = relative_depth_rms(map, truth, border);

  SUBCASE("defaults: only clear NCC gains are taken") {
    r3d::RefinementStats st;
    r3d::refine_depth_map(map, images[0], views, {}, &st);
    CHECK(relative_depth_rms(map, truth, border) < before / 3.0);
    CHECK(st.refined > st.input / 2);
  }
  SUBCASE("matching images") {
    r3d::RefinementOptions opt;
    opt.min_gain = 0.0;
    r3d::RefinementStats st;
    r3d::refine_depth_map(map, images[0], views, opt, &st);
    const double after = relative_depth_rms(map, truth, border);
    MESSAGE("relative depth RMS ", before, " -> ", after, ", refined ", st.refined, " of ", st.input);
    CHECK(st.input == static_cast<std::size_t>(map.width() * map.height()));
    CHECK(st.refined > st.input * 8 / 10);
    CHECK(after < before / 5.0);
    CHECK(map.confidence(80, 60) > 0.9F);
  }
  SUBCASE("unrelated images keep the input") {
    std::vector<r3d::RefinementImage> wrong;
    for (const r3d::Pose& p : rig.partners) {
      const r3d::PinholeCamera cam{rig.cam.pinhole(), p};
      wrong.push_back(r3d::make_refinement_image(cam, render_plane(cam, plane, 1.7)));
    }
    const std::vector<const r3d::RefinementImage*> wrong_views{&wrong[0], &wrong[1], &wrong[2]};
    r3d::RefinementOptions opt;
    opt.min_ncc = 0.5;
    r3d::RefinementStats st;
    r3d::refine_depth_map(map, images[0], wrong_views, opt, &st);
    CHECK(st.refined < st.input / 20);
    CHECK(relative_depth_rms(map, truth, border) < before * 1.2);
  }
}

TEST_CASE("remove_depth_speckles removes small segments") {
  r3d::DepthMap map;
  map.allocate(20, 20);
  map.depth.fill(2.0F);
  for (int y = 2; y < 5; ++y) {
    for (int x = 2; x < 5; ++x) {
      map.depth(x, y) = 3.0F;  // separate segment of 9 pixels
    }
  }
  for (int y = 10; y < 14; ++y) {
    for (int x = 10; x < 14; ++x) {
      map.depth(x, y) = 2.02F;  // 1% step: part of the large segment
    }
  }
  CHECK(r3d::remove_depth_speckles(map, 10, 0.02) == 9);
  CHECK_FALSE(map.valid(3, 3));
  CHECK(map.valid(11, 11));
  CHECK(map.num_valid() == 400 - 9);
  CHECK(r3d::remove_depth_speckles(map, 10, 0.02) == 0);
}

TEST_CASE("Visibility filter removes depths behind surfaces and in free space") {
  const MultiRig rig;
  const Plane plane = test_plane(rig.base);
  std::vector<r3d::DepthMap> maps{plane_depth_map(rig.grid(), plane, 0, 1)};
  for (std::size_t i = 0; i < rig.partners.size(); ++i) {
    maps.push_back(plane_depth_map({rig.cam.pinhole(), rig.partners[i]}, plane, static_cast<std::uint32_t>(i + 1), 0));
  }
  for (r3d::DepthMap& m : maps) {
    m.support.fill(3);
  }
  // Base view: a patch behind the plane (a reflection) and one in front of it, each confirmed by one view.
  for (int y = 100; y < 110; ++y) {
    for (int x = 100; x < 110; ++x) {
      maps[0].depth(x, y) *= 1.3F;
      maps[0].depth(x + 60, y) *= 0.7F;
      maps[0].support(x, y) = 1;
      maps[0].support(x + 60, y) = 1;
    }
  }
  std::vector<std::vector<std::size_t>> neighbours{{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
  const std::size_t valid = maps[0].num_valid();
  r3d::VisibilityStats st;
  r3d::filter_visibility(maps, neighbours, {}, &st);
  CHECK(st.removed == 200);
  CHECK(maps[0].num_valid() == valid - 200);
  CHECK_FALSE(maps[0].valid(105, 105));
  CHECK_FALSE(maps[0].valid(165, 105));
  CHECK(maps[0].valid(130, 105));
  CHECK(st.occluded > 0);
  CHECK(st.free_space > 0);
  for (std::size_t i = 1; i < maps.size(); ++i) {
    CHECK(maps[i].num_valid() == static_cast<std::size_t>(maps[i].width() * maps[i].height()));
  }
}

TEST_CASE("Octree fusion: finer views win, fold validation, passes") {
  const MultiRig rig;
  const Plane plane = test_plane(rig.base);
  const Eigen::Vector3d origin(0.1, 0.2, 0.3);
  SUBCASE("the result does not depend on the number of passes") {
    const std::vector<r3d::DepthMap> maps{plane_depth_map(rig.grid(), plane, 0, 1),
                                          plane_depth_map({rig.cam.pinhole(), rig.partners[0]}, plane, 1, 0)};
    r3d::FusionStats one;
    const r3d::PointCloud a = r3d::fuse_depth_maps(maps, origin, {}, &one);
    r3d::FusionOptions small;
    small.memory_budget = std::size_t{40} * 20000;
    r3d::FusionStats many;
    const r3d::PointCloud b = r3d::fuse_depth_maps(maps, origin, small, &many);
    CHECK(one.passes == 1);
    CHECK(many.passes > 3);
    REQUIRE(a.size() == b.size());
    bool same = true;
    for (std::size_t i = 0; i < a.size(); ++i) {
      same = same && a.positions[i] == b.positions[i] && a.normals[i] == b.normals[i];
    }
    CHECK(same);
    CHECK(one.input_points == maps[0].num_valid() + maps[1].num_valid());
    CHECK(one.levels >= 1);
  }
  SUBCASE("points of a view three times farther away are dropped where the near view sees the plane") {
    r3d::Pose far = rig.base;
    far.center = rig.base.center - 10.0 * rig.base.view_direction();
    const std::vector<r3d::DepthMap> maps{plane_depth_map(rig.grid(), plane, 0, 1),
                                          plane_depth_map({rig.cam.pinhole(), far}, plane, 1, 0)};
    // The near view covers about a ninth of the far view's footprint.
    const std::size_t overlap = maps[1].num_valid() / 9;
    r3d::FusionStats st;
    const r3d::PointCloud cloud = r3d::fuse_depth_maps(maps, origin, {}, &st);
    MESSAGE("coarser ", st.coarser_points, " of about ", overlap, ", levels ", st.levels);
    CHECK(st.coarser_points > overlap * 7 / 10);
    CHECK(st.coarser_points < overlap * 13 / 10);
    double worst = 0.0;
    for (std::size_t i = 0; i < cloud.size(); ++i) {
      worst = std::max(worst, std::abs((cloud.world_position(i) - plane.point).dot(plane.normal)));
    }
    CHECK(worst < 1e-4);
    r3d::FusionOptions all;
    all.resolution_ratio = 0.0;
    CHECK(r3d::fuse_depth_maps(maps, origin, all, &st).size() > cloud.size());
    CHECK(st.coarser_points == 0);
  }
  SUBCASE("fold validation keeps points that two views see") {
    const r3d::PinholeCamera partner{rig.cam.pinhole(), rig.partners[0]};
    const std::vector<r3d::DepthMap> maps{plane_depth_map(rig.grid(), plane, 0, 1), plane_depth_map(partner, plane, 1, 0)};
    r3d::FusionOptions opt;
    opt.voxel_factor = 2.0;
    opt.min_sources = 2;
    opt.fold_neighbourhood = false;
    opt.with_views = true;
    r3d::FusionStats st;
    const r3d::PointCloud cloud = r3d::fuse_depth_maps(maps, origin, opt, &st);
    CHECK(st.unsupported_points > 0);
    REQUIRE(cloud.visibility.size() == cloud.size());
    bool both = true;
    for (const std::vector<std::uint32_t>& views : cloud.visibility) {
      both = both && views == std::vector<std::uint32_t>{0, 1};
    }
    CHECK(both);
    // With the neighbouring leaves, points next to the other view's footprint are kept as well.
    opt.fold_neighbourhood = true;
    r3d::FusionStats wide;
    CHECK(r3d::fuse_depth_maps(maps, origin, opt, &wide).size() > cloud.size());
    CHECK(wide.unsupported_points < st.unsupported_points);
    REQUIRE(cloud.size() > 1000);
    std::size_t outside = 0;
    for (std::size_t i = 0; i < cloud.size(); ++i) {
      for (const r3d::PinholeCamera* cam : {&maps[0].camera, &partner}) {
        const Eigen::Vector2d q = cam->project(cloud.world_position(i));
        outside += (q.x() < -2.0 || q.y() < -2.0 || q.x() > cam->intrinsics.width + 1.0 ||
                    q.y() > cam->intrinsics.height + 1.0)
                       ? 1
                       : 0;
      }
    }
    CHECK(outside == 0);
  }
}

TEST_CASE("DepthMapStore keeps maps within the memory budget") {
  const MultiRig rig;
  const Plane plane = test_plane(rig.base);
  const r3d::test::TempDir dir;
  auto path = [&](std::size_t i) { return dir.path() / ("m" + std::to_string(i) + ".r3dd"); };
  auto make = [&](std::size_t i) {
    r3d::DepthMap m = plane_depth_map(rig.grid(), plane, static_cast<std::uint32_t>(i), 0);
    m.depth(10, 10) = static_cast<float>(i + 1);
    return m;
  };
  const std::size_t map_bytes = r3d::depth_map_bytes(make(0));
  SUBCASE("spilled maps come back from their files, which are removed with the store") {
    {
      r3d::DepthMapStore store(5, path, map_bytes * 5 / 2, false);
      for (std::size_t i = 0; i < 5; ++i) {
        store.put(i, make(i));
        CHECK(store.resident_bytes() <= map_bytes * 5 / 2);
      }
      CHECK(std::filesystem::exists(path(0)));     // evicted
      CHECK_FALSE(std::filesystem::exists(path(4)));  // still in memory
      for (std::size_t i = 0; i < 5; ++i) {
        const auto m = store.get(i);
        CHECK(m->view == i);
        CHECK(m->depth(10, 10) == static_cast<float>(i + 1));
        CHECK(store.resident_bytes() <= map_bytes * 5 / 2);
      }
      CHECK(store.contains(3));
    }
    for (std::size_t i = 0; i < 5; ++i) {
      CHECK_FALSE(std::filesystem::exists(path(i)));
    }
  }
  SUBCASE("persistent stores write every map and can adopt existing files") {
    {
      r3d::DepthMapStore store(3, path, std::size_t{1} << 30, true);
      for (std::size_t i = 0; i < 3; ++i) {
        store.put(i, make(i));
      }
      store.put(1, r3d::DepthMap{});  // a view without depths
      CHECK(store.get(1)->width() == 0);
    }
    CHECK(std::filesystem::exists(path(0)));
    CHECK(std::filesystem::exists(path(2)));
    r3d::DepthMapStore again(3, path, 0, true);
    CHECK_FALSE(again.contains(2));
    CHECK(again.get(2)->width() == 0);
    again.adopt_file(2);
    CHECK(again.contains(2));
    CHECK(again.get(2)->depth(10, 10) == 3.0F);
  }
}
