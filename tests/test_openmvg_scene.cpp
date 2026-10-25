#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include "r3d/core/error.hpp"
#include "r3d/io/image_io.hpp"
#include "r3d/io/openmvg_scene.hpp"
#include "test_util.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

constexpr std::uint32_t kUndefined = 0xFFFFFFFFU;

// Synthetic scene, written in the same layout as OpenMVG (cereal JSON and
// portable binary). Ids are deliberately unsorted and partly unusable:
//   view 1: view_priors, intrinsic 1, pose 1
//   view 0: intrinsic 0, pose 0
//   view 3: spherical intrinsic 5 -> skipped
//   view 2: no pose -> skipped
// Landmark 3 is only seen by view 2 and is dropped.
struct SyntheticView {
  std::uint32_t id;
  std::uint32_t intrinsic;
  std::uint32_t pose;
  bool priors;
  std::string filename;
};

const std::vector<SyntheticView> kViews = {
    {1, 1, 1, true, "img_1.jpg"},
    {0, 0, 0, false, "img_0.jpg"},
    {3, 5, 3, false, "img_3.jpg"},
    {2, 0, kUndefined, false, "img_2.jpg"},
};

r3d::CameraIntrinsics synthetic_camera(std::uint32_t key) {
  r3d::CameraIntrinsics cam;
  cam.model = r3d::CameraModel::radial_k3;
  cam.width = 64;
  cam.height = 48;
  cam.fx = cam.fy = key == 0 ? 50.0 : 55.0;
  cam.cx = 31.5;
  cam.cy = 23.25;
  cam.params = {key == 0 ? -0.1 : -0.05, 0.01, 0.002};
  return cam;
}

r3d::Pose synthetic_pose(std::uint32_t key) {
  r3d::Pose pose;
  pose.rotation = Eigen::AngleAxisd(0.1 * key, Eigen::Vector3d(0.2, 1.0, 0.1).normalized()).toRotationMatrix();
  pose.center = Eigen::Vector3d(0.5 * key, -0.1 * key, 0.05 * key);
  return pose;
}

struct SyntheticLandmark {
  std::uint32_t key;
  Eigen::Vector3d x;
  std::vector<std::uint32_t> views;
};

const std::vector<SyntheticLandmark> kLandmarks = {
    {10, {0.2, 0.1, 5.0}, {0, 1}},
    {7, {-0.3, 0.2, 4.0}, {1, 2}},
    {3, {0.0, 0.0, 6.0}, {2}},
};

Eigen::Vector2d observe(const SyntheticLandmark& lm, std::uint32_t view_id) {
  const SyntheticView& v = *std::find_if(kViews.begin(), kViews.end(), [&](const auto& w) { return w.id == view_id; });
  const std::uint32_t pose_key = v.pose == kUndefined ? 0 : v.pose;
  return synthetic_camera(v.intrinsic).project(synthetic_pose(pose_key).world_to_camera(lm.x));
}

class CerealBinaryWriter {
public:
  explicit CerealBinaryWriter(bool little_endian) : swap_(little_endian != (std::endian::native == std::endian::little)) {
    u8(little_endian ? 1 : 0);
  }
  void u8(std::uint8_t v) { bytes_.push_back(v); }
  void u32(std::uint32_t v) { put(v); }
  void u64(std::uint64_t v) { put(v); }
  void f64(double v) { put(v); }
  void str(const std::string& s) {
    u64(s.size());
    bytes_.insert(bytes_.end(), s.begin(), s.end());
  }
  void doubles(const std::vector<double>& v) {
    u64(v.size());
    for (const double d : v) {
      f64(d);
    }
  }
  [[nodiscard]] const std::vector<std::uint8_t>& bytes() const { return bytes_; }

private:
  template <typename T>
  void put(T v) {
    std::uint8_t b[sizeof(T)];
    std::memcpy(b, &v, sizeof(T));
    if (swap_) {
      std::reverse(b, b + sizeof(T));
    }
    bytes_.insert(bytes_.end(), b, b + sizeof(T));
  }
  bool swap_;
  std::vector<std::uint8_t> bytes_;
};

std::vector<std::uint8_t> make_binary(bool little_endian) {
  CerealBinaryWriter w(little_endian);
  w.str("0.3");
  w.str("does/not/exist");
  // Views. Polymorphic ids: "view_priors" is name 1, base views use msb2.
  w.u64(kViews.size());
  std::uint32_t next_pointer = 1;
  for (const SyntheticView& v : kViews) {
    w.u32(v.id);
    if (v.priors) {
      w.u32(0x80000001U);
      w.str("view_priors");
    } else {
      w.u32(0x40000000U);
    }
    w.u32(0x80000000U | next_pointer++);
    w.str("");
    w.str(v.filename);
    w.u32(64);
    w.u32(48);
    w.u32(v.id);
    w.u32(v.intrinsic);
    w.u32(v.pose);
    if (v.priors) {
      w.u8(1);
      w.doubles({1.0, 1.0, 1.0});
      w.doubles({0.1, 0.2, 0.3});
    }
  }
  // Intrinsics: radial_k3 (new name 2), radial_k3 again (name id 2), spherical (new name 3).
  w.u64(3);
  for (const std::uint32_t key : {0U, 1U}) {
    const r3d::CameraIntrinsics cam = synthetic_camera(key);
    w.u32(key);
    if (key == 0) {
      w.u32(0x80000002U);
      w.str("pinhole_radial_k3");
    } else {
      w.u32(2);
    }
    w.u32(0x80000000U | next_pointer++);
    w.u32(64);
    w.u32(48);
    w.f64(cam.fx);
    w.doubles({cam.cx, cam.cy});
    w.doubles({cam.params[0], cam.params[1], cam.params[2]});
  }
  w.u32(5);
  w.u32(0x80000003U);
  w.str("spherical");
  w.u32(0x80000000U | next_pointer++);
  w.u32(64);
  w.u32(32);
  // Extrinsics.
  w.u64(3);
  for (const std::uint32_t key : {3U, 1U, 0U}) {
    const r3d::Pose pose = synthetic_pose(key);
    w.u32(key);
    w.u64(3);
    for (int r = 0; r < 3; ++r) {
      w.doubles({pose.rotation(r, 0), pose.rotation(r, 1), pose.rotation(r, 2)});
    }
    w.doubles({pose.center.x(), pose.center.y(), pose.center.z()});
  }
  // Structure, then an empty control_points map.
  w.u64(kLandmarks.size());
  for (const SyntheticLandmark& lm : kLandmarks) {
    w.u32(lm.key);
    w.doubles({lm.x.x(), lm.x.y(), lm.x.z()});
    w.u64(lm.views.size());
    for (const std::uint32_t view : lm.views) {
      const Eigen::Vector2d p = observe(lm, view);
      w.u32(view);
      w.u32(42);
      w.doubles({p.x(), p.y()});
    }
  }
  w.u64(0);
  return w.bytes();
}

json make_json() {
  json views = json::array();
  std::uint32_t next_pointer = 1;
  bool priors_named = false;
  for (const SyntheticView& v : kViews) {
    json data = {{"local_path", ""}, {"filename", v.filename}, {"width", 64},        {"height", 48},
                 {"id_view", v.id},  {"id_intrinsic", v.intrinsic}, {"id_pose", v.pose}};
    json value;
    if (v.priors) {
      data["use_pose_center_prior"] = true;
      data["center_weight"] = {1.0, 1.0, 1.0};
      data["center"] = {0.1, 0.2, 0.3};
      value["polymorphic_id"] = priors_named ? 1U : 0x80000001U;
      if (!priors_named) {
        value["polymorphic_name"] = "view_priors";
      }
      priors_named = true;
    } else {
      value["polymorphic_id"] = 0x40000000U;
    }
    value["ptr_wrapper"] = {{"id", 0x80000000U | next_pointer++}, {"data", data}};
    views.push_back({{"key", v.id}, {"value", value}});
  }
  json intrinsics = json::array();
  for (const std::uint32_t key : {0U, 1U}) {
    const r3d::CameraIntrinsics cam = synthetic_camera(key);
    json value;
    value["polymorphic_id"] = key == 0 ? 0x80000002U : 2U;
    if (key == 0) {
      value["polymorphic_name"] = "pinhole_radial_k3";
    }
    value["ptr_wrapper"] = {{"id", 0x80000000U | next_pointer++},
                            {"data",
                             {{"width", 64},
                              {"height", 48},
                              {"focal_length", cam.fx},
                              {"principal_point", {cam.cx, cam.cy}},
                              {"disto_k3", {cam.params[0], cam.params[1], cam.params[2]}}}}};
    intrinsics.push_back({{"key", key}, {"value", value}});
  }
  intrinsics.push_back(
      {{"key", 5},
       {"value",
        {{"polymorphic_id", 0x80000003U},
         {"polymorphic_name", "spherical"},
         {"ptr_wrapper", {{"id", 0x80000000U | next_pointer++}, {"data", {{"value0", {{"width", 64}, {"height", 32}}}}}}}}}});
  json extrinsics = json::array();
  for (const std::uint32_t key : {3U, 1U, 0U}) {
    const r3d::Pose pose = synthetic_pose(key);
    json rotation = json::array();
    for (int r = 0; r < 3; ++r) {
      rotation.push_back({pose.rotation(r, 0), pose.rotation(r, 1), pose.rotation(r, 2)});
    }
    extrinsics.push_back(
        {{"key", key}, {"value", {{"rotation", rotation}, {"center", {pose.center.x(), pose.center.y(), pose.center.z()}}}}});
  }
  json structure = json::array();
  for (const SyntheticLandmark& lm : kLandmarks) {
    json observations = json::array();
    for (const std::uint32_t view : lm.views) {
      const Eigen::Vector2d p = observe(lm, view);
      observations.push_back({{"key", view}, {"value", {{"id_feat", 42}, {"x", {p.x(), p.y()}}}}});
    }
    structure.push_back({{"key", lm.key}, {"value", {{"X", {lm.x.x(), lm.x.y(), lm.x.z()}}, {"observations", observations}}}});
  }
  json control_points = json::array();
  control_points.push_back(
      {{"key", 0}, {"value", {{"X", {1.0, 2.0, 3.0}}, {"observations", json::array()}}}});
  return {{"sfm_data_version", "0.3"}, {"root_path", "does/not/exist"}, {"views", views},
          {"intrinsics", intrinsics},  {"extrinsics", extrinsics},       {"structure", structure},
          {"control_points", control_points}};
}

void touch_images(const fs::path& dir) {
  fs::create_directories(dir);
  for (const SyntheticView& v : kViews) {
    std::ofstream(dir / v.filename) << "x";
  }
}

void check_scene(const r3d::Scene& scene, const r3d::OpenMvgImportReport& report, const fs::path& image_dir) {
  CHECK(report.version == "0.3");
  CHECK(report.num_views == 4);
  CHECK(report.num_views_without_pose == 1);
  CHECK(report.num_views_unsupported == 1);
  CHECK(report.num_points == 3);
  CHECK(report.num_points_dropped == 1);
  CHECK(report.num_observations_dropped == 2);

  REQUIRE(scene.cameras.size() == 2);
  for (std::uint32_t key = 0; key < 2; ++key) {
    const r3d::CameraIntrinsics expected = synthetic_camera(key);
    const r3d::CameraIntrinsics& cam = scene.cameras[key];
    CHECK(cam.model == r3d::CameraModel::radial_k3);
    CHECK(cam.width == 64);
    CHECK(cam.fx == expected.fx);
    CHECK(cam.cy == expected.cy);
    CHECK(cam.params == expected.params);
  }

  REQUIRE(scene.views.size() == 2);
  CHECK(scene.views[0].source_id == 0);
  CHECK(scene.views[1].source_id == 1);
  CHECK(scene.views[1].camera == 1);
  CHECK(scene.views[0].image_path == (image_dir / "img_0.jpg").lexically_normal());
  CHECK(scene.views[1].pose.rotation.isApprox(synthetic_pose(1).rotation, 1e-15));
  CHECK(scene.views[1].pose.center == synthetic_pose(1).center);

  REQUIRE(scene.num_points() == 2);
  CHECK(scene.points[0] == kLandmarks[1].x);  // landmark 7 sorts first
  CHECK(scene.track(0).size() == 1);
  CHECK(scene.track(1).size() == 2);
  CHECK(scene.track(1)[0].view == 0);
  CHECK(scene.track(1)[1].view == 1);

  const r3d::ReprojectionStats stats = r3d::compute_reprojection_stats(scene);
  CHECK(stats.num_observations == 3);
  CHECK(stats.rms < 1e-4);  // observations are stored as float
}

}  // namespace

TEST_CASE("OpenMVG binary and JSON scenes read identically") {
  const r3d::test::TempDir dir;
  touch_images(dir.path());
  r3d::write_file(dir.path() / "sfm_data.bin", make_binary(true));
  r3d::write_file(dir.path() / "big_endian.bin", make_binary(false));
  {
    std::ofstream(dir.path() / "sfm_data.json") << make_json().dump(2);
  }
  for (const char* name : {"sfm_data.bin", "big_endian.bin", "sfm_data.json"}) {
    CAPTURE(name);
    r3d::OpenMvgImportReport report;
    const r3d::Scene scene = r3d::read_openmvg_scene(dir.path() / name, {}, &report);
    check_scene(scene, report, dir.path());
  }
}

TEST_CASE("OpenMVG reader resolves images and reports errors") {
  const r3d::test::TempDir dir;
  const fs::path scene_dir = dir.path() / "scene";
  const fs::path image_dir = dir.path() / "images";
  fs::create_directories(scene_dir);
  touch_images(image_dir);
  const fs::path bin = scene_dir / "sfm_data.bin";
  r3d::write_file(bin, make_binary(true));

  // Images are neither under root_path nor next to the file.
  CHECK_THROWS_AS(static_cast<void>(r3d::read_openmvg_scene(bin)), r3d::IoError);
  r3d::OpenMvgReadOptions options;
  options.require_images = false;
  CHECK(r3d::read_openmvg_scene(bin, options).views.size() == 2);
  options.require_images = true;
  options.image_dir = image_dir;
  r3d::OpenMvgImportReport report;
  const r3d::Scene scene = r3d::read_openmvg_scene(bin, options, &report);
  check_scene(scene, report, image_dir);

  options.load_structure = false;
  CHECK(r3d::read_openmvg_scene(bin, options).num_points() == 0);

  // Truncated or corrupt files.
  std::vector<std::uint8_t> bytes = make_binary(true);
  bytes.resize(bytes.size() / 2);
  r3d::write_file(scene_dir / "truncated.bin", bytes);
  CHECK_THROWS_AS(static_cast<void>(r3d::read_openmvg_scene(scene_dir / "truncated.bin", options)), r3d::IoError);
  r3d::write_file(scene_dir / "garbage.bin", std::vector<std::uint8_t>(100, 0xAB));
  CHECK_THROWS_AS(static_cast<void>(r3d::read_openmvg_scene(scene_dir / "garbage.bin", options)), r3d::IoError);
  {
    std::ofstream(scene_dir / "broken.json") << R"({"sfm_data_version": "0.3", "views": [)";
  }
  CHECK_THROWS_AS(static_cast<void>(r3d::read_openmvg_scene(scene_dir / "broken.json", options)), r3d::IoError);
  CHECK_THROWS_AS(static_cast<void>(r3d::read_openmvg_scene(scene_dir / "missing.bin", options)), r3d::IoError);
  CHECK_THROWS_AS(static_cast<void>(r3d::read_openmvg_scene(scene_dir / "scene.txt", options)), r3d::IoError);
}

#if defined(R3D_TESTDATA_DIR)
TEST_CASE("OpenMVG test scenes from Regard3D") {
  const fs::path root = fs::path(R3D_TESTDATA_DIR);
  for (const char* name : {"kermit", "sceaux_castle", "luzern_loewe"}) {
    const fs::path file = root / name / "sfm_data.bin";
    if (!fs::exists(file)) {
      MESSAGE("skipped, not found: ", file.string());
      continue;
    }
    CAPTURE(name);
    r3d::OpenMvgImportReport report;
    const r3d::Scene scene = r3d::read_openmvg_scene(file, {}, &report);
    CHECK(scene.views.size() == report.num_views);
    CHECK(scene.num_points() == report.num_points);
    const r3d::ReprojectionStats stats = r3d::compute_reprojection_stats(scene);
    CHECK(stats.num_behind_camera == 0);
    CHECK(stats.rms < 1.0);
  }
}
#endif
