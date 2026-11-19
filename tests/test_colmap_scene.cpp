#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <doctest/doctest.h>

#include "r3d/core/error.hpp"
#include "r3d/io/colmap_scene.hpp"
#include "r3d/io/image_io.hpp"
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

struct Camera {
  std::uint32_t id;
  std::string model;
  int model_id;  // binary format
  int width;
  int height;
  std::vector<double> params;
};

struct Point2D {
  double x;
  double y;
  std::int64_t point3d;  // -1: not triangulated
};

struct Image {
  std::uint32_t id;
  std::uint32_t camera;
  Eigen::Quaterniond q;
  Eigen::Vector3d t;
  std::string name;
  std::vector<Point2D> points2d;
};

struct Point3D {
  std::uint64_t id;
  Eigen::Vector3d x;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> track;  // image id, point2D index
};

struct Model {
  std::vector<Camera> cameras;
  std::vector<Image> images;
  std::vector<Point3D> points;
};

const std::vector<double> kOpencv = {610.0, 605.0, 320.5, 240.25, -0.08, 0.02, 0.001, -0.0007};
const std::vector<double> kSimpleRadial = {500.0, 300.0, 200.0, -0.05};
const std::vector<Eigen::Vector3d> kPoints = {{0.1, 0.2, 5.0}, {-0.5, 0.3, 6.0}, {0.4, -0.2, 4.5}};

// Three cameras (one unsupported), three images, three points seen by every
// image plus one point seen only by the image of the unsupported camera.
Model make_model() {
  Model m;
  m.cameras = {{3, "OPENCV", 4, 640, 480, kOpencv},
               {1, "SIMPLE_RADIAL", 2, 600, 400, kSimpleRadial},
               {7, "FOV", 7, 640, 480, {500, 500, 320, 240, 0.9}}};
  m.images = {
      {5, 3, Eigen::Quaterniond(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY())), Eigen::Vector3d(0.3, 0, 0),
       "sub/b image.jpg", {}},
      {2, 1, Eigen::Quaterniond(Eigen::AngleAxisd(-0.15, Eigen::Vector3d::UnitY())), Eigen::Vector3d(-0.4, 0.1, 0.2),
       "a.jpg", {}},
      {9, 7, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero(), "c.jpg", {}},
  };
  for (Image& img : m.images) {
    // POINTS2D in reverse point order (point i at index 2 - i) plus one untriangulated point.
    for (int i = 2; i >= 0; --i) {
      const Eigen::Vector2d x =
          img.camera == 7 ? Eigen::Vector2d(1, 1)
                          : colmap_project(img.camera == 3 ? "OPENCV" : "SIMPLE_RADIAL",
                                           img.camera == 3 ? kOpencv : kSimpleRadial, img.q, img.t, kPoints[i]);
      img.points2d.push_back({x.x(), x.y(), 100 + i});
    }
    img.points2d.push_back({10, 10, -1});
  }
  for (int i = 2; i >= 0; --i) {
    Point3D p{static_cast<std::uint64_t>(100 + i), kPoints[i], {}};
    for (const Image& img : m.images) {
      p.track.emplace_back(img.id, 2 - i);
    }
    m.points.push_back(p);
  }
  m.points.push_back({200, Eigen::Vector3d(0, 0, 1), {{9, 3}}});
  return m;
}

void write_text(const Model& m, const fs::path& dir) {
  fs::create_directories(dir);
  std::ofstream cams(dir / "cameras.txt");
  cams.precision(17);
  cams << "# Camera list with one line of data per camera:\n";
  for (const Camera& c : m.cameras) {
    cams << c.id << ' ' << c.model << ' ' << c.width << ' ' << c.height;
    for (const double v : c.params) cams << ' ' << v;
    cams << '\n';
  }
  std::ofstream imgs(dir / "images.txt");
  imgs.precision(17);
  imgs << "# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
  for (const Image& img : m.images) {
    imgs << img.id << ' ' << img.q.w() << ' ' << img.q.x() << ' ' << img.q.y() << ' ' << img.q.z() << ' '
         << img.t.x() << ' ' << img.t.y() << ' ' << img.t.z() << ' ' << img.camera << ' ' << img.name << '\n';
    for (const Point2D& p : img.points2d) {
      imgs << p.x << ' ' << p.y << ' ' << p.point3d << ' ';
    }
    imgs << '\n';
  }
  std::ofstream pts(dir / "points3D.txt");
  pts.precision(17);
  pts << "# 3D point list\n";
  for (const Point3D& p : m.points) {
    pts << p.id << ' ' << p.x.x() << ' ' << p.x.y() << ' ' << p.x.z() << " 128 128 128 0.5";
    for (const auto& [image, idx] : p.track) pts << ' ' << image << ' ' << idx;
    pts << '\n';
  }
}

class Bytes {
public:
  template <typename T>
  Bytes& put(T v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    data.insert(data.end(), p, p + sizeof(T));
    return *this;
  }
  Bytes& str(const std::string& s) {
    data.insert(data.end(), s.begin(), s.end());
    data.push_back(0);
    return *this;
  }
  std::vector<std::uint8_t> data;
};

void write_binary(const Model& m, const fs::path& dir) {
  fs::create_directories(dir);
  Bytes cams;
  cams.put<std::uint64_t>(m.cameras.size());
  for (const Camera& c : m.cameras) {
    cams.put<std::uint32_t>(c.id).put<std::int32_t>(c.model_id);
    cams.put<std::uint64_t>(static_cast<std::uint64_t>(c.width)).put<std::uint64_t>(static_cast<std::uint64_t>(c.height));
    for (const double v : c.params) cams.put(v);
  }
  r3d::write_file(dir / "cameras.bin", cams.data);
  Bytes imgs;
  imgs.put<std::uint64_t>(m.images.size());
  for (const Image& img : m.images) {
    imgs.put(img.id).put(img.q.w()).put(img.q.x()).put(img.q.y()).put(img.q.z());
    imgs.put(img.t.x()).put(img.t.y()).put(img.t.z()).put(img.camera).str(img.name);
    imgs.put<std::uint64_t>(img.points2d.size());
    for (const Point2D& p : img.points2d) {
      imgs.put(p.x).put(p.y).put(static_cast<std::uint64_t>(p.point3d));  // -1 -> kInvalidPoint3DId
    }
  }
  r3d::write_file(dir / "images.bin", imgs.data);
  Bytes pts;
  pts.put<std::uint64_t>(m.points.size());
  for (const Point3D& p : m.points) {
    pts.put(p.id).put(p.x.x()).put(p.x.y()).put(p.x.z());
    pts.put<std::uint8_t>(128).put<std::uint8_t>(128).put<std::uint8_t>(128).put(0.5);
    pts.put<std::uint64_t>(p.track.size());
    for (const auto& [image, idx] : p.track) pts.put(image).put(idx);
  }
  r3d::write_file(dir / "points3D.bin", pts.data);
}

void touch_images(const fs::path& images_dir) {
  fs::create_directories(images_dir / "sub");
  std::ofstream(images_dir / "a.jpg") << "x";
  std::ofstream(images_dir / "sub" / "b image.jpg") << "x";
  std::ofstream(images_dir / "c.jpg") << "x";
}

void check_scene(const r3d::Scene& scene, const r3d::ColmapImportReport& report, const fs::path& images_dir) {
  const Model m = make_model();
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
  const Eigen::Vector3d center = -(m.images[1].q.toRotationMatrix().transpose() * m.images[1].t);
  CHECK((scene.views[0].pose.center - center).norm() < 1e-12);

  REQUIRE(scene.num_points() == 3);
  CHECK(scene.points[0] == kPoints[0]);  // sorted by point id
  CHECK(scene.track(0).size() == 2);

  // Our camera models + the pixel shift reproduce COLMAP's projections.
  const r3d::ReprojectionStats stats = r3d::compute_reprojection_stats(scene);
  CHECK(stats.num_observations == 6);
  CHECK(stats.max < 1e-3);
}

}  // namespace

TEST_CASE("COLMAP text and binary models read identically") {
  const r3d::test::TempDir dir;
  const fs::path images_dir = dir.path() / "images";
  touch_images(images_dir);
  const Model m = make_model();
  write_text(m, dir.path() / "text" / "sparse" / "0");
  write_binary(m, dir.path() / "binary" / "sparse" / "0");

  r3d::ColmapReadOptions options;
  options.image_dir = images_dir;
  r3d::ColmapImportReport text_report;
  const r3d::Scene text = r3d::read_colmap_scene(dir.path() / "text" / "sparse" / "0", options, &text_report);
  CHECK_FALSE(text_report.binary);
  check_scene(text, text_report, images_dir);

  // A project directory is resolved to sparse/0.
  r3d::ColmapImportReport binary_report;
  const r3d::Scene binary = r3d::read_colmap_scene(dir.path() / "binary", options, &binary_report);
  CHECK(binary_report.binary);
  CHECK(binary_report.model_dir == dir.path() / "binary" / "sparse" / "0");
  check_scene(binary, binary_report, images_dir);

  for (std::size_t v = 0; v < text.views.size(); ++v) {
    CHECK(text.views[v].pose.rotation == binary.views[v].pose.rotation);
    CHECK(text.views[v].pose.center == binary.views[v].pose.center);
  }
  CHECK(text.points == binary.points);
  REQUIRE(text.observations.size() == binary.observations.size());
  for (std::size_t i = 0; i < text.observations.size(); ++i) {
    CHECK(text.observations[i].view == binary.observations[i].view);
    CHECK(text.observations[i].x == binary.observations[i].x);
    CHECK(text.observations[i].y == binary.observations[i].y);
  }

  // Binary is preferred when both formats are present.
  write_text(m, dir.path() / "binary" / "sparse" / "0");
  CHECK(r3d::find_colmap_model(dir.path() / "binary") == dir.path() / "binary" / "sparse" / "0");
  r3d::ColmapImportReport both_report;
  static_cast<void>(r3d::read_colmap_scene(dir.path() / "binary", options, &both_report));
  CHECK(both_report.binary);
}

TEST_CASE("COLMAP reader resolves images and reports errors") {
  const r3d::test::TempDir dir;
  const fs::path model = dir.path() / "sparse" / "0";
  const fs::path images_dir = dir.path() / "images";
  touch_images(images_dir);
  const Model m = make_model();
  write_text(m, model);

  // Default image lookup: <model>/../../images.
  r3d::ColmapImportReport report;
  check_scene(r3d::read_colmap_scene(model, {}, &report), report, images_dir);

  r3d::ColmapReadOptions options;
  options.image_dir = dir.path() / "nowhere";
  CHECK_THROWS_AS(static_cast<void>(r3d::read_colmap_scene(model, options)), r3d::IoError);
  options.require_images = false;
  CHECK(r3d::read_colmap_scene(model, options).views.size() == 2);
  options.load_structure = false;
  CHECK(r3d::read_colmap_scene(model, options).num_points() == 0);
  CHECK_THROWS_AS(static_cast<void>(r3d::read_colmap_scene(dir.path() / "missing")), r3d::IoError);
  {
    std::ofstream(model / "points3D.txt") << "1 0 0 1 0 0 0 0 2 99\n";  // POINT2D_IDX out of range
  }
  CHECK_THROWS_AS(static_cast<void>(r3d::read_colmap_scene(model)), r3d::IoError);

  // Corrupt binary files.
  const fs::path bin = dir.path() / "bin";
  write_binary(m, bin);
  options = {};
  options.image_dir = images_dir;
  CHECK(r3d::read_colmap_scene(bin, options).views.size() == 2);
  std::vector<std::uint8_t> images = r3d::read_file(bin / "images.bin");
  r3d::write_file(bin / "images.bin", std::vector<std::uint8_t>(images.begin(), images.begin() + 100));
  CHECK_THROWS_AS(static_cast<void>(r3d::read_colmap_scene(bin, options)), r3d::IoError);  // truncated
  images.push_back(0);
  r3d::write_file(bin / "images.bin", images);
  CHECK_THROWS_AS(static_cast<void>(r3d::read_colmap_scene(bin, options)), r3d::IoError);  // trailing byte
  images.pop_back();
  r3d::write_file(bin / "images.bin", images);
  r3d::write_file(bin / "points3D.bin", std::vector<std::uint8_t>(16, 0xAB));  // absurd point count
  CHECK_THROWS_AS(static_cast<void>(r3d::read_colmap_scene(bin, options)), r3d::IoError);
  write_binary(m, bin);
  std::vector<std::uint8_t> cameras = r3d::read_file(bin / "cameras.bin");
  const std::int32_t bad_model = 99;
  std::memcpy(cameras.data() + 12, &bad_model, sizeof(bad_model));  // model id of the first camera
  r3d::write_file(bin / "cameras.bin", cameras);
  CHECK_THROWS_AS(static_cast<void>(r3d::read_colmap_scene(bin, options)), r3d::IoError);
}
