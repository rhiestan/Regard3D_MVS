#include "r3d/io/colmap_scene.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>

#include "r3d/core/error.hpp"
#include "r3d/core/log.hpp"
#include "r3d/io/image_io.hpp"
#include "r3d/io/text_parse.hpp"

namespace r3d {
namespace {

namespace fs = std::filesystem;

// COLMAP pixel coordinates have their origin at the top-left corner of the
// top-left pixel; r3d at its centre.
constexpr double kPixelShift = -0.5;

// Camera model ids and parameter counts of the binary format
// (colmap/sensor/models.h). The count is needed to skip unsupported cameras.
struct ColmapModel {
  int id;
  std::string_view name;
  std::size_t num_params;
};
constexpr std::array<ColmapModel, 12> kColmapModels{{
    {0, "SIMPLE_PINHOLE", 3},
    {1, "PINHOLE", 4},
    {2, "SIMPLE_RADIAL", 4},
    {3, "RADIAL", 5},
    {4, "OPENCV", 8},
    {5, "OPENCV_FISHEYE", 8},
    {6, "FULL_OPENCV", 12},
    {7, "FOV", 5},
    {8, "SIMPLE_RADIAL_FISHEYE", 4},
    {9, "RADIAL_FISHEYE", 5},
    {10, "THIN_PRISM_FISHEYE", 12},
    {11, "RAD_TAN_THIN_PRISM_FISHEYE", 16},
}};

// Format-independent content of a model, in COLMAP conventions.
struct RawCamera {
  std::uint32_t id = 0;
  std::string model;
  int width = 0;
  int height = 0;
  std::vector<double> params;
};

struct RawImage {
  std::uint32_t id = 0;
  std::uint32_t camera_id = 0;
  Pose pose;
  std::string name;
  std::vector<std::array<double, 2>> points2d;
};

struct RawTrackElement {
  std::uint32_t image_id;
  std::uint32_t point2d_idx;
};

struct RawPoint {
  std::uint64_t id = 0;
  Eigen::Vector3d x;
  std::size_t begin = 0;  // range in RawModel::track
  std::size_t end = 0;
};

struct RawModel {
  std::vector<RawCamera> cameras;
  std::vector<RawImage> images;
  std::vector<RawPoint> points;
  std::vector<RawTrackElement> track;
};

// x_cam = R X + t, so C = -R^T t.
Pose colmap_pose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
  Pose pose;
  pose.rotation = q.normalized().toRotationMatrix();
  pose.center = -pose.rotation.transpose() * t;
  return pose;
}

// --- Text format ---------------------------------------------------------------

class LineReader {
public:
  explicit LineReader(const fs::path& path) : path_(path), data_(read_file(path)) {}

  // Next line (without the line break); false at the end of the file.
  bool next(std::string_view& line) {
    if (pos_ >= data_.size()) {
      return false;
    }
    const char* begin = reinterpret_cast<const char*>(data_.data()) + pos_;
    std::size_t len = 0;
    while (pos_ + len < data_.size() && data_[pos_ + len] != '\n') {
      ++len;
    }
    pos_ += len + 1;
    ++line_number_;
    line = std::string_view(begin, len);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    return true;
  }

  // Next line that is neither empty nor a comment.
  bool next_content(std::string_view& line) {
    while (next(line)) {
      const std::string_view t = text::trim(line);
      if (!t.empty() && t.front() != '#') {
        return true;
      }
    }
    return false;
  }

  template <typename T>
  T number(std::string_view token) const {
    const std::optional<T> v = text::parse_number<T>(token);
    if (!v) {
      fail(std::format("invalid number '{}'", token));
    }
    return *v;
  }

  [[noreturn]] void fail(std::string_view what) const {
    throw_io_error("{}:{}: {}", path_.string(), line_number_, what);
  }

private:
  fs::path path_;
  std::vector<std::uint8_t> data_;
  std::size_t pos_ = 0;
  std::size_t line_number_ = 0;
};

// cameras.txt: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
void read_cameras_text(const fs::path& path, RawModel& model) {
  LineReader in(path);
  std::string_view line;
  while (in.next_content(line)) {
    const auto tok = text::split_ws(line);
    if (tok.size() < 4) {
      in.fail("expected CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]");
    }
    RawCamera cam;
    cam.id = in.number<std::uint32_t>(tok[0]);
    cam.model = std::string(tok[1]);
    cam.width = in.number<int>(tok[2]);
    cam.height = in.number<int>(tok[3]);
    for (std::size_t i = 4; i < tok.size(); ++i) {
      cam.params.push_back(in.number<double>(tok[i]));
    }
    model.cameras.push_back(std::move(cam));
  }
}

// images.txt: two lines per image,
//   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
//   POINTS2D[] as (X, Y, POINT3D_ID)
void read_images_text(const fs::path& path, RawModel& model) {
  LineReader in(path);
  std::string_view line;
  while (in.next_content(line)) {
    const auto tok = text::split_ws(line);
    if (tok.size() < 10) {
      in.fail("expected IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME");
    }
    RawImage img;
    img.id = in.number<std::uint32_t>(tok[0]);
    const Eigen::Quaterniond q(in.number<double>(tok[1]), in.number<double>(tok[2]), in.number<double>(tok[3]),
                               in.number<double>(tok[4]));
    const Eigen::Vector3d t(in.number<double>(tok[5]), in.number<double>(tok[6]), in.number<double>(tok[7]));
    img.camera_id = in.number<std::uint32_t>(tok[8]);
    img.pose = colmap_pose(q, t);
    // The name is the rest of the line and may contain spaces.
    img.name = std::string(text::trim(line.substr(static_cast<std::size_t>(tok[9].data() - line.data()))));
    std::string_view points;
    if (in.next(points)) {
      const auto ptok = text::split_ws(points);
      if (ptok.size() % 3 != 0) {
        in.fail("POINTS2D must be triples X Y POINT3D_ID");
      }
      img.points2d.reserve(ptok.size() / 3);
      for (std::size_t i = 0; i < ptok.size(); i += 3) {
        img.points2d.push_back({in.number<double>(ptok[i]), in.number<double>(ptok[i + 1])});
      }
    }
    model.images.push_back(std::move(img));
  }
}

// points3D.txt: POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)
void read_points_text(const fs::path& path, RawModel& model) {
  LineReader in(path);
  std::string_view line;
  while (in.next_content(line)) {
    const auto tok = text::split_ws(line);
    if (tok.size() < 8 || (tok.size() - 8) % 2 != 0) {
      in.fail("expected POINT3D_ID X Y Z R G B ERROR TRACK[]");
    }
    RawPoint p;
    p.id = in.number<std::uint64_t>(tok[0]);
    p.x = Eigen::Vector3d(in.number<double>(tok[1]), in.number<double>(tok[2]), in.number<double>(tok[3]));
    p.begin = model.track.size();
    for (std::size_t i = 8; i < tok.size(); i += 2) {
      model.track.push_back({in.number<std::uint32_t>(tok[i]), in.number<std::uint32_t>(tok[i + 1])});
    }
    p.end = model.track.size();
    model.points.push_back(p);
  }
}

// --- Binary format (little-endian) ----------------------------------------------

static_assert(std::endian::native == std::endian::little, "the COLMAP binary reader assumes a little-endian host");

class BinaryReader {
public:
  explicit BinaryReader(const fs::path& path) : path_(path), data_(read_file(path)) {}

  template <typename T>
  T get() {
    static_assert(std::is_trivially_copyable_v<T>);
    need(sizeof(T));
    T v;
    std::memcpy(&v, data_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return v;
  }

  // Element count, checked against the bytes left so that a corrupt file
  // cannot trigger a huge allocation.
  std::size_t count(std::size_t min_element_size) {
    const auto n = get<std::uint64_t>();
    if (n > (data_.size() - pos_) / min_element_size) {
      fail(std::format("count {} exceeds the file size", n));
    }
    return static_cast<std::size_t>(n);
  }

  // NUL-terminated string.
  std::string read_string() {
    const std::uint8_t* begin = data_.data() + pos_;
    const std::uint8_t* last = data_.data() + data_.size();
    const std::uint8_t* end = std::find(begin, last, std::uint8_t{0});
    if (end == last) {
      fail("unterminated string");
    }
    pos_ += static_cast<std::size_t>(end - begin) + 1;
    return std::string(reinterpret_cast<const char*>(begin), static_cast<std::size_t>(end - begin));
  }

  void expect_end() const {
    if (pos_ != data_.size()) {
      fail(std::format("{} unexpected bytes at the end", data_.size() - pos_));
    }
  }

  [[noreturn]] void fail(std::string_view what) const {
    throw_io_error("{}: offset {}: {}", path_.string(), pos_, what);
  }

private:
  void need(std::size_t n) const {
    if (data_.size() - pos_ < n) {
      fail("unexpected end of file");
    }
  }

  fs::path path_;
  std::vector<std::uint8_t> data_;
  std::size_t pos_ = 0;
};

int checked_size(BinaryReader& in, std::uint64_t v) {
  if (v > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    in.fail(std::format("invalid image size {}", v));
  }
  return static_cast<int>(v);
}

// cameras.bin: uint64 count; per camera: uint32 id, int32 model id,
// uint64 width, uint64 height, double params[count of the model].
void read_cameras_binary(const fs::path& path, RawModel& model) {
  BinaryReader in(path);
  const std::size_t n = in.count(24);
  for (std::size_t i = 0; i < n; ++i) {
    RawCamera cam;
    cam.id = in.get<std::uint32_t>();
    const auto model_id = in.get<std::int32_t>();
    const auto it = std::find_if(kColmapModels.begin(), kColmapModels.end(),
                                 [model_id](const ColmapModel& m) { return m.id == model_id; });
    if (it == kColmapModels.end()) {
      in.fail(std::format("camera {}: unknown camera model id {}", cam.id, model_id));
    }
    cam.model = std::string(it->name);
    cam.width = checked_size(in, in.get<std::uint64_t>());
    cam.height = checked_size(in, in.get<std::uint64_t>());
    for (std::size_t k = 0; k < it->num_params; ++k) {
      cam.params.push_back(in.get<double>());
    }
    model.cameras.push_back(std::move(cam));
  }
  in.expect_end();
}

// images.bin: uint64 count; per image: uint32 id, double qw qx qy qz tx ty tz,
// uint32 camera id, NUL-terminated name, uint64 point count, then per point
// double x, double y, uint64 point3D id.
void read_images_binary(const fs::path& path, RawModel& model) {
  BinaryReader in(path);
  const std::size_t n = in.count(73);
  model.images.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    RawImage img;
    img.id = in.get<std::uint32_t>();
    std::array<double, 7> qt{};
    for (double& v : qt) {
      v = in.get<double>();
    }
    img.pose = colmap_pose(Eigen::Quaterniond(qt[0], qt[1], qt[2], qt[3]), Eigen::Vector3d(qt[4], qt[5], qt[6]));
    img.camera_id = in.get<std::uint32_t>();
    img.name = in.read_string();
    const std::size_t num_points = in.count(24);
    img.points2d.resize(num_points);
    for (auto& p : img.points2d) {
      p[0] = in.get<double>();
      p[1] = in.get<double>();
      static_cast<void>(in.get<std::uint64_t>());
    }
    model.images.push_back(std::move(img));
  }
  in.expect_end();
}

// points3D.bin: uint64 count; per point: uint64 id, double x y z, uint8 r g b,
// double error, uint64 track length, then per element uint32 image id,
// uint32 point2D index.
void read_points_binary(const fs::path& path, RawModel& model) {
  BinaryReader in(path);
  const std::size_t n = in.count(43);
  model.points.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    RawPoint p;
    p.id = in.get<std::uint64_t>();
    p.x.x() = in.get<double>();
    p.x.y() = in.get<double>();
    p.x.z() = in.get<double>();
    for (int c = 0; c < 3; ++c) {
      static_cast<void>(in.get<std::uint8_t>());
    }
    static_cast<void>(in.get<double>());
    const std::size_t length = in.count(8);
    p.begin = model.track.size();
    for (std::size_t k = 0; k < length; ++k) {
      const auto image_id = in.get<std::uint32_t>();
      const auto idx = in.get<std::uint32_t>();
      model.track.push_back({image_id, idx});
    }
    p.end = model.track.size();
    model.points.push_back(p);
  }
  in.expect_end();
}

// --- Scene construction ----------------------------------------------------------

// Maps a COLMAP camera to an r3d camera model, or nullopt if unsupported.
std::optional<CameraIntrinsics> convert_camera(const RawCamera& raw) {
  CameraIntrinsics cam;
  cam.width = raw.width;
  cam.height = raw.height;
  const std::vector<double>& p = raw.params;
  const std::string_view model = raw.model;
  const auto expect = [&p](std::size_t n) { return p.size() == n; };
  const auto set_k = [&cam](double fx, double fy, double cx, double cy) {
    cam.fx = fx;
    cam.fy = fy;
    cam.cx = cx + kPixelShift;
    cam.cy = cy + kPixelShift;
  };
  if (model == "SIMPLE_PINHOLE" && expect(3)) {
    set_k(p[0], p[0], p[1], p[2]);
  } else if (model == "PINHOLE" && expect(4)) {
    set_k(p[0], p[1], p[2], p[3]);
  } else if (model == "SIMPLE_RADIAL" && expect(4)) {
    set_k(p[0], p[0], p[1], p[2]);
    cam.model = CameraModel::radial_k1;
    cam.params = {p[3]};
  } else if (model == "RADIAL" && expect(5)) {
    set_k(p[0], p[0], p[1], p[2]);
    cam.model = CameraModel::radial_k3;
    cam.params = {p[3], p[4], 0.0};
  } else if (model == "OPENCV" && expect(8)) {
    set_k(p[0], p[1], p[2], p[3]);
    cam.model = CameraModel::brown_t2;
    cam.params = {p[4], p[5], 0.0, p[6], p[7]};
  } else if (model == "OPENCV_FISHEYE" && expect(8)) {
    set_k(p[0], p[1], p[2], p[3]);
    cam.model = CameraModel::fisheye;
    cam.params = {p[4], p[5], p[6], p[7]};
  } else if (model == "FULL_OPENCV" && expect(12) && p[9] == 0.0 && p[10] == 0.0 && p[11] == 0.0) {
    set_k(p[0], p[1], p[2], p[3]);
    if (p[6] == 0.0 && p[7] == 0.0) {
      // No tangential terms: OpenMVG's pinhole_radial_k3, as written by openMVG2Colmap.
      cam.model = CameraModel::radial_k3;
      cam.params = {p[4], p[5], p[8]};
    } else {
      cam.model = CameraModel::brown_t2;
      cam.params = {p[4], p[5], p[8], p[6], p[7]};
    }
  } else {
    return std::nullopt;
  }
  return cam;
}

fs::path resolve_image(const std::string& name, const std::vector<fs::path>& dirs, bool require) {
  const fs::path rel(std::u8string(name.begin(), name.end()));
  for (const fs::path& dir : dirs) {
    const fs::path candidate = (dir / rel).lexically_normal();
    std::error_code ec;
    if (fs::is_regular_file(candidate, ec)) {
      return candidate;
    }
  }
  if (require) {
    std::string tried;
    for (const fs::path& dir : dirs) {
      tried += "\n  " + (dir / rel).lexically_normal().string();
    }
    throw_io_error("COLMAP image '{}' not found; tried:{}", name, tried);
  }
  return (dirs.front() / rel).lexically_normal();
}

Scene build_scene(RawModel& raw, const fs::path& model_dir, const ColmapReadOptions& options,
                  ColmapImportReport& rep) {
  Scene scene;
  rep.num_cameras = raw.cameras.size();
  std::sort(raw.cameras.begin(), raw.cameras.end(), [](const RawCamera& a, const RawCamera& b) { return a.id < b.id; });
  std::unordered_map<std::uint32_t, std::uint32_t> camera_index;
  for (const RawCamera& rc : raw.cameras) {
    const std::optional<CameraIntrinsics> cam = convert_camera(rc);
    if (!cam) {
      log_warn("COLMAP camera {}: unsupported model {} with {} parameters, its images are skipped", rc.id, rc.model,
               rc.params.size());
      continue;
    }
    if (!camera_index.emplace(rc.id, static_cast<std::uint32_t>(scene.cameras.size())).second) {
      throw_io_error("{}: duplicate camera id {}", model_dir.string(), rc.id);
    }
    scene.cameras.push_back(*cam);
  }

  rep.num_images = raw.images.size();
  std::sort(raw.images.begin(), raw.images.end(), [](const RawImage& a, const RawImage& b) { return a.id < b.id; });
  std::vector<fs::path> dirs;
  if (!options.image_dir.empty()) {
    dirs.push_back(options.image_dir);
  } else {
    dirs = {model_dir / ".." / ".." / "images", model_dir / ".." / "images", model_dir / "images", model_dir / "..",
            model_dir};
  }
  std::unordered_map<std::uint32_t, std::uint32_t> view_index;  // image id -> view
  std::vector<const RawImage*> view_image;
  for (const RawImage& img : raw.images) {
    const auto cam = camera_index.find(img.camera_id);
    if (cam == camera_index.end()) {
      ++rep.num_images_unsupported;
      continue;
    }
    if (view_index.contains(img.id)) {
      throw_io_error("{}: duplicate image id {}", model_dir.string(), img.id);
    }
    SceneView view;
    view.source_id = img.id;
    view.image_path = resolve_image(img.name, dirs, options.require_images);
    view.width = scene.cameras[cam->second].width;
    view.height = scene.cameras[cam->second].height;
    view.camera = cam->second;
    view.pose = img.pose;
    view_index[img.id] = static_cast<std::uint32_t>(scene.views.size());
    view_image.push_back(&img);
    scene.views.push_back(std::move(view));
  }

  rep.num_points = raw.points.size();
  std::sort(raw.points.begin(), raw.points.end(), [](const RawPoint& a, const RawPoint& b) { return a.id < b.id; });
  std::vector<TrackObservation> track;
  for (const RawPoint& p : raw.points) {
    track.clear();
    for (std::size_t k = p.begin; k < p.end; ++k) {
      const RawTrackElement& e = raw.track[k];
      const auto it = view_index.find(e.image_id);
      if (it == view_index.end()) {
        continue;
      }
      const RawImage& img = *view_image[it->second];
      if (e.point2d_idx >= img.points2d.size()) {
        throw_io_error("{}: point {}: POINT2D_IDX {} out of range for image {}", model_dir.string(), p.id,
                       e.point2d_idx, e.image_id);
      }
      const auto& xy = img.points2d[e.point2d_idx];
      track.push_back({it->second, static_cast<float>(xy[0] + kPixelShift), static_cast<float>(xy[1] + kPixelShift)});
    }
    if (track.empty()) {
      ++rep.num_points_dropped;
      continue;
    }
    std::sort(track.begin(), track.end(),
              [](const TrackObservation& a, const TrackObservation& b) { return a.view < b.view; });
    scene.add_point(p.x, track);
  }
  return scene;
}

bool has_model(const fs::path& dir) {
  std::error_code ec;
  return fs::is_regular_file(dir / "cameras.bin", ec) || fs::is_regular_file(dir / "cameras.txt", ec);
}

}  // namespace

fs::path find_colmap_model(const fs::path& path) {
  for (const fs::path& dir : {path, path / "sparse" / "0", path / "sparse"}) {
    if (has_model(dir)) {
      return dir;
    }
  }
  throw_io_error("{}: no COLMAP model (cameras.bin or cameras.txt) here or in sparse/0 or sparse", path.string());
}

Scene read_colmap_scene(const fs::path& path, const ColmapReadOptions& options, ColmapImportReport* report) {
  ColmapImportReport local_report;
  ColmapImportReport& rep = report != nullptr ? *report : local_report;
  rep = {};
  rep.model_dir = find_colmap_model(path);
  std::error_code ec;
  rep.binary = fs::is_regular_file(rep.model_dir / "cameras.bin", ec);

  RawModel raw;
  if (rep.binary) {
    read_cameras_binary(rep.model_dir / "cameras.bin", raw);
    read_images_binary(rep.model_dir / "images.bin", raw);
    if (options.load_structure) {
      read_points_binary(rep.model_dir / "points3D.bin", raw);
    }
  } else {
    read_cameras_text(rep.model_dir / "cameras.txt", raw);
    read_images_text(rep.model_dir / "images.txt", raw);
    if (options.load_structure) {
      read_points_text(rep.model_dir / "points3D.txt", raw);
    }
  }
  Scene scene = build_scene(raw, rep.model_dir, options, rep);
  scene.validate();
  log_info("COLMAP {} model {}: {} of {} images, {} cameras, {} points, {} observations",
           rep.binary ? "binary" : "text", rep.model_dir.string(), scene.views.size(), rep.num_images,
           scene.cameras.size(), scene.num_points(), scene.observations.size());
  return scene;
}

}  // namespace r3d
