#include "r3d/io/colmap_scene.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
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

// Maps a COLMAP camera to an r3d camera model, or nullopt if unsupported.
std::optional<CameraIntrinsics> convert_camera(std::string_view model, int width, int height,
                                               const std::vector<double>& p) {
  CameraIntrinsics cam;
  cam.width = width;
  cam.height = height;
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
    cam.model = CameraModel::brown_t2;
    cam.params = {p[4], p[5], p[8], p[6], p[7]};
  } else {
    return std::nullopt;
  }
  return cam;
}

struct ColmapImage {
  std::uint32_t id = 0;
  std::uint32_t camera_id = 0;
  Pose pose;
  std::string name;
  std::vector<std::array<double, 2>> points2d;
};

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

}  // namespace

Scene read_colmap_scene(const fs::path& model_dir, const ColmapReadOptions& options, ColmapImportReport* report) {
  ColmapImportReport local_report;
  ColmapImportReport& rep = report != nullptr ? *report : local_report;
  rep = {};
  Scene scene;

  // cameras.txt: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
  std::unordered_map<std::uint32_t, std::uint32_t> camera_index;
  {
    LineReader in(model_dir / "cameras.txt");
    std::vector<std::pair<std::uint32_t, CameraIntrinsics>> cameras;
    std::string_view line;
    while (in.next_content(line)) {
      const auto tok = text::split_ws(line);
      if (tok.size() < 4) {
        in.fail("expected CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]");
      }
      ++rep.num_cameras;
      const auto id = in.number<std::uint32_t>(tok[0]);
      std::vector<double> params;
      for (std::size_t i = 4; i < tok.size(); ++i) {
        params.push_back(in.number<double>(tok[i]));
      }
      const std::optional<CameraIntrinsics> cam =
          convert_camera(tok[1], in.number<int>(tok[2]), in.number<int>(tok[3]), params);
      if (!cam) {
        log_warn("COLMAP camera {}: unsupported model {} with {} parameters, its images are skipped", id, tok[1],
                 params.size());
        continue;
      }
      cameras.emplace_back(id, *cam);
    }
    std::sort(cameras.begin(), cameras.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [id, cam] : cameras) {
      camera_index[id] = static_cast<std::uint32_t>(scene.cameras.size());
      scene.cameras.push_back(cam);
    }
  }

  // images.txt: two lines per image,
  //   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
  //   POINTS2D[] as (X, Y, POINT3D_ID)
  std::vector<ColmapImage> images;
  {
    LineReader in(model_dir / "images.txt");
    std::string_view line;
    while (in.next_content(line)) {
      const auto tok = text::split_ws(line);
      if (tok.size() < 10) {
        in.fail("expected IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME");
      }
      ColmapImage img;
      img.id = in.number<std::uint32_t>(tok[0]);
      const Eigen::Quaterniond q(in.number<double>(tok[1]), in.number<double>(tok[2]), in.number<double>(tok[3]),
                                 in.number<double>(tok[4]));
      const Eigen::Vector3d t(in.number<double>(tok[5]), in.number<double>(tok[6]), in.number<double>(tok[7]));
      img.camera_id = in.number<std::uint32_t>(tok[8]);
      // x_cam = R X + t, so C = -R^T t.
      img.pose.rotation = q.normalized().toRotationMatrix();
      img.pose.center = -img.pose.rotation.transpose() * t;
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
      images.push_back(std::move(img));
    }
  }
  rep.num_images = images.size();
  std::sort(images.begin(), images.end(), [](const ColmapImage& a, const ColmapImage& b) { return a.id < b.id; });

  std::vector<fs::path> dirs;
  if (!options.image_dir.empty()) {
    dirs.push_back(options.image_dir);
  } else {
    dirs = {model_dir / ".." / ".." / "images", model_dir / ".." / "images", model_dir / "images", model_dir};
  }
  std::unordered_map<std::uint32_t, std::uint32_t> view_index;  // image id -> view
  std::vector<const ColmapImage*> view_image;
  for (const ColmapImage& img : images) {
    const auto cam = camera_index.find(img.camera_id);
    if (cam == camera_index.end()) {
      ++rep.num_images_unsupported;
      continue;
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

  // points3D.txt: POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)
  if (options.load_structure) {
    LineReader in(model_dir / "points3D.txt");
    struct Point {
      std::uint64_t id;
      Eigen::Vector3d x;
      std::size_t begin;
      std::size_t end;
    };
    std::vector<Point> points;
    std::vector<TrackObservation> obs;
    std::string_view line;
    while (in.next_content(line)) {
      const auto tok = text::split_ws(line);
      if (tok.size() < 8 || (tok.size() - 8) % 2 != 0) {
        in.fail("expected POINT3D_ID X Y Z R G B ERROR TRACK[]");
      }
      ++rep.num_points;
      Point p{in.number<std::uint64_t>(tok[0]),
              Eigen::Vector3d(in.number<double>(tok[1]), in.number<double>(tok[2]), in.number<double>(tok[3])),
              obs.size(), 0};
      for (std::size_t i = 8; i < tok.size(); i += 2) {
        const auto image_id = in.number<std::uint32_t>(tok[i]);
        const auto idx = in.number<std::size_t>(tok[i + 1]);
        const auto it = view_index.find(image_id);
        if (it == view_index.end()) {
          continue;
        }
        const ColmapImage& img = *view_image[it->second];
        if (idx >= img.points2d.size()) {
          in.fail(std::format("POINT2D_IDX {} out of range for image {}", idx, image_id));
        }
        obs.push_back({it->second, static_cast<float>(img.points2d[idx][0] + kPixelShift),
                       static_cast<float>(img.points2d[idx][1] + kPixelShift)});
      }
      p.end = obs.size();
      if (p.end == p.begin) {
        ++rep.num_points_dropped;
        continue;
      }
      points.push_back(p);
    }
    std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.id < b.id; });
    std::vector<TrackObservation> track;
    for (const Point& p : points) {
      track.assign(obs.begin() + static_cast<std::ptrdiff_t>(p.begin), obs.begin() + static_cast<std::ptrdiff_t>(p.end));
      std::sort(track.begin(), track.end(),
                [](const TrackObservation& a, const TrackObservation& b) { return a.view < b.view; });
      scene.add_point(p.x, track);
    }
  }

  scene.validate();
  log_info("COLMAP {}: {} of {} images, {} cameras, {} points, {} observations", model_dir.string(),
           scene.views.size(), rep.num_images, scene.cameras.size(), scene.num_points(), scene.observations.size());
  return scene;
}

}  // namespace r3d
