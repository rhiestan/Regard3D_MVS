#include "r3d/io/openmvg_scene.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "r3d/core/error.hpp"
#include "r3d/core/log.hpp"
#include "r3d/io/image_io.hpp"

namespace r3d {
namespace {

namespace fs = std::filesystem;

constexpr std::uint32_t kUndefinedIndex = 0xFFFFFFFFU;  // OpenMVG UndefinedIndexT
// cereal id flags (cereal/details/helpers.hpp): msb marks the first
// occurrence of a pointer or polymorphic type name, msb2 a pointer whose
// dynamic type is the declared base type (no type name follows).
constexpr std::uint32_t kMsb = 0x80000000U;
constexpr std::uint32_t kMsb2 = 0x40000000U;

// ---------------------------------------------------------------------------
// Format-independent raw content of an sfm_data file.

struct RawView {
  std::string local_path;
  std::string filename;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t id_view = kUndefinedIndex;
  std::uint32_t id_intrinsic = kUndefinedIndex;
  std::uint32_t id_pose = kUndefinedIndex;
};

struct RawIntrinsic {
  std::uint32_t key = 0;
  std::string type;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  double focal = 0.0;
  double ppx = 0.0;
  double ppy = 0.0;
  std::vector<double> params;
};

struct RawPose {
  std::uint32_t key = 0;
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
};

struct RawObservation {
  std::uint32_t view_id = 0;
  double x = 0.0;
  double y = 0.0;
};

struct RawLandmark {
  std::uint32_t key = 0;
  Eigen::Vector3d x = Eigen::Vector3d::Zero();
  std::size_t obs_begin = 0;
  std::size_t obs_end = 0;
};

struct RawScene {
  std::string version;
  std::string root_path;
  std::vector<RawView> views;
  std::vector<RawIntrinsic> intrinsics;
  std::vector<RawPose> poses;
  std::vector<RawLandmark> landmarks;
  std::vector<RawObservation> observations;
};

// Polymorphic type names, registered on first occurrence (same id space for
// views and intrinsics, as in cereal).
class PolymorphicNames {
public:
  // Returns the type name for a polymorphic_id: "" for the base type,
  // nullopt for a null pointer.
  template <typename ReadName, typename Fail>
  std::optional<std::string> resolve(std::uint32_t id, ReadName&& read_name, Fail&& fail) {
    if (id == 0) {
      return std::nullopt;
    }
    if (id == kMsb2) {
      return std::string();
    }
    if ((id & kMsb) != 0) {
      std::string name = read_name();
      names_[id & ~kMsb] = name;
      return name;
    }
    const auto it = names_.find(id);
    if (it == names_.end()) {
      fail(std::format("unknown polymorphic type id {}", id));
    }
    return it->second;
  }

private:
  std::unordered_map<std::uint32_t, std::string> names_;
};

// ---------------------------------------------------------------------------
// cereal PortableBinaryInputArchive

class PortableBinaryReader {
public:
  PortableBinaryReader(std::span<const std::uint8_t> data, const fs::path& path) : data_(data), path_(path) {
    // The archive starts with one byte: 1 if the data was written little endian.
    const auto little_endian = read<std::uint8_t>();
    if (little_endian > 1) {
      fail("not a cereal portable binary archive");
    }
    swap_ = (little_endian == 1) != (std::endian::native == std::endian::little);
  }

  template <typename T>
  T read() {
    T value{};
    read_bytes(&value, sizeof(T), sizeof(T));
    return value;
  }

  template <typename T>
  [[nodiscard]] std::optional<T> peek(std::size_t offset = 0) const {
    if (offset + sizeof(T) > remaining()) {
      return std::nullopt;
    }
    T value{};
    std::memcpy(&value, data_.data() + pos_ + offset, sizeof(T));
    if (swap_) {
      swap_bytes(&value, sizeof(T));
    }
    return value;
  }

  // Container size tag; rejects sizes that cannot fit in the rest of the file.
  std::uint64_t read_size(std::size_t min_bytes_per_element) {
    const auto n = read<std::uint64_t>();
    if (n > remaining() / std::max<std::size_t>(min_bytes_per_element, 1)) {
      fail(std::format("implausible container size {}", n));
    }
    return n;
  }

  std::string read_string() {
    const std::uint64_t n = read_size(1);
    std::string s(reinterpret_cast<const char*>(data_.data() + pos_), static_cast<std::size_t>(n));
    pos_ += static_cast<std::size_t>(n);
    return s;
  }

  // std::vector<double> that must have exactly `expected` elements.
  void read_doubles(double* out, std::size_t expected, std::string_view what) {
    const std::uint64_t n = read_size(sizeof(double));
    if (n != expected) {
      fail(std::format("{} has {} values, expected {}", what, n, expected));
    }
    read_bytes(out, expected * sizeof(double), sizeof(double));
  }

  std::vector<double> read_double_vector() {
    const auto n = static_cast<std::size_t>(read_size(sizeof(double)));
    std::vector<double> v(n);
    read_bytes(v.data(), n * sizeof(double), sizeof(double));
    return v;
  }

  std::optional<std::string> read_polymorphic_type() {
    const auto id = read<std::uint32_t>();
    return names_.resolve(
        id, [this] { return read_string(); }, [this](const std::string& msg) { fail(msg); });
  }

  [[nodiscard]] std::size_t remaining() const noexcept { return data_.size() - pos_; }

  [[noreturn]] void fail(std::string_view what) const {
    throw_io_error("{}: {} (at byte {})", path_.string(), what, pos_);
  }

private:
  static void swap_bytes(void* p, std::size_t n) {
    auto* b = static_cast<std::uint8_t*>(p);
    std::reverse(b, b + n);
  }

  void read_bytes(void* out, std::size_t bytes, std::size_t element_size) {
    if (bytes > remaining()) {
      fail("unexpected end of file");
    }
    std::memcpy(out, data_.data() + pos_, bytes);
    pos_ += bytes;
    if (swap_ && element_size > 1) {
      auto* b = static_cast<std::uint8_t*>(out);
      for (std::size_t i = 0; i < bytes; i += element_size) {
        swap_bytes(b + i, element_size);
      }
    }
  }

  std::span<const std::uint8_t> data_;
  const fs::path& path_;
  std::size_t pos_ = 0;
  bool swap_ = false;
  PolymorphicNames names_;
};

void read_binary_view_fields(PortableBinaryReader& in, RawView& v) {
  v.local_path = in.read_string();
  v.filename = in.read_string();
  v.width = in.read<std::uint32_t>();
  v.height = in.read<std::uint32_t>();
  v.id_view = in.read<std::uint32_t>();
  v.id_intrinsic = in.read<std::uint32_t>();
  v.id_pose = in.read<std::uint32_t>();
}

// ViewPriors::save writes the pose-centre prior only if it is enabled
// (bool true, then two std::vector<double> of size 3), and nothing otherwise.
void skip_binary_view_priors(PortableBinaryReader& in) {
  const auto flag = in.peek<std::uint8_t>();
  const auto size = in.peek<std::uint64_t>(1);
  if (flag == std::uint8_t{1} && size == std::uint64_t{3}) {
    static_cast<void>(in.read<std::uint8_t>());
    std::array<double, 3> tmp{};
    in.read_doubles(tmp.data(), 3, "view prior center_weight");
    in.read_doubles(tmp.data(), 3, "view prior center");
  }
}

RawScene parse_binary(const fs::path& path, bool load_structure) {
  const std::vector<std::uint8_t> bytes = read_file(path);
  PortableBinaryReader in(bytes, path);
  RawScene raw;
  raw.version = in.read_string();
  raw.root_path = in.read_string();
  if (raw.version != "0.1" && raw.version != "0.2" && raw.version != "0.3") {
    in.fail(std::format("unsupported sfm_data version '{}'", raw.version));
  }

  // Views: map<IndexT, shared_ptr<View>>. Polymorphic since version 0.3.
  {
    const bool polymorphic = raw.version >= "0.3";
    std::unordered_map<std::uint32_t, RawView> by_pointer;
    const std::uint64_t n = in.read_size(8);
    raw.views.reserve(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
      static_cast<void>(in.read<std::uint32_t>());  // key (== id_view)
      std::string type;
      if (polymorphic) {
        std::optional<std::string> t = in.read_polymorphic_type();
        if (!t) {
          continue;  // null view
        }
        type = std::move(*t);
        if (!type.empty() && type != "view_priors") {
          in.fail(std::format("unsupported view type '{}'", type));
        }
      }
      const auto pointer = in.read<std::uint32_t>();
      if (pointer == 0) {
        continue;
      }
      if ((pointer & kMsb) == 0) {
        const auto it = by_pointer.find(pointer);
        if (it == by_pointer.end()) {
          in.fail("reference to an unknown view");
        }
        raw.views.push_back(it->second);
        continue;
      }
      RawView v;
      read_binary_view_fields(in, v);
      if (type == "view_priors") {
        skip_binary_view_priors(in);
      }
      by_pointer[pointer & ~kMsb] = v;
      raw.views.push_back(std::move(v));
    }
  }

  // Intrinsics: map<IndexT, shared_ptr<IntrinsicBase>>, always polymorphic.
  {
    std::unordered_map<std::uint32_t, RawIntrinsic> by_pointer;
    const std::uint64_t n = in.read_size(8);
    for (std::uint64_t i = 0; i < n; ++i) {
      const auto key = in.read<std::uint32_t>();
      std::optional<std::string> type = in.read_polymorphic_type();
      if (!type) {
        continue;
      }
      const auto pointer = in.read<std::uint32_t>();
      if (pointer == 0) {
        continue;
      }
      if ((pointer & kMsb) == 0) {
        const auto it = by_pointer.find(pointer);
        if (it == by_pointer.end()) {
          in.fail("reference to an unknown intrinsic");
        }
        RawIntrinsic copy = it->second;
        copy.key = key;
        raw.intrinsics.push_back(std::move(copy));
        continue;
      }
      // The binary layout depends on the type, so an unknown type ends the parse.
      const std::optional<CameraModel> model = parse_openmvg_camera_model(*type);
      if (!model) {
        in.fail(std::format("unsupported camera model '{}'", *type));
      }
      RawIntrinsic intr;
      intr.key = key;
      intr.type = std::move(*type);
      intr.width = in.read<std::uint32_t>();
      intr.height = in.read<std::uint32_t>();
      if (*model != CameraModel::spherical) {
        intr.focal = in.read<double>();
        std::array<double, 2> pp{};
        in.read_doubles(pp.data(), 2, "principal_point");
        intr.ppx = pp[0];
        intr.ppy = pp[1];
        if (distortion_parameter_count(*model) > 0) {
          intr.params = in.read_double_vector();
        }
      }
      by_pointer[pointer & ~kMsb] = intr;
      raw.intrinsics.push_back(std::move(intr));
    }
  }

  // Extrinsics: map<IndexT, Pose3> with rotation as 3 rows and the centre.
  {
    const std::uint64_t n = in.read_size(8);
    raw.poses.reserve(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
      RawPose pose;
      pose.key = in.read<std::uint32_t>();
      if (in.read_size(8) != 3) {
        in.fail("rotation must have 3 rows");
      }
      for (int r = 0; r < 3; ++r) {
        std::array<double, 3> row{};
        in.read_doubles(row.data(), 3, "rotation row");
        pose.rotation.row(r) = Eigen::Vector3d(row[0], row[1], row[2]).transpose();
      }
      in.read_doubles(pose.center.data(), 3, "center");
      raw.poses.push_back(pose);
    }
  }

  // Structure: map<IndexT, Landmark{X, map<IndexT, Observation{id_feat, x}>}>.
  // Control points follow; they are not needed and not read.
  if (load_structure) {
    const std::uint64_t n = in.read_size(8);
    raw.landmarks.reserve(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
      RawLandmark lm;
      lm.key = in.read<std::uint32_t>();
      in.read_doubles(lm.x.data(), 3, "landmark X");
      const std::uint64_t num_obs = in.read_size(8);
      lm.obs_begin = raw.observations.size();
      for (std::uint64_t k = 0; k < num_obs; ++k) {
        RawObservation obs;
        obs.view_id = in.read<std::uint32_t>();
        static_cast<void>(in.read<std::uint32_t>());  // id_feat
        std::array<double, 2> x{};
        in.read_doubles(x.data(), 2, "observation x");
        obs.x = x[0];
        obs.y = x[1];
        raw.observations.push_back(obs);
      }
      lm.obs_end = raw.observations.size();
      raw.landmarks.push_back(lm);
    }
  }
  return raw;
}

// ---------------------------------------------------------------------------
// cereal JSONInputArchive

using nlohmann::json;

// The value object of a cereal shared_ptr: {"polymorphic_id", "polymorphic_name"?,
// "ptr_wrapper": {"id", "data"?}}. Returns the data object, or nullptr for a
// null pointer; references to earlier pointers are resolved via `by_pointer`.
const json* resolve_json_pointer(const json& value, bool polymorphic, PolymorphicNames& names, std::string& type,
                                 std::unordered_map<std::uint32_t, const json*>& by_pointer,
                                 std::unordered_map<std::uint32_t, std::string>& type_by_pointer,
                                 const fs::path& path) {
  const auto fail = [&path](const std::string& msg) { throw_io_error("{}: {}", path.string(), msg); };
  type.clear();
  if (polymorphic) {
    const auto id = value.at("polymorphic_id").get<std::uint32_t>();
    std::optional<std::string> t = names.resolve(
        id, [&value] { return value.at("polymorphic_name").get<std::string>(); }, fail);
    if (!t) {
      return nullptr;
    }
    type = std::move(*t);
  }
  const json& wrapper = value.at("ptr_wrapper");
  const auto id = wrapper.at("id").get<std::uint32_t>();
  if (id == 0) {
    return nullptr;
  }
  if ((id & kMsb) == 0) {
    const auto it = by_pointer.find(id);
    if (it == by_pointer.end()) {
      fail("reference to an unknown pointer");
    }
    type = type_by_pointer[id];
    return it->second;
  }
  const json* data = &wrapper.at("data");
  by_pointer[id & ~kMsb] = data;
  type_by_pointer[id & ~kMsb] = type;
  return data;
}

void add_json_landmark(const json& element, RawScene& raw) {
  const json& value = element.at("value");
  RawLandmark lm;
  lm.key = element.at("key").get<std::uint32_t>();
  const json& x = value.at("X");
  lm.x = Eigen::Vector3d(x.at(0).get<double>(), x.at(1).get<double>(), x.at(2).get<double>());
  lm.obs_begin = raw.observations.size();
  for (const json& obs : value.at("observations")) {
    const json& p = obs.at("value").at("x");
    raw.observations.push_back({obs.at("key").get<std::uint32_t>(), p.at(0).get<double>(), p.at(1).get<double>()});
  }
  lm.obs_end = raw.observations.size();
  raw.landmarks.push_back(lm);
}

RawScene parse_json(const fs::path& path, bool load_structure) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw_io_error("cannot open '{}'", path.string());
  }
  RawScene raw;
  try {
    // Landmarks are converted as soon as each one is parsed and then
    // discarded, so a large structure never exists as a JSON DOM.
    int top_level_depth = -1;
    std::string top_level_key;
    const json::parser_callback_t callback = [&](int depth, json::parse_event_t event, json& parsed) {
      if (event == json::parse_event_t::key) {
        if (top_level_depth < 0) {
          top_level_depth = depth;
        }
        if (depth == top_level_depth) {
          top_level_key = parsed.get<std::string>();
        }
        return true;
      }
      const bool in_structure = top_level_key == "structure" || top_level_key == "control_points";
      if (in_structure && event == json::parse_event_t::object_end && depth == top_level_depth + 1) {
        if (load_structure && top_level_key == "structure") {
          add_json_landmark(parsed, raw);
        }
        return false;
      }
      return true;
    };
    const json root = json::parse(file, callback);

    raw.version = root.at("sfm_data_version").get<std::string>();
    raw.root_path = root.at("root_path").get<std::string>();
    if (raw.version != "0.1" && raw.version != "0.2" && raw.version != "0.3") {
      throw_io_error("{}: unsupported sfm_data version '{}'", path.string(), raw.version);
    }

    PolymorphicNames names;
    std::unordered_map<std::uint32_t, const json*> by_pointer;
    std::unordered_map<std::uint32_t, std::string> type_by_pointer;
    std::string type;
    const bool polymorphic_views = raw.version >= "0.3";
    for (const json& element : root.at("views")) {
      const json* data =
          resolve_json_pointer(element.at("value"), polymorphic_views, names, type, by_pointer, type_by_pointer, path);
      if (data == nullptr) {
        continue;
      }
      if (!type.empty() && type != "view_priors") {
        throw_io_error("{}: unsupported view type '{}'", path.string(), type);
      }
      RawView v;
      v.local_path = data->at("local_path").get<std::string>();
      v.filename = data->at("filename").get<std::string>();
      v.width = data->at("width").get<std::uint32_t>();
      v.height = data->at("height").get<std::uint32_t>();
      v.id_view = data->at("id_view").get<std::uint32_t>();
      v.id_intrinsic = data->at("id_intrinsic").get<std::uint32_t>();
      v.id_pose = data->at("id_pose").get<std::uint32_t>();
      raw.views.push_back(std::move(v));
    }

    for (const json& element : root.at("intrinsics")) {
      const json* data = resolve_json_pointer(element.at("value"), true, names, type, by_pointer, type_by_pointer, path);
      if (data == nullptr) {
        continue;
      }
      // Intrinsic_Spherical serialises its base class as a nested object.
      if (!data->contains("width") && data->size() == 1 && data->begin()->is_object()) {
        data = &*data->begin();
      }
      RawIntrinsic intr;
      intr.key = element.at("key").get<std::uint32_t>();
      intr.type = type;
      intr.width = data->at("width").get<std::uint32_t>();
      intr.height = data->at("height").get<std::uint32_t>();
      if (data->contains("focal_length")) {
        intr.focal = data->at("focal_length").get<double>();
        const json& pp = data->at("principal_point");
        intr.ppx = pp.at(0).get<double>();
        intr.ppy = pp.at(1).get<double>();
      }
      for (const char* name : {"disto_k1", "disto_k3", "disto_t2", "fisheye"}) {
        if (data->contains(name)) {
          intr.params = data->at(name).get<std::vector<double>>();
        }
      }
      raw.intrinsics.push_back(std::move(intr));
    }

    for (const json& element : root.at("extrinsics")) {
      RawPose pose;
      pose.key = element.at("key").get<std::uint32_t>();
      const json& value = element.at("value");
      const json& rotation = value.at("rotation");
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          pose.rotation(r, c) = rotation.at(static_cast<std::size_t>(r)).at(static_cast<std::size_t>(c)).get<double>();
        }
      }
      const json& center = value.at("center");
      pose.center = Eigen::Vector3d(center.at(0).get<double>(), center.at(1).get<double>(), center.at(2).get<double>());
      raw.poses.push_back(pose);
    }
  } catch (const json::exception& e) {
    throw_io_error("{}: invalid sfm_data JSON: {}", path.string(), e.what());
  }
  return raw;
}

// ---------------------------------------------------------------------------
// Image path resolution

bool is_valid_utf8(std::string_view s) {
  std::size_t i = 0;
  while (i < s.size()) {
    const auto c = static_cast<unsigned char>(s[i]);
    std::size_t len = 0;
    if (c < 0x80) {
      len = 1;
    } else if ((c >> 5) == 0x6) {
      len = 2;
    } else if ((c >> 4) == 0xE) {
      len = 3;
    } else if ((c >> 3) == 0x1E) {
      len = 4;
    } else {
      return false;
    }
    if (i + len > s.size()) {
      return false;
    }
    for (std::size_t k = 1; k < len; ++k) {
      if ((static_cast<unsigned char>(s[i + k]) >> 6) != 0x2) {
        return false;
      }
    }
    i += len;
  }
  return true;
}

// Paths in sfm_data are UTF-8 on all platforms OpenMVG writes them correctly;
// older Windows tools wrote the ANSI code page, which is the fallback there.
fs::path path_from_file_string(std::string s) {
#if !defined(_WIN32)
  std::replace(s.begin(), s.end(), '\\', '/');
#endif
  if (is_valid_utf8(s)) {
    return fs::path(std::u8string(s.begin(), s.end()));
  }
  return fs::path(s);
}

fs::path resolve_image(const RawView& v, const fs::path& root, const fs::path& sfm_dir,
                       const OpenMvgReadOptions& options) {
  const fs::path local = path_from_file_string(v.local_path);
  const fs::path file = path_from_file_string(v.filename);
  std::vector<fs::path> candidates;
  if (!options.image_dir.empty()) {
    candidates.push_back(options.image_dir / local / file);
    candidates.push_back(options.image_dir / file);
  }
  candidates.push_back(root / local / file);
  candidates.push_back(sfm_dir / local / file);
  candidates.push_back(sfm_dir / file);
  for (fs::path& candidate : candidates) {
    candidate = candidate.lexically_normal();
    std::error_code ec;
    if (fs::is_regular_file(candidate, ec)) {
      return candidate;
    }
  }
  if (options.require_images) {
    std::string tried;
    for (const fs::path& candidate : candidates) {
      tried += "\n  " + candidate.string();
    }
    throw_io_error("image '{}' of view {} not found; tried:{}", file.string(), v.id_view, tried);
  }
  return candidates.front();
}

// ---------------------------------------------------------------------------

Scene build_scene(RawScene& raw, const fs::path& sfm_path, const OpenMvgReadOptions& options,
                  OpenMvgImportReport& report) {
  report.version = raw.version;
  report.root_path = raw.root_path;
  report.num_views = raw.views.size();
  report.num_points = raw.landmarks.size();

  Scene scene;

  // Cameras, ordered by intrinsic id.
  std::sort(raw.intrinsics.begin(), raw.intrinsics.end(),
            [](const RawIntrinsic& a, const RawIntrinsic& b) { return a.key < b.key; });
  std::unordered_map<std::uint32_t, std::uint32_t> camera_index;
  for (const RawIntrinsic& intr : raw.intrinsics) {
    const std::optional<CameraModel> model = parse_openmvg_camera_model(intr.type);
    if (!model) {
      log_warn("intrinsic {}: unsupported camera model '{}', its views are skipped", intr.key, intr.type);
      continue;
    }
    if (*model == CameraModel::spherical) {
      log_warn("intrinsic {}: camera model '{}' is not supported for dense matching, its views are skipped",
               intr.key, intr.type);
      continue;
    }
    const auto num_params = static_cast<std::size_t>(distortion_parameter_count(*model));
    if (intr.params.size() != num_params) {
      throw_io_error("{}: intrinsic {} ({}) has {} distortion parameters, expected {}", sfm_path.string(), intr.key,
                     intr.type, intr.params.size(), num_params);
    }
    CameraIntrinsics cam;
    cam.model = *model;
    cam.width = static_cast<int>(intr.width);
    cam.height = static_cast<int>(intr.height);
    cam.fx = cam.fy = intr.focal;
    cam.cx = intr.ppx;
    cam.cy = intr.ppy;
    std::copy(intr.params.begin(), intr.params.end(), cam.params.begin());
    if (cam.width <= 0 || cam.height <= 0 || !(cam.fx > 0.0)) {
      throw_io_error("{}: intrinsic {} is invalid ({}x{}, focal {})", sfm_path.string(), intr.key, cam.width,
                     cam.height, cam.fx);
    }
    camera_index[intr.key] = static_cast<std::uint32_t>(scene.cameras.size());
    scene.cameras.push_back(cam);
  }

  std::unordered_map<std::uint32_t, const RawPose*> poses;
  for (const RawPose& pose : raw.poses) {
    poses[pose.key] = &pose;
  }

  // Views, ordered by view id; only calibrated ones are kept.
  std::sort(raw.views.begin(), raw.views.end(), [](const RawView& a, const RawView& b) { return a.id_view < b.id_view; });
  const fs::path sfm_dir = sfm_path.parent_path();
  const fs::path root_path = path_from_file_string(raw.root_path);
  const fs::path root = root_path.is_absolute() ? root_path : sfm_dir / root_path;
  std::unordered_map<std::uint32_t, std::uint32_t> view_index;
  for (std::size_t i = 0; i < raw.views.size(); ++i) {
    const RawView& v = raw.views[i];
    if (i > 0 && raw.views[i - 1].id_view == v.id_view) {
      throw_io_error("{}: duplicate view id {}", sfm_path.string(), v.id_view);
    }
    const auto pose = poses.find(v.id_pose);
    if (pose == poses.end()) {
      ++report.num_views_without_pose;
      continue;
    }
    const auto cam = camera_index.find(v.id_intrinsic);
    if (cam == camera_index.end()) {
      ++report.num_views_unsupported;
      continue;
    }
    const CameraIntrinsics& intr = scene.cameras[cam->second];
    if (static_cast<int>(v.width) != intr.width || static_cast<int>(v.height) != intr.height) {
      log_warn("view {} ({}x{}) does not match the size of intrinsic {} ({}x{}), skipped", v.id_view, v.width,
               v.height, v.id_intrinsic, intr.width, intr.height);
      ++report.num_views_unsupported;
      continue;
    }
    SceneView view;
    view.source_id = v.id_view;
    view.image_path = resolve_image(v, root, sfm_dir, options);
    view.width = intr.width;
    view.height = intr.height;
    view.camera = cam->second;
    view.pose.rotation = pose->second->rotation;
    view.pose.center = pose->second->center;
    view_index[v.id_view] = static_cast<std::uint32_t>(scene.views.size());
    scene.views.push_back(std::move(view));
  }

  // Structure, ordered by landmark id, with observations in kept views only.
  std::sort(raw.landmarks.begin(), raw.landmarks.end(),
            [](const RawLandmark& a, const RawLandmark& b) { return a.key < b.key; });
  scene.points.reserve(raw.landmarks.size());
  scene.track_offsets.reserve(raw.landmarks.size() + 1);
  scene.observations.reserve(raw.observations.size());
  std::vector<TrackObservation> track;
  for (const RawLandmark& lm : raw.landmarks) {
    track.clear();
    for (std::size_t k = lm.obs_begin; k < lm.obs_end; ++k) {
      const RawObservation& obs = raw.observations[k];
      const auto it = view_index.find(obs.view_id);
      if (it == view_index.end()) {
        ++report.num_observations_dropped;
        continue;
      }
      track.push_back({it->second, static_cast<float>(obs.x), static_cast<float>(obs.y)});
    }
    if (track.empty()) {
      ++report.num_points_dropped;
      continue;
    }
    std::sort(track.begin(), track.end(),
              [](const TrackObservation& a, const TrackObservation& b) { return a.view < b.view; });
    scene.add_point(lm.x, track);
  }

  scene.validate();
  return scene;
}

}  // namespace

Scene read_openmvg_scene(const fs::path& path, const OpenMvgReadOptions& options, OpenMvgImportReport* report) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  RawScene raw;
  if (ext == ".bin") {
    raw = parse_binary(path, options.load_structure);
  } else if (ext == ".json") {
    raw = parse_json(path, options.load_structure);
  } else {
    throw_io_error("{}: unknown sfm_data format (expected .json or .bin)", path.string());
  }
  OpenMvgImportReport local_report;
  OpenMvgImportReport& rep = report != nullptr ? *report : local_report;
  rep = {};
  Scene scene = build_scene(raw, path, options, rep);
  log_info("{}: {} of {} views calibrated, {} cameras, {} points, {} observations", path.filename().string(),
           scene.views.size(), rep.num_views, scene.cameras.size(), scene.num_points(), scene.observations.size());
  if (rep.num_views_unsupported > 0) {
    log_warn("{} views skipped because of unsupported or inconsistent intrinsics", rep.num_views_unsupported);
  }
  return scene;
}

}  // namespace r3d
