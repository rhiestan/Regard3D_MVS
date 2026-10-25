#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "r3d/core/error.hpp"

namespace r3d {

// Point cloud with optional per-point attributes (structure of arrays).
//
// Positions are float and relative to `origin` (double, world coordinates),
// so georeferenced scenes keep sub-millimetre precision. Each optional array
// is either empty or has exactly size() entries.
struct PointCloud {
  Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3f> positions;
  std::vector<Eigen::Vector3f> normals;
  std::vector<std::array<std::uint8_t, 3>> colors;
  // Named float attributes, e.g. "value" (scale for FSSR) or "confidence".
  std::vector<std::pair<std::string, std::vector<float>>> scalars;
  // Optional ids of the views observing each point.
  std::vector<std::vector<std::uint32_t>> visibility;

  [[nodiscard]] std::size_t size() const noexcept { return positions.size(); }
  [[nodiscard]] bool has_normals() const noexcept { return !normals.empty(); }
  [[nodiscard]] bool has_colors() const noexcept { return !colors.empty(); }
  [[nodiscard]] bool has_visibility() const noexcept { return !visibility.empty(); }

  [[nodiscard]] Eigen::Vector3d world_position(std::size_t i) const {
    return origin + positions[i].cast<double>();
  }

  [[nodiscard]] const std::vector<float>* find_scalar(std::string_view name) const {
    for (const auto& [scalar_name, values] : scalars) {
      if (scalar_name == name) {
        return &values;
      }
    }
    return nullptr;
  }

  // Returns the named attribute, creating it (zero-filled) if absent.
  // Creating an attribute invalidates references to the other attributes.
  std::vector<float>& scalar(std::string_view name) {
    for (auto& [scalar_name, values] : scalars) {
      if (scalar_name == name) {
        return values;
      }
    }
    scalars.emplace_back(std::string(name), std::vector<float>(size(), 0.0f));
    return scalars.back().second;
  }

  // Throws if any attribute array has an inconsistent length.
  void validate() const {
    const std::size_t n = size();
    if (!normals.empty() && normals.size() != n) {
      throw_error("point cloud: {} normals for {} points", normals.size(), n);
    }
    if (!colors.empty() && colors.size() != n) {
      throw_error("point cloud: {} colors for {} points", colors.size(), n);
    }
    if (!visibility.empty() && visibility.size() != n) {
      throw_error("point cloud: {} visibility lists for {} points", visibility.size(), n);
    }
    for (const auto& [name, values] : scalars) {
      if (values.size() != n) {
        throw_error("point cloud: scalar '{}' has {} values for {} points", name, values.size(), n);
      }
    }
  }
};

}  // namespace r3d
