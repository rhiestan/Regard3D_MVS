#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "r3d/core/point_cloud.hpp"

namespace r3d {

enum class PlyFormat { binary_little_endian, ascii };

struct PlyWriteOptions {
  PlyFormat format = PlyFormat::binary_little_endian;
  // Write x/y/z as double in world coordinates (origin added). Otherwise x/y/z
  // are float relative to the origin, which is stored in an
  // "r3d_origin <x> <y> <z>" comment when non-zero.
  bool double_positions = false;
  std::vector<std::string> comments;
};

// Vertex properties written, in order: x y z, [nx ny nz], [red green blue
// (uchar)], one float property per scalar attribute (e.g. "value",
// "confidence"), [list uchar uint view_ids]. At most 255 view ids are written
// per point.
void write_ply(const std::filesystem::path& path, const PointCloud& cloud, const PlyWriteOptions& options = {});

// Reads the "vertex" element of an ascii, binary_little_endian or
// binary_big_endian PLY file; other elements (e.g. faces) are skipped.
// Recognised properties: x y z, nx ny nz, red green blue (alpha ignored),
// list view_ids; every other scalar property becomes a float attribute.
// Without an r3d_origin comment and with double coordinates, the origin is
// set to the rounded first point to preserve precision.
[[nodiscard]] PointCloud read_ply(const std::filesystem::path& path);

}  // namespace r3d
