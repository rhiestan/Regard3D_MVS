#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
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

// Writes a PLY file part by part, for clouds that are not held in memory at
// once. The properties are those of the first non-empty part; every part must
// have the same attributes and origin. The vertex count is written on close()
// (padded with spaces in the header). The destructor closes but loses errors.
class PlyWriter {
 public:
  explicit PlyWriter(const std::filesystem::path& path, const PlyWriteOptions& options = {});
  ~PlyWriter();
  PlyWriter(const PlyWriter&) = delete;
  PlyWriter& operator=(const PlyWriter&) = delete;

  void append(const PointCloud& part);
  void close();
  [[nodiscard]] std::size_t size() const noexcept;  // points written

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend void write_ply(const std::filesystem::path& path, const PointCloud& cloud, const PlyWriteOptions& options);
};

// Reads the "vertex" element of an ascii, binary_little_endian or
// binary_big_endian PLY file; other elements (e.g. faces) are skipped.
// Recognised properties: x y z, nx ny nz, red green blue (alpha ignored),
// list view_ids; every other scalar property becomes a float attribute.
// Without an r3d_origin comment and with double coordinates, the origin is
// set to the rounded first point to preserve precision.
[[nodiscard]] PointCloud read_ply(const std::filesystem::path& path);

}  // namespace r3d
