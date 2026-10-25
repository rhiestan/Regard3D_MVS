#pragma once

#include <filesystem>

#include "r3d/core/image.hpp"

namespace r3d {

// Portable float map (PFM), as used by the Middlebury and ETH3D benchmarks:
// 1 ("Pf") or 3 ("PF") channels, stored bottom row first. Infinity marks
// unknown disparities by convention.
[[nodiscard]] ImageF32 read_pfm(const std::filesystem::path& path);
// Writes little-endian, 1 or 3 channels.
void write_pfm(const std::filesystem::path& path, ImageView<const float> image);

}  // namespace r3d
