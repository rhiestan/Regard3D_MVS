#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "r3d/core/image.hpp"

namespace r3d {

struct ImageFileInfo {
  int width = 0;
  int height = 0;
  int channels = 0;
  int bit_depth = 8;
};

// Supported for reading: JPEG (baseline/progressive), PNG (8/16 bit), PNM,
// BMP, TGA. Paths may contain any Unicode characters on all platforms.
[[nodiscard]] ImageFileInfo read_image_info(const std::filesystem::path& path);

// desired_channels: 0 = as stored, otherwise 1..4 (converted as needed).
// 16-bit files are reduced to 8 bit.
[[nodiscard]] ImageU8 read_image_u8(const std::filesystem::path& path, int desired_channels = 0);
// 8-bit files are scaled to 16 bit (x * 257).
[[nodiscard]] ImageU16 read_image_u16(const std::filesystem::path& path, int desired_channels = 0);

// Lossless PNG with 1-4 channels. compression_level follows stb's "quality"
// scale: values below 5 behave like 5; higher is smaller and slower.
void write_png(const std::filesystem::path& path, ImageView<const std::uint8_t> image, int compression_level = 8);
void write_png(const std::filesystem::path& path, ImageView<const std::uint16_t> image, int compression_level = 8);

// Whole-file helpers (binary).
[[nodiscard]] std::vector<std::uint8_t> read_file(const std::filesystem::path& path);
void write_file(const std::filesystem::path& path, const std::vector<std::uint8_t>& data);

}  // namespace r3d
