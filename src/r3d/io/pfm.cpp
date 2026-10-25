#include "r3d/io/pfm.hpp"

#include <bit>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <exception>
#include <format>
#include <string>
#include <vector>

#include "r3d/core/error.hpp"
#include "r3d/io/image_io.hpp"

namespace r3d {
namespace {

// Next whitespace-separated header token starting at pos.
std::string next_token(const std::vector<std::uint8_t>& data, std::size_t& pos) {
  while (pos < data.size() && std::isspace(data[pos]) != 0) {
    ++pos;
  }
  std::string token;
  while (pos < data.size() && std::isspace(data[pos]) == 0) {
    token += static_cast<char>(data[pos++]);
  }
  return token;
}

}  // namespace

ImageF32 read_pfm(const std::filesystem::path& path) {
  const std::vector<std::uint8_t> data = read_file(path);
  std::size_t pos = 0;
  const std::string magic = next_token(data, pos);
  int channels = 0;
  if (magic == "Pf") {
    channels = 1;
  } else if (magic == "PF") {
    channels = 3;
  } else {
    throw_error("{}: not a PFM file", path.string());
  }
  int width = 0;
  int height = 0;
  double scale = 0.0;
  try {
    width = std::stoi(next_token(data, pos));
    height = std::stoi(next_token(data, pos));
    scale = std::stod(next_token(data, pos));
  } catch (const std::exception&) {
    throw_error("{}: invalid PFM header", path.string());
  }
  ++pos;  // single whitespace character after the scale
  if (width <= 0 || height <= 0 || scale == 0.0) {
    throw_error("{}: invalid PFM header", path.string());
  }
  const bool little = scale < 0.0;
  const std::size_t row_values = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
  const std::size_t bytes = row_values * static_cast<std::size_t>(height) * sizeof(float);
  if (data.size() < pos + bytes) {
    throw_error("{}: truncated PFM data", path.string());
  }
  ImageF32 img = ImageF32::uninitialized(width, height, channels);
  const bool swap = little != (std::endian::native == std::endian::little);
  for (int y = 0; y < height; ++y) {
    const std::uint8_t* src = data.data() + pos + static_cast<std::size_t>(height - 1 - y) * row_values * sizeof(float);
    float* dst = img.row(y);
    for (std::size_t i = 0; i < row_values; ++i) {
      std::uint32_t u = 0;
      std::memcpy(&u, src + i * sizeof(float), sizeof(u));
      if (swap) {
        u = ((u & 0xFFU) << 24) | ((u & 0xFF00U) << 8) | ((u >> 8) & 0xFF00U) | (u >> 24);
      }
      dst[i] = std::bit_cast<float>(u);
    }
  }
  return img;
}

void write_pfm(const std::filesystem::path& path, ImageView<const float> image) {
  if (image.channels() != 1 && image.channels() != 3) {
    throw_error("write_pfm: {} channels (1 or 3 supported)", image.channels());
  }
  const std::string header =
      std::format("{}\n{} {}\n-1\n", image.channels() == 1 ? "Pf" : "PF", image.width(), image.height());
  const std::size_t row_bytes = static_cast<std::size_t>(image.width()) * static_cast<std::size_t>(image.channels()) *
                                sizeof(float);
  std::vector<std::uint8_t> data(header.begin(), header.end());
  data.reserve(data.size() + row_bytes * static_cast<std::size_t>(image.height()));
  for (int y = image.height() - 1; y >= 0; --y) {
    const auto* row = reinterpret_cast<const std::uint8_t*>(image.row(y));
    if constexpr (std::endian::native == std::endian::little) {
      data.insert(data.end(), row, row + row_bytes);
    } else {
      for (std::size_t i = 0; i < row_bytes; i += 4) {
        data.insert(data.end(), {row[i + 3], row[i + 2], row[i + 1], row[i]});
      }
    }
  }
  write_file(path, data);
}

}  // namespace r3d
