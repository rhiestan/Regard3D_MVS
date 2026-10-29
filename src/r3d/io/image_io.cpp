#include "r3d/io/image_io.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <string>

#include "r3d/core/error.hpp"
#include "r3d/io/stb_wrapper.hpp"

namespace r3d {
namespace {

std::string path_string(const std::filesystem::path& path) {
  const std::u8string u8 = path.u8string();
  return {u8.begin(), u8.end()};
}

template <typename T>
Image<T> decode_into_image(const std::filesystem::path& path, int desired_channels) {
  if (desired_channels < 0 || desired_channels > 4) {
    throw_error("invalid desired channel count {}", desired_channels);
  }
  const std::vector<std::uint8_t> encoded = read_file(path);
  stb::Decoded decoded;
  try {
    decoded = stb::decode(encoded, desired_channels, sizeof(T) == 2);
  } catch (const IoError& e) {
    throw_io_error("{}: {}", path_string(path), e.what());
  }
  auto image = Image<T>::uninitialized(decoded.width, decoded.height, decoded.channels);
  const auto* src = static_cast<const T*>(decoded.pixels.get());
  const std::size_t row_elems = static_cast<std::size_t>(decoded.width) * static_cast<std::size_t>(decoded.channels);
  for (int y = 0; y < decoded.height; ++y) {
    std::memcpy(image.row(y), src + static_cast<std::size_t>(y) * row_elems, row_elems * sizeof(T));
  }
  return image;
}

// --- PNG encoder (8/16 bit, adaptive per-row filtering) ----------------------

std::uint32_t crc32_update(std::uint32_t crc, std::span<const std::uint8_t> data) {
  static const std::array<std::uint32_t, 256> table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t n = 0; n < 256; ++n) {
      std::uint32_t c = n;
      for (int k = 0; k < 8; ++k) {
        c = (c & 1u) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      }
      t[n] = c;
    }
    return t;
  }();
  for (const std::uint8_t byte : data) {
    crc = table[(crc ^ byte) & 0xFFu] ^ (crc >> 8);
  }
  return crc;
}

void append_be32(std::vector<std::uint8_t>& out, std::uint32_t v) {
  out.push_back(static_cast<std::uint8_t>(v >> 24));
  out.push_back(static_cast<std::uint8_t>(v >> 16));
  out.push_back(static_cast<std::uint8_t>(v >> 8));
  out.push_back(static_cast<std::uint8_t>(v));
}

void append_chunk(std::vector<std::uint8_t>& out, const char (&type)[5], std::span<const std::uint8_t> payload) {
  append_be32(out, static_cast<std::uint32_t>(payload.size()));
  const std::size_t type_pos = out.size();
  out.insert(out.end(), type, type + 4);
  out.insert(out.end(), payload.begin(), payload.end());
  const std::uint32_t crc =
      crc32_update(0xFFFFFFFFu, std::span<const std::uint8_t>(out.data() + type_pos, 4 + payload.size()));
  append_be32(out, crc ^ 0xFFFFFFFFu);
}

std::uint8_t paeth(int a, int b, int c) {
  const int p = a + b - c;
  const int pa = std::abs(p - a);
  const int pb = std::abs(p - b);
  const int pc = std::abs(p - c);
  if (pa <= pb && pa <= pc) {
    return static_cast<std::uint8_t>(a);
  }
  return static_cast<std::uint8_t>(pb <= pc ? b : c);
}

// Applies PNG filter `type` to `row` (prev may be all zeros for the first row).
void filter_row(int type, const std::uint8_t* row, const std::uint8_t* prev, std::size_t len, std::size_t bpp,
                std::uint8_t* out) {
  for (std::size_t i = 0; i < len; ++i) {
    const int x = row[i];
    const int a = i >= bpp ? row[i - bpp] : 0;
    const int b = prev[i];
    const int c = i >= bpp ? prev[i - bpp] : 0;
    int predicted = 0;
    switch (type) {
      case 0: predicted = 0; break;
      case 1: predicted = a; break;
      case 2: predicted = b; break;
      case 3: predicted = (a + b) / 2; break;
      default: predicted = paeth(a, b, c); break;
    }
    out[i] = static_cast<std::uint8_t>(x - predicted);
  }
}

// Encodes big-endian sample bytes (`rows` holds height rows of `row_bytes`).
std::vector<std::uint8_t> encode_png(const std::vector<std::uint8_t>& rows, int width, int height, int channels,
                                     int bit_depth, int level) {
  static constexpr std::array<std::uint8_t, 5> kColorType = {0, 0, 4, 2, 6};  // by channel count
  const std::size_t bpp = static_cast<std::size_t>(channels) * static_cast<std::size_t>(bit_depth / 8);
  const std::size_t row_bytes = static_cast<std::size_t>(width) * bpp;

  // Filtered stream: one filter byte + row per row. The filter is chosen per
  // row by the minimum sum of absolute (signed) residuals, as libpng does.
  std::vector<std::uint8_t> filtered((row_bytes + 1) * static_cast<std::size_t>(height));
  const std::vector<std::uint8_t> zero_row(row_bytes, 0);
  std::vector<std::uint8_t> candidate(row_bytes);
  for (int y = 0; y < height; ++y) {
    const std::uint8_t* row = rows.data() + static_cast<std::size_t>(y) * row_bytes;
    const std::uint8_t* prev = y > 0 ? row - row_bytes : zero_row.data();
    std::uint8_t* out = filtered.data() + static_cast<std::size_t>(y) * (row_bytes + 1);
    long long best_score = -1;
    for (int type = 0; type < 5; ++type) {
      filter_row(type, row, prev, row_bytes, bpp, candidate.data());
      long long score = 0;
      for (const std::uint8_t v : candidate) {
        score += std::abs(static_cast<int>(static_cast<std::int8_t>(v)));
      }
      if (best_score < 0 || score < best_score) {
        best_score = score;
        out[0] = static_cast<std::uint8_t>(type);
        std::memcpy(out + 1, candidate.data(), row_bytes);
      }
    }
  }

  std::vector<std::uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
  std::vector<std::uint8_t> ihdr;
  append_be32(ihdr, static_cast<std::uint32_t>(width));
  append_be32(ihdr, static_cast<std::uint32_t>(height));
  ihdr.push_back(static_cast<std::uint8_t>(bit_depth));
  ihdr.push_back(kColorType[static_cast<std::size_t>(channels)]);
  ihdr.push_back(0);  // compression: deflate
  ihdr.push_back(0);  // filter method 0
  ihdr.push_back(0);  // no interlace
  append_chunk(png, "IHDR", ihdr);
  append_chunk(png, "IDAT", stb::zlib_compress(filtered, level));
  append_chunk(png, "IEND", {});
  return png;
}

template <typename T>
void write_png_impl(const std::filesystem::path& path, ImageView<const T> image, int level) {
  if (image.empty() || image.channels() < 1 || image.channels() > 4) {
    throw_error("write_png: unsupported image {}x{}x{}", image.width(), image.height(), image.channels());
  }
  const std::size_t row_elems = static_cast<std::size_t>(image.width()) * static_cast<std::size_t>(image.channels());
  std::vector<std::uint8_t> rows(row_elems * sizeof(T) * static_cast<std::size_t>(image.height()));
  std::uint8_t* out = rows.data();
  for (int y = 0; y < image.height(); ++y) {
    const T* in = image.row(y);
    for (std::size_t i = 0; i < row_elems; ++i) {
      if constexpr (sizeof(T) == 1) {
        *out++ = in[i];
      } else {
        *out++ = static_cast<std::uint8_t>(in[i] >> 8);  // PNG samples are big-endian
        *out++ = static_cast<std::uint8_t>(in[i] & 0xFFu);
      }
    }
  }
  write_file(path, encode_png(rows, image.width(), image.height(), image.channels(), 8 * sizeof(T), level));
}

}  // namespace

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw_io_error("cannot open '{}' for reading", path_string(path));
  }
  const std::streamoff size = file.tellg();
  if (size < 0) {
    throw_io_error("cannot determine size of '{}'", path_string(path));
  }
  std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
  file.seekg(0);
  if (!data.empty() && !file.read(reinterpret_cast<char*>(data.data()), size)) {
    throw_io_error("error reading '{}'", path_string(path));
  }
  return data;
}

void write_file(const std::filesystem::path& path, const std::vector<std::uint8_t>& data) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    throw_io_error("cannot open '{}' for writing", path_string(path));
  }
  file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  if (!file) {
    throw_io_error("error writing '{}'", path_string(path));
  }
}

ImageFileInfo read_image_info(const std::filesystem::path& path) {
  const std::vector<std::uint8_t> encoded = read_file(path);
  stb::DecodedInfo info;
  if (!stb::info(encoded, info)) {
    throw_io_error("'{}': unsupported or corrupt image file", path_string(path));
  }
  return {info.width, info.height, info.channels, info.bit_depth};
}

ImageU8 read_image_u8(const std::filesystem::path& path, int desired_channels) {
  return decode_into_image<std::uint8_t>(path, desired_channels);
}

ImageU16 read_image_u16(const std::filesystem::path& path, int desired_channels) {
  return decode_into_image<std::uint16_t>(path, desired_channels);
}

void write_png(const std::filesystem::path& path, ImageView<const std::uint8_t> image, int compression_level) {
  write_png_impl(path, image, compression_level);
}

void write_png(const std::filesystem::path& path, ImageView<const std::uint16_t> image, int compression_level) {
  write_png_impl(path, image, compression_level);
}

void write_jpeg(const std::filesystem::path& path, ImageView<const std::uint8_t> image, int quality) {
  if (image.empty() || (image.channels() != 1 && image.channels() != 3)) {
    throw_error("write_jpeg: unsupported image {}x{}x{}", image.width(), image.height(), image.channels());
  }
  const std::size_t row_elems = static_cast<std::size_t>(image.width()) * static_cast<std::size_t>(image.channels());
  std::vector<std::uint8_t> pixels(row_elems * static_cast<std::size_t>(image.height()));
  for (int y = 0; y < image.height(); ++y) {
    std::memcpy(pixels.data() + static_cast<std::size_t>(y) * row_elems, image.row(y), row_elems);
  }
  write_file(path, stb::jpeg_encode(pixels.data(), image.width(), image.height(), image.channels(),
                                    std::clamp(quality, 1, 100)));
}

}  // namespace r3d
