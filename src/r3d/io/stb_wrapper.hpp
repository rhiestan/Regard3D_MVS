#pragma once

// Internal wrapper around stb_image / stb_image_write (not part of the public API).

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "r3d/core/error.hpp"

namespace r3d::stb {

struct DecodedInfo {
  int width = 0;
  int height = 0;
  int channels = 0;
  int bit_depth = 8;
};

// Tightly packed pixel buffer owned by stb.
struct Decoded {
  struct Free {
    void operator()(void* p) const noexcept;
  };
  int width = 0;
  int height = 0;
  int channels = 0;       // channels in `pixels`
  int file_channels = 0;  // channels stored in the file
  std::unique_ptr<void, Free> pixels;
};

bool info(std::span<const std::uint8_t> encoded, DecodedInfo& out);

// desired_channels 0 keeps the file's channel count. With sixteen_bit, 8-bit
// files are scaled to 16 bit (x * 257); otherwise 16-bit files are reduced to 8 bit.
Decoded decode(std::span<const std::uint8_t> encoded, int desired_channels, bool sixteen_bit);

// zlib stream (RFC 1950) of `data`; level as for stb (higher = smaller, slower).
std::vector<std::uint8_t> zlib_compress(std::span<const std::uint8_t> data, int level);

// Baseline JPEG of tightly packed 8-bit pixels (1-4 channels); quality 1..100.
std::vector<std::uint8_t> jpeg_encode(const std::uint8_t* pixels, int width, int height, int channels, int quality);

}  // namespace r3d::stb
