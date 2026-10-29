// The only translation unit that compiles the stb implementations. The rest
// of the library uses the small wrapper API in stb_wrapper.hpp.

#include "r3d/io/stb_wrapper.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push, 0)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wnull-dereference"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#endif
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif

#define STBI_NO_STDIO
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_PNM
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#define STBI_FAILURE_USERMSG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STBI_WRITE_NO_STDIO
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace r3d::stb {

namespace {
int checked_int(std::size_t size) {
  if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw_io_error("buffer of {} bytes exceeds the 2 GiB limit of the image codec", size);
  }
  return static_cast<int>(size);
}
}  // namespace

bool info(std::span<const std::uint8_t> encoded, DecodedInfo& out) {
  const int len = checked_int(encoded.size());
  if (stbi_info_from_memory(encoded.data(), len, &out.width, &out.height, &out.channels) == 0) {
    return false;
  }
  out.bit_depth = stbi_is_16_bit_from_memory(encoded.data(), len) != 0 ? 16 : 8;
  return true;
}

Decoded decode(std::span<const std::uint8_t> encoded, int desired_channels, bool sixteen_bit) {
  const int len = checked_int(encoded.size());
  Decoded result;
  void* pixels = nullptr;
  if (sixteen_bit) {
    pixels = stbi_load_16_from_memory(encoded.data(), len, &result.width, &result.height, &result.file_channels,
                                      desired_channels);
  } else {
    pixels = stbi_load_from_memory(encoded.data(), len, &result.width, &result.height, &result.file_channels,
                                   desired_channels);
  }
  if (pixels == nullptr) {
    throw_io_error("image decoding failed: {}", stbi_failure_reason());
  }
  result.channels = desired_channels != 0 ? desired_channels : result.file_channels;
  result.pixels.reset(pixels);
  return result;
}

void Decoded::Free::operator()(void* p) const noexcept { stbi_image_free(p); }

std::vector<std::uint8_t> zlib_compress(std::span<const std::uint8_t> data, int level) {
  int out_len = 0;
  // stb takes a non-const pointer but does not modify the input.
  unsigned char* out = stbi_zlib_compress(const_cast<unsigned char*>(data.data()), checked_int(data.size()),
                                          &out_len, level);
  if (out == nullptr) {
    throw_io_error("zlib compression failed");
  }
  std::vector<std::uint8_t> result(out, out + out_len);
  STBIW_FREE(out);
  return result;
}

std::vector<std::uint8_t> jpeg_encode(const std::uint8_t* pixels, int width, int height, int channels, int quality) {
  std::vector<std::uint8_t> result;
  auto append = [](void* context, void* data, int size) {
    auto* out = static_cast<std::vector<std::uint8_t>*>(context);
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    out->insert(out->end(), bytes, bytes + size);
  };
  if (stbi_write_jpg_to_func(append, &result, width, height, channels, pixels, quality) == 0) {
    throw_io_error("JPEG encoding failed");
  }
  return result;
}

}  // namespace r3d::stb
