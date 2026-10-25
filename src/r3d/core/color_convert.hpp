#pragma once

#include <cstddef>
#include <cstdint>

#include "r3d/core/image.hpp"

namespace r3d {

// Luma weights (BT.601) in 8.8 fixed point: gray = (77 R + 150 G + 29 B + 128) >> 8.
// Integer arithmetic makes every SIMD target bit-exact with the scalar code.
inline constexpr std::uint32_t kLumaR = 77;
inline constexpr std::uint32_t kLumaG = 150;
inline constexpr std::uint32_t kLumaB = 29;

// Converts a 1-, 3- or 4-channel 8-bit image to a single-channel image
// (4 channels: alpha is ignored). Rows are processed in parallel.
[[nodiscard]] ImageU8 to_gray(ImageView<const std::uint8_t> src);

// SIMD row kernels (runtime dispatch). `n` is the number of pixels.
void rgb_to_gray_row(const std::uint8_t* rgb, std::uint8_t* gray, std::size_t n);
void rgba_to_gray_row(const std::uint8_t* rgba, std::uint8_t* gray, std::size_t n);

namespace scalar {
// Reference implementations, used for tails and in tests.
void rgb_to_gray_row(const std::uint8_t* rgb, std::uint8_t* gray, std::size_t n);
void rgba_to_gray_row(const std::uint8_t* rgba, std::uint8_t* gray, std::size_t n);
}  // namespace scalar

}  // namespace r3d
