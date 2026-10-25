#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "r3d/core/aligned_buffer.hpp"
#include "r3d/core/error.hpp"

namespace r3d {

// Non-owning view of an interleaved image. `stride` is the distance between
// rows in elements (not bytes) and may exceed width * channels.
template <typename T>
class ImageView {
public:
  ImageView() = default;
  ImageView(T* data, int width, int height, int channels, std::ptrdiff_t stride)
      : data_(data), width_(width), height_(height), channels_(channels), stride_(stride) {
    assert(width >= 0 && height >= 0 && channels > 0);
    assert(stride >= static_cast<std::ptrdiff_t>(width) * channels);
  }

  // Implicit conversion ImageView<T> -> ImageView<const T>.
  template <typename U>
    requires(std::is_same_v<const U, T> && !std::is_same_v<U, T>)
  ImageView(const ImageView<U>& other)  // NOLINT(google-explicit-constructor)
      : ImageView(other.data(), other.width(), other.height(), other.channels(), other.stride()) {}

  [[nodiscard]] T* data() const noexcept { return data_; }
  [[nodiscard]] int width() const noexcept { return width_; }
  [[nodiscard]] int height() const noexcept { return height_; }
  [[nodiscard]] int channels() const noexcept { return channels_; }
  [[nodiscard]] std::ptrdiff_t stride() const noexcept { return stride_; }
  [[nodiscard]] bool empty() const noexcept { return width_ == 0 || height_ == 0; }

  [[nodiscard]] T* row(int y) const noexcept {
    assert(y >= 0 && y < height_);
    return data_ + static_cast<std::ptrdiff_t>(y) * stride_;
  }

  [[nodiscard]] T& operator()(int x, int y, int c = 0) const noexcept {
    assert(x >= 0 && x < width_ && c >= 0 && c < channels_);
    return row(y)[static_cast<std::ptrdiff_t>(x) * channels_ + c];
  }

  // Sub-rectangle view sharing the same memory.
  [[nodiscard]] ImageView subview(int x, int y, int width, int height) const {
    assert(x >= 0 && y >= 0 && x + width <= width_ && y + height <= height_);
    return ImageView(row(y) + static_cast<std::ptrdiff_t>(x) * channels_, width, height, channels_, stride_);
  }

private:
  T* data_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  int channels_ = 1;
  std::ptrdiff_t stride_ = 0;
};

// Owning interleaved image. Every row starts on a 64-byte boundary, so rows
// can be processed with aligned SIMD loads.
template <typename T>
class Image {
  static_assert(std::is_trivially_copyable_v<T>, "Image requires trivially copyable pixel type");
  static_assert(kBufferAlignment % sizeof(T) == 0, "pixel size must divide the buffer alignment");

public:
  Image() = default;

  // Zero-initialised image.
  Image(int width, int height, int channels = 1) : Image(uninitialized(width, height, channels)) {
    buffer_.fill(T{});
  }

  // Image with unspecified pixel values (use when every pixel is written).
  [[nodiscard]] static Image uninitialized(int width, int height, int channels = 1) {
    if (width < 0 || height < 0 || channels <= 0) {
      throw_error("invalid image dimensions {}x{}x{}", width, height, channels);
    }
    Image img;
    img.width_ = width;
    img.height_ = height;
    img.channels_ = channels;
    img.stride_ = aligned_stride(width, channels);
    img.buffer_.reset_uninitialized(static_cast<std::size_t>(img.stride_) * static_cast<std::size_t>(height));
    return img;
  }

  [[nodiscard]] int width() const noexcept { return width_; }
  [[nodiscard]] int height() const noexcept { return height_; }
  [[nodiscard]] int channels() const noexcept { return channels_; }
  [[nodiscard]] std::ptrdiff_t stride() const noexcept { return stride_; }
  [[nodiscard]] bool empty() const noexcept { return width_ == 0 || height_ == 0; }

  [[nodiscard]] T* data() noexcept { return buffer_.data(); }
  [[nodiscard]] const T* data() const noexcept { return buffer_.data(); }

  [[nodiscard]] T* row(int y) noexcept {
    assert(y >= 0 && y < height_);
    return buffer_.data() + static_cast<std::ptrdiff_t>(y) * stride_;
  }
  [[nodiscard]] const T* row(int y) const noexcept {
    assert(y >= 0 && y < height_);
    return buffer_.data() + static_cast<std::ptrdiff_t>(y) * stride_;
  }

  [[nodiscard]] T& operator()(int x, int y, int c = 0) noexcept {
    assert(x >= 0 && x < width_ && c >= 0 && c < channels_);
    return row(y)[static_cast<std::ptrdiff_t>(x) * channels_ + c];
  }
  [[nodiscard]] const T& operator()(int x, int y, int c = 0) const noexcept {
    assert(x >= 0 && x < width_ && c >= 0 && c < channels_);
    return row(y)[static_cast<std::ptrdiff_t>(x) * channels_ + c];
  }

  [[nodiscard]] ImageView<T> view() noexcept { return {data(), width_, height_, channels_, stride_}; }
  [[nodiscard]] ImageView<const T> view() const noexcept { return {data(), width_, height_, channels_, stride_}; }

  // Fills every pixel, including row padding.
  void fill(const T& value) { buffer_.fill(value); }

  // Row stride (in elements) for the given geometry, rounded up to the alignment.
  [[nodiscard]] static std::ptrdiff_t aligned_stride(int width, int channels) noexcept {
    constexpr std::ptrdiff_t kElemsPerLine = static_cast<std::ptrdiff_t>(kBufferAlignment / sizeof(T));
    const std::ptrdiff_t elems = static_cast<std::ptrdiff_t>(width) * channels;
    return (elems + kElemsPerLine - 1) / kElemsPerLine * kElemsPerLine;
  }

private:
  AlignedBuffer<T> buffer_;
  int width_ = 0;
  int height_ = 0;
  int channels_ = 1;
  std::ptrdiff_t stride_ = 0;
};

using ImageU8 = Image<std::uint8_t>;
using ImageU16 = Image<std::uint16_t>;
using ImageF32 = Image<float>;

}  // namespace r3d
