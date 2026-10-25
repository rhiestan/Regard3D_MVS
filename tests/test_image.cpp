#include <cstdint>
#include <utility>

#include <doctest/doctest.h>

#include "r3d/core/image.hpp"

using r3d::Image;
using r3d::ImageView;

TEST_CASE("Image rows are 64-byte aligned and zero-initialised") {
  const Image<std::uint8_t> img(37, 5, 3);
  CHECK(img.width() == 37);
  CHECK(img.height() == 5);
  CHECK(img.channels() == 3);
  CHECK(img.stride() == 128);  // 111 bytes rounded up to 2 cache lines
  for (int y = 0; y < img.height(); ++y) {
    CHECK(reinterpret_cast<std::uintptr_t>(img.row(y)) % r3d::kBufferAlignment == 0);
    for (int x = 0; x < img.width(); ++x) {
      for (int c = 0; c < 3; ++c) {
        CHECK(img(x, y, c) == 0);
      }
    }
  }
}

TEST_CASE("Image stride accounts for element size") {
  CHECK(Image<float>::aligned_stride(10, 1) == 16);
  CHECK(Image<float>::aligned_stride(16, 1) == 16);
  CHECK(Image<float>::aligned_stride(17, 1) == 32);
  CHECK(Image<std::uint16_t>::aligned_stride(33, 1) == 64);
  CHECK(Image<double>::aligned_stride(0, 1) == 0);
}

TEST_CASE("Image pixel access, copy and move") {
  Image<float> img(4, 3, 2);
  img(3, 2, 1) = 42.0f;
  img(0, 0, 0) = -1.0f;

  Image<float> copy = img;
  CHECK(copy(3, 2, 1) == 42.0f);
  copy(3, 2, 1) = 7.0f;
  CHECK(img(3, 2, 1) == 42.0f);  // deep copy

  Image<float> moved = std::move(copy);
  CHECK(moved(3, 2, 1) == 7.0f);
  CHECK(moved(0, 0, 0) == -1.0f);
}

TEST_CASE("ImageView converts to const and supports subviews") {
  Image<std::uint16_t> img(8, 6, 1);
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 8; ++x) {
      img(x, y) = static_cast<std::uint16_t>(y * 100 + x);
    }
  }
  const ImageView<std::uint16_t> view = img.view();
  const ImageView<const std::uint16_t> cview = view;
  CHECK(cview(5, 4) == 405);

  const ImageView<const std::uint16_t> sub = cview.subview(2, 1, 3, 4);
  CHECK(sub.width() == 3);
  CHECK(sub.height() == 4);
  CHECK(sub(0, 0) == 102);
  CHECK(sub(2, 3) == 404);
  CHECK(sub.stride() == img.stride());
}

TEST_CASE("Invalid image dimensions throw") {
  CHECK_THROWS_AS(Image<std::uint8_t>(-1, 4), r3d::Error);
  CHECK_THROWS_AS(Image<std::uint8_t>(4, 4, 0), r3d::Error);
  const Image<std::uint8_t> empty(0, 0);
  CHECK(empty.empty());
}
