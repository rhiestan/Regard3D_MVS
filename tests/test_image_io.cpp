#include <cstdint>
#include <fstream>

#include <doctest/doctest.h>

#include "r3d/io/image_io.hpp"
#include "test_util.hpp"

namespace {

template <typename T>
r3d::Image<T> pattern(int w, int h, int c) {
  r3d::Image<T> img(w, h, c);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int k = 0; k < c; ++k) {
        const unsigned v = static_cast<unsigned>(x * 131 + y * 977 + k * 4099) * 2654435761u;
        img(x, y, k) = static_cast<T>(v >> (32 - 8 * sizeof(T)));
      }
  return img;
}

template <typename T>
void check_equal(const r3d::Image<T>& a, const r3d::Image<T>& b) {
  REQUIRE(a.width() == b.width());
  REQUIRE(a.height() == b.height());
  REQUIRE(a.channels() == b.channels());
  for (int y = 0; y < a.height(); ++y)
    for (int x = 0; x < a.width(); ++x)
      for (int k = 0; k < a.channels(); ++k) REQUIRE(a(x, y, k) == b(x, y, k));
}

}  // namespace

TEST_CASE("8-bit PNG round trip") {
  for (const int channels : {1, 2, 3, 4}) {
    CAPTURE(channels);
    const r3d::test::TempDir dir;
    const auto path = dir.path() / "img8.png";
    const auto img = pattern<std::uint8_t>(53, 17, channels);
    r3d::write_png(path, img.view());

    const r3d::ImageFileInfo info = r3d::read_image_info(path);
    CHECK(info.width == 53);
    CHECK(info.height == 17);
    CHECK(info.channels == channels);
    CHECK(info.bit_depth == 8);
    check_equal(r3d::read_image_u8(path), img);
  }
}

TEST_CASE("16-bit PNG round trip") {
  for (const int channels : {1, 3, 4}) {
    CAPTURE(channels);
    const r3d::test::TempDir dir;
    const auto path = dir.path() / "img16.png";
    const auto img = pattern<std::uint16_t>(31, 22, channels);
    r3d::write_png(path, img.view());

    CHECK(r3d::read_image_info(path).bit_depth == 16);
    check_equal(r3d::read_image_u16(path), img);
  }
}

TEST_CASE("Channel conversion and bit-depth promotion on read") {
  const r3d::test::TempDir dir;
  const auto path = dir.path() / "gray.png";
  r3d::ImageU8 gray(4, 2, 1);
  gray(1, 1) = 200;
  r3d::write_png(path, gray.view());

  const r3d::ImageU8 rgb = r3d::read_image_u8(path, 3);
  CHECK(rgb.channels() == 3);
  CHECK(rgb(1, 1, 0) == 200);
  CHECK(rgb(1, 1, 2) == 200);

  const r3d::ImageU16 wide = r3d::read_image_u16(path);
  CHECK(wide(1, 1) == 200 * 257);
}

TEST_CASE("Unicode paths and error reporting") {
  const r3d::test::TempDir dir;
  const auto path = dir.path() / std::filesystem::path(u8"bild_äöü_写真.png");
  const auto img = pattern<std::uint8_t>(8, 8, 3);
  r3d::write_png(path, img.view());
  check_equal(r3d::read_image_u8(path), img);

  CHECK_THROWS_AS(static_cast<void>(r3d::read_image_u8(dir.path() / "missing.png")), r3d::IoError);
  const auto junk = dir.path() / "junk.png";
  {
    std::ofstream(junk, std::ios::binary) << "not an image";
  }
  CHECK_THROWS_AS(static_cast<void>(r3d::read_image_u8(junk)), r3d::IoError);
  CHECK_THROWS_AS(static_cast<void>(r3d::read_image_info(junk)), r3d::IoError);
}
