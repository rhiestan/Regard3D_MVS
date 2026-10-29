#include <cstdint>
#include <fstream>

#include <doctest/doctest.h>

#include "r3d/io/ply.hpp"
#include "test_util.hpp"


namespace {

r3d::PointCloud make_cloud(std::size_t n, bool full) {
  r3d::PointCloud cloud;
  for (std::size_t i = 0; i < n; ++i) {
    const auto f = static_cast<float>(i);
    cloud.positions.emplace_back(f * 0.5f, -f, f * f * 0.001f);
    if (full) {
      cloud.normals.emplace_back(0.0f, 0.6f, 0.8f);
      cloud.colors.push_back({static_cast<std::uint8_t>(i), 128, static_cast<std::uint8_t>(255 - i % 256)});
      std::vector<std::uint32_t> views;
      for (std::uint32_t k = 0; k < i % 4; ++k) views.push_back(static_cast<std::uint32_t>(i) * 10 + k);
      cloud.visibility.push_back(views);
    }
  }
  if (full) {
    cloud.scalar("value");
    cloud.scalar("confidence");  // may reallocate: take references only afterwards
    auto& scale = cloud.scalar("value");
    auto& confidence = cloud.scalar("confidence");
    for (std::size_t i = 0; i < n; ++i) {
      scale[i] = 0.01f * static_cast<float>(i + 1);
      confidence[i] = 1.0f / static_cast<float>(i + 1);
    }
  }
  return cloud;
}

void check_same(const r3d::PointCloud& a, const r3d::PointCloud& b) {
  REQUIRE(a.size() == b.size());
  REQUIRE(a.normals.size() == b.normals.size());
  REQUIRE(a.colors.size() == b.colors.size());
  REQUIRE(a.visibility.size() == b.visibility.size());
  REQUIRE(a.scalars.size() == b.scalars.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    const Eigen::Vector3d pa = a.world_position(i);
    const Eigen::Vector3d pb = b.world_position(i);
    R3D_CHECK_NEAR(pa.x(), pb.x(), 1e-4);
    R3D_CHECK_NEAR(pa.y(), pb.y(), 1e-4);
    R3D_CHECK_NEAR(pa.z(), pb.z(), 1e-4);
    if (a.has_normals()) CHECK(a.normals[i] == b.normals[i]);
    if (a.has_colors()) CHECK(a.colors[i] == b.colors[i]);
    if (a.has_visibility()) CHECK(a.visibility[i] == b.visibility[i]);
  }
  for (std::size_t s = 0; s < a.scalars.size(); ++s) {
    CHECK(a.scalars[s].first == b.scalars[s].first);
    CHECK(a.scalars[s].second == b.scalars[s].second);
  }
}

}  // namespace

TEST_CASE("PLY round trip with all attributes") {
  for (const auto format : {r3d::PlyFormat::binary_little_endian, r3d::PlyFormat::ascii}) {
    for (const bool full : {false, true}) {
      CAPTURE(static_cast<int>(format));
      CAPTURE(full);
      const r3d::test::TempDir dir;
      const auto path = dir.path() / "cloud.ply";
      const r3d::PointCloud cloud = make_cloud(1000, full);
      r3d::write_ply(path, cloud, {.format = format, .double_positions = false, .comments = {"unit test"}});
      check_same(r3d::read_ply(path), cloud);
    }
  }
}

TEST_CASE("PLY keeps precision for georeferenced coordinates") {
  for (const bool double_positions : {false, true}) {
    CAPTURE(double_positions);
    const r3d::test::TempDir dir;
    const auto path = dir.path() / "geo.ply";
    r3d::PointCloud cloud = make_cloud(10, false);
    cloud.origin = Eigen::Vector3d(2600000.25, 1200000.5, 450.0);  // Swiss LV95-like magnitudes
    r3d::PlyWriteOptions options;
    options.double_positions = double_positions;
    r3d::write_ply(path, cloud, options);

    const r3d::PointCloud loaded = r3d::read_ply(path);
    for (std::size_t i = 0; i < cloud.size(); ++i) {
      const Eigen::Vector3d diff = loaded.world_position(i) - cloud.world_position(i);
      CHECK(diff.norm() < 1e-4);
    }
  }
}

TEST_CASE("PLY reader handles foreign files") {
  const r3d::test::TempDir dir;
  const auto path = dir.path() / "foreign.ply";
  {
    std::ofstream f(path, std::ios::binary);
    f << "ply\r\nformat ascii 1.0\r\nobj_info test\r\n"
         "element vertex 2\r\nproperty double x\r\nproperty double y\r\nproperty double z\r\n"
         "property float red\r\nproperty float green\r\nproperty float blue\r\nproperty float alpha\r\n"
         "property ushort quality\r\n"
         "element face 1\r\nproperty list uchar int vertex_indices\r\nend_header\r\n"
         "1000000.5 2.0 3.0 1.0 0.0 0.5 1.0 7\r\n"
         "1000001.5 2.0 3.0 0.0 1.0 0.0 1.0 9\r\n"
         "3 0 1 0\r\n";
  }
  const r3d::PointCloud cloud = r3d::read_ply(path);
  REQUIRE(cloud.size() == 2);
  CHECK(cloud.origin.x() == 1000001.0);  // rounded first point
  R3D_CHECK_NEAR(cloud.world_position(1).x(), 1000001.5, 1e-6);
  CHECK(cloud.colors[0] == std::array<std::uint8_t, 3>{255, 0, 128});
  const std::vector<float>* quality = cloud.find_scalar("quality");
  REQUIRE(quality != nullptr);
  CHECK((*quality)[1] == 9.0f);
}

TEST_CASE("PLY writer rejects inconsistent clouds and bad names") {
  const r3d::test::TempDir dir;
  r3d::PointCloud cloud = make_cloud(5, false);
  cloud.normals.resize(3);
  CHECK_THROWS_AS(r3d::write_ply(dir.path() / "a.ply", cloud), r3d::Error);

  r3d::PointCloud named = make_cloud(5, false);
  named.scalar("bad name");
  CHECK_THROWS_AS(r3d::write_ply(dir.path() / "b.ply", named), r3d::Error);

  CHECK_THROWS_AS(static_cast<void>(r3d::read_ply(dir.path() / "missing.ply")), r3d::IoError);
}

TEST_CASE("PLY writer in parts") {
  for (const auto format : {r3d::PlyFormat::binary_little_endian, r3d::PlyFormat::ascii}) {
    CAPTURE(static_cast<int>(format));
    const r3d::test::TempDir dir;
    const r3d::PointCloud cloud = make_cloud(300, true);
    r3d::PlyWriteOptions options;
    options.format = format;
    {
      r3d::PlyWriter writer(dir.path() / "parts.ply", options);
      writer.append(r3d::PointCloud{});  // empty parts are skipped
      for (std::size_t begin = 0; begin < cloud.size(); begin += 70) {
        const std::size_t end = std::min(cloud.size(), begin + 70);
        r3d::PointCloud part;
        part.positions.assign(cloud.positions.begin() + begin, cloud.positions.begin() + end);
        part.normals.assign(cloud.normals.begin() + begin, cloud.normals.begin() + end);
        part.colors.assign(cloud.colors.begin() + begin, cloud.colors.begin() + end);
        part.visibility.assign(cloud.visibility.begin() + begin, cloud.visibility.begin() + end);
        for (const auto& [name, values] : cloud.scalars) {
          part.scalars.emplace_back(name, std::vector<float>(values.begin() + begin, values.begin() + end));
        }
        writer.append(part);
      }
      CHECK(writer.size() == 300);
      CHECK_THROWS(writer.append(make_cloud(3, false)));  // other attributes
      writer.close();
    }
    check_same(r3d::read_ply(dir.path() / "parts.ply"), cloud);
    {
      r3d::PlyWriter empty(dir.path() / "empty.ply", options);
    }
    CHECK(r3d::read_ply(dir.path() / "empty.ply").size() == 0);
  }
}
