#include <algorithm>
#include <array>
#include <cmath>

#include <doctest/doctest.h>

#include "r3d/core/camera_model.hpp"
#include "r3d/core/error.hpp"
#include "test_util.hpp"

using r3d::CameraIntrinsics;
using r3d::CameraModel;

namespace {

CameraIntrinsics make_camera(CameraModel model) {
  CameraIntrinsics cam;
  cam.model = model;
  cam.width = 3000;
  cam.height = 2000;
  cam.fx = cam.fy = 2500.0;
  cam.cx = 1510.25;
  cam.cy = 990.75;
  switch (model) {
    case CameraModel::radial_k1:
      cam.params = {-0.12};
      break;
    case CameraModel::radial_k3:
      cam.params = {-0.22, 0.17, 0.04};
      break;
    case CameraModel::brown_t2:
      cam.params = {-0.2, 0.1, -0.02, 0.0015, -0.0008};
      break;
    case CameraModel::fisheye:
      cam.fx = cam.fy = 1200.0;
      cam.params = {0.05, -0.01, 0.002, -0.0004};
      break;
    default:
      break;
  }
  return cam;
}

struct Reference {
  CameraModel model;
  std::array<double, 2> pixel;
  std::array<double, 2> distorted;    // OpenMVG get_d_pixel(pixel)
  std::array<double, 2> undistorted;  // OpenMVG get_ud_pixel(pixel)
};

// Generated with OpenMVG (D:/Projects/OpenMVG_fast, commit 4c13316a) from the
// cameras of make_camera().
constexpr Reference kReference[] = {
    {CameraModel::radial_k1, {0, 0}, {94.600338315000045, 62.059450545000004}, {-118.70654093796406, -77.873534470642653}},
    {CameraModel::radial_k1, {2999, 1999}, {2906.589510839, 1936.4153647714002}, {3114.6664015784095, 2077.3346091630101}},
    {CameraModel::radial_k1, {1510.25, 990.75}, {1510.25, 990.75}, {1510.25, 990.75}},
    {CameraModel::radial_k1, {200.5, 1700.25}, {256.29749537049997, 1670.0241378389999}, {136.05407660147375, 1735.160771255014}},
    {CameraModel::radial_k1, {2500, 300}, {2472.3172665550001, 319.31987686500008}, {2530.3041510619769, 278.85062657634705}},
    {CameraModel::radial_k1, {1000, 1500}, {1005.091305115, 1494.918672945}, {994.74991530021941, 1505.2397954598005}},
    {CameraModel::radial_k3, {0, 0}, {94.886229773841933, 62.2470002638197}, {-100.37779216323815, -65.849559732314674}},
    {CameraModel::radial_k3, {2999, 1999}, {2905.5413226032933, 1935.705483469199}, {3098.1510610271825, 2066.1496606419187}},
    {CameraModel::radial_k3, {1510.25, 990.75}, {1510.25, 990.75}, {1510.25, 990.75}},
    {CameraModel::radial_k3, {200.5, 1700.25}, {272.38869877313937, 1661.3074294487174}, {119.47449794998874, 1744.142035659082}},
    {CameraModel::radial_k3, {2500, 300}, {2458.8903030823376, 328.69060181447378}, {2546.0014528542683, 267.89542454247453}},
    {CameraModel::radial_k3, {1000, 1500}, {1008.7225874716057, 1491.2945072613127}, {990.82209394402673, 1509.1599189789404}},
    {CameraModel::brown_t2, {0, 0}, {120.10466365362117, 81.433060925620907}, {-151.37617797155212, -102.83438171283728}},
    {CameraModel::brown_t2, {2999, 1999}, {2880.0440956200168, 1921.0779945920895}, {3148.6290750755484, 2096.8263925774772}},
    {CameraModel::brown_t2, {1510.25, 990.75}, {1510.25, 990.75}, {1510.25, 990.75}},
    {CameraModel::brown_t2, {200.5, 1700.25}, {275.23746860486563, 1660.7109073379088}, {111.72215187724692, 1747.1890931570956}},
    {CameraModel::brown_t2, {2500, 300}, {2457.074831107941, 330.50633784753313}, {2548.9685156408304, 265.1904152402858}},
    {CameraModel::brown_t2, {1000, 1500}, {1007.4938462570847, 1492.6666796554109}, {992.15840726927945, 1507.673551296326}},
    {CameraModel::fisheye, {0, 0}, {482.5230699299359, 316.54344084296235}, {-4589.1199563813489, -3010.5416962654003}},
    {CameraModel::fisheye, {2999, 1999}, {2525.6513019536892, 1678.4264820787957}, {7335.0822221244907, 4935.5943915748221}},
    {CameraModel::fisheye, {1510.25, 990.75}, {1510.25, 990.75}, {1510.25, 990.75}},
    {CameraModel::fisheye, {200.5, 1700.25}, {536.03868485280293, 1518.4865360541603}, {-1034.5551010390295, 2369.2872927560156}},
    {CameraModel::fisheye, {2500, 300}, {2307.393796094133, 434.42054594390277}, {2939.058927559629, -6.4207670743255676}},
    {CameraModel::fisheye, {1000, 1500}, {1044.4218750461412, 1455.6651839936358}, {940.20835975023419, 1559.6744591811723}},
};

}  // namespace

TEST_CASE("Camera models match OpenMVG distortion") {
  for (const Reference& ref : kReference) {
    CAPTURE(r3d::openmvg_camera_model_name(ref.model));
    CAPTURE(ref.pixel[0]);
    CAPTURE(ref.pixel[1]);
    const CameraIntrinsics cam = make_camera(ref.model);
    const Eigen::Vector2d p(ref.pixel[0], ref.pixel[1]);

    const Eigen::Vector2d d = cam.distort_pixel(p);
    R3D_CHECK_NEAR(d.x(), ref.distorted[0], 1e-9);
    R3D_CHECK_NEAR(d.y(), ref.distorted[1], 1e-9);

    // OpenMVG's inverses stop at a fixed tolerance or iteration count.
    const Eigen::Vector2d u = cam.undistort_pixel(p);
    R3D_CHECK_NEAR(u.x(), ref.undistorted[0], 1e-5);
    R3D_CHECK_NEAR(u.y(), ref.undistorted[1], 1e-5);

    // Our inverse is exact to round-off.
    const Eigen::Vector2d back = cam.distort_pixel(u);
    R3D_CHECK_NEAR(back.x(), p.x(), 1e-9);
    R3D_CHECK_NEAR(back.y(), p.y(), 1e-9);
  }
}

TEST_CASE("Undistortion inverts distortion over the whole image") {
  for (const CameraModel model :
       {CameraModel::radial_k1, CameraModel::radial_k3, CameraModel::brown_t2, CameraModel::fisheye}) {
    CAPTURE(r3d::openmvg_camera_model_name(model));
    const CameraIntrinsics cam = make_camera(model);
    double max_error = 0.0;
    for (int y = -50; y <= cam.height + 50; y += 37) {
      for (int x = -50; x <= cam.width + 50; x += 41) {
        const Eigen::Vector2d p(x, y);
        max_error = std::max(max_error, (cam.distort_pixel(cam.undistort_pixel(p)) - p).norm());
        max_error = std::max(max_error, (cam.undistort_pixel(cam.distort_pixel(p)) - p).norm());
      }
    }
    CHECK(max_error < 1e-8);
  }
}

TEST_CASE("Camera model names and projection") {
  for (const CameraModel model : {CameraModel::pinhole, CameraModel::radial_k1, CameraModel::radial_k3,
                                  CameraModel::brown_t2, CameraModel::fisheye, CameraModel::spherical}) {
    CHECK(r3d::parse_openmvg_camera_model(r3d::openmvg_camera_model_name(model)) == model);
  }
  CHECK_FALSE(r3d::parse_openmvg_camera_model("pinhole_radial_k2").has_value());
  CHECK(r3d::distortion_parameter_count(CameraModel::brown_t2) == 5);

  const CameraIntrinsics cam = make_camera(CameraModel::radial_k3);
  CHECK(cam.has_distortion());
  CHECK_FALSE(make_camera(CameraModel::pinhole).has_distortion());
  const Eigen::Vector3d x_cam(0.3, -0.2, 2.0);
  const Eigen::Vector2d projected = cam.project(x_cam);
  const Eigen::Vector2d expected = cam.distort_pixel(cam.pinhole().project(x_cam));
  R3D_CHECK_NEAR(projected.x(), expected.x(), 1e-9);
  R3D_CHECK_NEAR(projected.y(), expected.y(), 1e-9);

  CameraIntrinsics sphere;
  sphere.model = CameraModel::spherical;
  CHECK_FALSE(sphere.is_perspective());
  CHECK_THROWS_AS(static_cast<void>(sphere.project(x_cam)), r3d::Error);
}
