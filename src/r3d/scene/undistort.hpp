#pragma once

#include "r3d/core/camera.hpp"
#include "r3d/core/camera_model.hpp"
#include "r3d/core/image.hpp"

namespace r3d {

// Which part of the undistorted image plane the output image covers. The
// focal length is always kept, so the resolution at the image centre is unchanged.
enum class UndistortFit {
  keep_all,   // bounding box of the whole source image; invalid corners are masked
  inner,      // largest axis-aligned rectangle with only valid pixels
  same_size,  // same K and size as the source (OpenMVG / texrecon convention)
};

struct UndistortOptions {
  UndistortFit fit = UndistortFit::keep_all;
  // keep_all: each side may extend at most this factor beyond the source
  // image (from the principal point), against fisheye-like blow-up.
  double max_growth = 1.5;
};

// Pinhole camera of the undistorted images of `cam`.
[[nodiscard]] PinholeIntrinsics undistorted_camera(const CameraIntrinsics& cam, const UndistortOptions& options = {});

// True if the distortion is one-to-one over the whole source image (positive
// Jacobian determinant everywhere). Over-fitted polynomials can fold over
// near the border, where undistortion is then ambiguous.
[[nodiscard]] bool distortion_is_monotonic(const CameraIntrinsics& cam);

// Resamples `src` (an image of `cam`) into the pinhole camera `target` with
// bicubic interpolation. Pixels that map outside the source are set to 0;
// if `valid` is not null it receives a mask (255 valid, 0 invalid).
template <typename T>
[[nodiscard]] Image<T> undistort_image(const ImageView<const T>& src, const CameraIntrinsics& cam,
                                       const PinholeIntrinsics& target, ImageU8* valid = nullptr);

template <typename T>
[[nodiscard]] Image<T> undistort_image(const Image<T>& src, const CameraIntrinsics& cam, const PinholeIntrinsics& target,
                                       ImageU8* valid = nullptr) {
  return undistort_image(src.view(), cam, target, valid);
}

}  // namespace r3d
