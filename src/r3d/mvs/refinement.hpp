#pragma once

#include <cstddef>
#include <span>

#include "r3d/core/camera.hpp"
#include "r3d/core/image.hpp"
#include "r3d/mvs/depth_map.hpp"

namespace r3d {

// Patch-based depth refinement (plan.md section 3.6, Phase 5).
//
// SGM depths are biased towards fronto-parallel surfaces and locked to
// sub-pixel positions. Every valid pixel is refined as a small plane: in the
// base view's inverse depth w = 1 / z, a plane is affine in pixel
// coordinates, w(q) = w + a dx + b dy for the window offset (dx, dy). The
// three parameters (w, a, b) are fitted by Gauss-Newton so that the window,
// warped by the plane into the neighbouring (undistorted) images, has the
// highest normalised cross-correlation (NCC) with the base window; the
// objective is the sum over views of the squared differences of the
// normalised windows (2 - 2 NCC per view), which also absorbs a per-view
// gain and offset. Views below `min_ncc` at the initial depth are not used
// for that pixel (occlusions, highlights). Deterministic, no random search
// (Goesele et al. 2007; Baltsavias' MPGC).
//
// A refined depth replaces the input only when the mean NCC improved and
// the depth moved by at most `max_shift_px` in every used view; otherwise
// the input depth stays. The confidence becomes the mean NCC.
struct RefinementOptions {
  int radius = 4;              // window half size in pixels
  int step = 2;                // spacing of the window samples (4 / 2: 5x5 samples over 9x9 pixels)
  int iterations = 6;          // Gauss-Newton steps per pixel
  int max_views = 4;           // neighbouring images per depth map
  double min_ncc = 0.3;        // per view at the initial depth
  double max_shift_px = 2.0;   // displacement of the refined depth in the used views
  double min_gain = 0.02;      // mean NCC improvement a refined depth needs
  double min_texture = 1.0;    // standard deviation of the base window (grey levels 0..255)
  double blur_sigma = 0.0;     // Gaussian pre-filter of all images (pixels), see make_refinement_image
};

struct RefinementStats {
  std::size_t input = 0;       // valid pixels
  std::size_t refined = 0;     // depth replaced
  std::size_t kept = 0;        // input depth kept (no texture, no view, no improvement, too far)
};

// A view's image on its depth-map grid: intensity (0..255) and its x / y
// derivatives as three interleaved channels.
struct RefinementImage {
  PinholeCamera camera;
  ImageF32 data;
};

// Builds the refinement image from a single-channel image on `camera`'s grid,
// optionally smoothed by a Gaussian of `blur_sigma` pixels first.
[[nodiscard]] RefinementImage make_refinement_image(const PinholeCamera& camera, const ImageF32& gray,
                                                    double blur_sigma = 0.0);

// Refines `map` in place. `base` must be on map.camera's grid; `views` are
// the neighbouring images (at most RefinementOptions::max_views are used).
void refine_depth_map(DepthMap& map, const RefinementImage& base, std::span<const RefinementImage* const> views,
                      const RefinementOptions& options = {}, RefinementStats* stats = nullptr);

}  // namespace r3d
