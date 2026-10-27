#include "r3d/mvs/refinement.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/LU>

#include "r3d/core/error.hpp"
#include "r3d/core/thread_pool.hpp"

namespace r3d {

namespace {

constexpr int kMaxSamples = 225;  // 15 x 15 window samples
constexpr int kMaxViews = 8;
constexpr double kDiscontinuity = 0.05;  // relative depth jump that stops the initial slope estimate

// Plane-induced warp from the base grid into a view, in inverse depth:
// h = m * (qx, qy, 1) + b * w(q), pixel = h.xy / h.z.
struct ViewGeometry {
  Eigen::Matrix3d m;  // K_j R_j R_0^T K_0^-1
  Eigen::Vector3d b;  // K_j R_j (C_0 - C_j)
  const ImageF32* data = nullptr;
};

struct Sample {
  double dx = 0.0;  // window offset in pixels
  double dy = 0.0;
  double sx = 0.0;  // offset / radius (slope parameters are per window radius)
  double sy = 0.0;
};

struct ViewResult {
  bool valid = false;
  double ncc = 0.0;
  Eigen::Matrix3d h = Eigen::Matrix3d::Zero();  // J^T J
  Eigen::Vector3d g = Eigen::Vector3d::Zero();  // J^T e
};

// NCC of the warped window with the normalised base window n0 and, with
// `jacobian`, the Gauss-Newton terms of sum_q (n_q - n0_q)^2 for the
// parameters (w, a, b) of w(q) = w + a sx + b sy.
ViewResult evaluate_view(const ViewGeometry& geo, double x, double y, const Eigen::Vector3d& theta,
                         std::span<const Sample> samples, const double* n0, bool jacobian) {
  ViewResult res;
  const ImageF32& img = *geo.data;
  const double max_x = img.width() - 1;
  const double max_y = img.height() - 1;
  const Eigen::Vector3d hc = geo.m * Eigen::Vector3d(x, y, 1.0);
  const Eigen::Vector3d mx = geo.m.col(0);
  const Eigen::Vector3d my = geo.m.col(1);
  const auto n = static_cast<int>(samples.size());
  std::array<double, kMaxSamples> v{};
  std::array<double, kMaxSamples> dv{};  // dv / dw(q)
  double sum = 0.0;
  for (int s = 0; s < n; ++s) {
    const Sample& sm = samples[static_cast<std::size_t>(s)];
    const double wq = theta[0] + theta[1] * sm.sx + theta[2] * sm.sy;
    const Eigen::Vector3d h = hc + mx * sm.dx + my * sm.dy + geo.b * wq;
    if (!(h.z() > 0.0)) {
      return res;
    }
    const double inv_z = 1.0 / h.z();
    const double px = h.x() * inv_z;
    const double py = h.y() * inv_z;
    if (!(px >= 0.0 && py >= 0.0 && px < max_x && py < max_y)) {
      return res;
    }
    const auto ix = static_cast<int>(px);
    const auto iy = static_cast<int>(py);
    const double fx = px - ix;
    const double fy = py - iy;
    const float* r0 = img.row(iy) + static_cast<std::ptrdiff_t>(ix) * 3;
    const float* r1 = img.row(iy + 1) + static_cast<std::ptrdiff_t>(ix) * 3;
    std::array<double, 3> val{};
    for (int c = 0; c < 3; ++c) {
      val[static_cast<std::size_t>(c)] = (1.0 - fy) * ((1.0 - fx) * r0[c] + fx * r0[3 + c]) +
                                         fy * ((1.0 - fx) * r1[c] + fx * r1[3 + c]);
    }
    v[static_cast<std::size_t>(s)] = val[0];
    sum += val[0];
    if (jacobian) {
      const double dpx = (geo.b.x() - px * geo.b.z()) * inv_z;
      const double dpy = (geo.b.y() - py * geo.b.z()) * inv_z;
      dv[static_cast<std::size_t>(s)] = val[1] * dpx + val[2] * dpy;
    }
  }
  const double mean = sum / n;
  double norm2 = 0.0;
  for (int s = 0; s < n; ++s) {
    v[static_cast<std::size_t>(s)] -= mean;
    norm2 += v[static_cast<std::size_t>(s)] * v[static_cast<std::size_t>(s)];
  }
  if (!(norm2 > 1e-6 * n)) {
    return res;
  }
  const double inv_norm = 1.0 / std::sqrt(norm2);
  double ncc = 0.0;
  for (int s = 0; s < n; ++s) {
    v[static_cast<std::size_t>(s)] *= inv_norm;  // now the normalised window
    ncc += v[static_cast<std::size_t>(s)] * n0[s];
  }
  res.valid = true;
  res.ncc = ncc;
  if (!jacobian) {
    return res;
  }
  // d(v - mean) per parameter, then the derivative of the normalisation:
  // dn = (I - n n^T) d(v - mean) / |v - mean|.
  Eigen::Vector3d dmean = Eigen::Vector3d::Zero();
  for (int s = 0; s < n; ++s) {
    const Sample& sm = samples[static_cast<std::size_t>(s)];
    dmean += dv[static_cast<std::size_t>(s)] * Eigen::Vector3d(1.0, sm.sx, sm.sy);
  }
  dmean /= n;
  Eigen::Vector3d proj = Eigen::Vector3d::Zero();
  for (int s = 0; s < n; ++s) {
    const Sample& sm = samples[static_cast<std::size_t>(s)];
    const Eigen::Vector3d d = dv[static_cast<std::size_t>(s)] * Eigen::Vector3d(1.0, sm.sx, sm.sy) - dmean;
    proj += v[static_cast<std::size_t>(s)] * d;
  }
  for (int s = 0; s < n; ++s) {
    const Sample& sm = samples[static_cast<std::size_t>(s)];
    const double ns = v[static_cast<std::size_t>(s)];
    const Eigen::Vector3d d = dv[static_cast<std::size_t>(s)] * Eigen::Vector3d(1.0, sm.sx, sm.sy) - dmean;
    const Eigen::Vector3d j = (d - ns * proj) * inv_norm;
    res.h.noalias() += j * j.transpose();
    res.g += j * (ns - n0[s]);
  }
  return res;
}

// Pixel displacement in a view per unit of inverse depth at the window centre.
double centre_motion(const ViewGeometry& geo, double x, double y, double w) {
  const Eigen::Vector3d h = geo.m * Eigen::Vector3d(x, y, 1.0) + geo.b * w;
  if (!(h.z() > 0.0)) {
    return 0.0;
  }
  const double px = h.x() / h.z();
  const double py = h.y() / h.z();
  return std::hypot(geo.b.x() - px * geo.b.z(), geo.b.y() - py * geo.b.z()) / h.z();
}

// Inverse-depth slope along one axis from the neighbours at +-1 (one side
// where the other is missing or across a depth jump), per pixel.
double initial_slope(const ImageF32& depth, int x, int y, int ax, int ay) {
  const float z = depth(x, y);
  auto neighbour = [&](int k, double& w) {
    const int nx = x + k * ax;
    const int ny = y + k * ay;
    if (nx < 0 || ny < 0 || nx >= depth.width() || ny >= depth.height()) {
      return false;
    }
    const float zn = depth(nx, ny);
    if (!(zn > 0.0F) || std::abs(zn - z) > kDiscontinuity * std::min(z, zn)) {
      return false;
    }
    w = 1.0 / zn;
    return true;
  };
  double wp = 0.0, wm = 0.0;
  const bool p = neighbour(1, wp);
  const bool m = neighbour(-1, wm);
  const double w0 = 1.0 / z;
  if (p && m) {
    return 0.5 * (wp - wm);
  }
  if (p) {
    return wp - w0;
  }
  if (m) {
    return w0 - wm;
  }
  return 0.0;
}

// Separable Gaussian, taps up to 3 sigma, borders clamped.
ImageF32 gaussian_blur(const ImageF32& src, double sigma) {
  const int r = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
  std::vector<float> k(static_cast<std::size_t>(2 * r + 1));
  float ksum = 0.0F;
  for (int i = -r; i <= r; ++i) {
    k[static_cast<std::size_t>(i + r)] = static_cast<float>(std::exp(-0.5 * i * i / (sigma * sigma)));
    ksum += k[static_cast<std::size_t>(i + r)];
  }
  for (float& v : k) {
    v /= ksum;
  }
  const int w = src.width();
  const int h = src.height();
  ImageF32 tmp(w, h, 1);
  ImageF32 out(w, h, 1);
  parallel_for(0, static_cast<std::size_t>(h), 32, [&](std::size_t begin, std::size_t end) {
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      const float* row = src.row(y);
      float* o = tmp.row(y);
      for (int x = 0; x < w; ++x) {
        float acc = 0.0F;
        for (int i = -r; i <= r; ++i) {
          acc += k[static_cast<std::size_t>(i + r)] * row[std::clamp(x + i, 0, w - 1)];
        }
        o[x] = acc;
      }
    }
  });
  parallel_for(0, static_cast<std::size_t>(h), 32, [&](std::size_t begin, std::size_t end) {
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      float* o = out.row(y);
      for (int x = 0; x < w; ++x) {
        o[x] = 0.0F;
      }
      for (int i = -r; i <= r; ++i) {
        const float* row = tmp.row(std::clamp(y + i, 0, h - 1));
        const float kv = k[static_cast<std::size_t>(i + r)];
        for (int x = 0; x < w; ++x) {
          o[x] += kv * row[x];
        }
      }
    }
  });
  return out;
}

}  // namespace

RefinementImage make_refinement_image(const PinholeCamera& camera, const ImageF32& gray_in, double blur_sigma) {
  const ImageF32 blurred = blur_sigma > 0.0 ? gaussian_blur(gray_in, blur_sigma) : ImageF32();
  const ImageF32& gray = blur_sigma > 0.0 ? blurred : gray_in;
  if (gray.channels() != 1 || gray.width() != camera.intrinsics.width || gray.height() != camera.intrinsics.height) {
    throw_error("refinement image: {}x{}x{} does not match the camera grid {}x{}", gray.width(), gray.height(),
                gray.channels(), camera.intrinsics.width, camera.intrinsics.height);
  }
  RefinementImage out;
  out.camera = camera;
  const int w = gray.width();
  const int h = gray.height();
  out.data = ImageF32(w, h, 3);
  parallel_for(0, static_cast<std::size_t>(h), 32, [&](std::size_t begin, std::size_t end) {
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      const float* row = gray.row(y);
      const float* up = gray.row(std::max(y - 1, 0));
      const float* down = gray.row(std::min(y + 1, h - 1));
      const float sy = (y > 0 && y < h - 1) ? 0.5F : 1.0F;
      float* o = out.data.row(y);
      for (int x = 0; x < w; ++x) {
        const int xl = std::max(x - 1, 0);
        const int xr = std::min(x + 1, w - 1);
        o[3 * x] = row[x];
        o[3 * x + 1] = (row[xr] - row[xl]) / static_cast<float>(std::max(xr - xl, 1));
        o[3 * x + 2] = (down[x] - up[x]) * (h > 1 ? sy : 0.0F);
      }
    }
  });
  return out;
}

void refine_depth_map(DepthMap& map, const RefinementImage& base, std::span<const RefinementImage* const> views,
                      const RefinementOptions& options, RefinementStats* stats) {
  RefinementStats local;
  RefinementStats& st = stats != nullptr ? *stats : local;
  st = RefinementStats{};
  if (options.radius < 1 || options.step < 1 || options.iterations < 0) {
    throw_error("refinement: invalid window (radius {}, step {}) or iterations {}", options.radius, options.step,
                options.iterations);
  }
  if (base.data.width() != map.width() || base.data.height() != map.height() || base.data.channels() != 3) {
    throw_error("refinement: base image does not match the depth map");
  }
  std::vector<Sample> samples;
  for (int dy = -options.radius; dy <= options.radius; dy += options.step) {
    for (int dx = -options.radius; dx <= options.radius; dx += options.step) {
      samples.push_back({static_cast<double>(dx), static_cast<double>(dy), static_cast<double>(dx) / options.radius,
                         static_cast<double>(dy) / options.radius});
    }
  }
  if (samples.size() > static_cast<std::size_t>(kMaxSamples)) {
    throw_error("refinement: window of {} samples (at most {})", samples.size(), kMaxSamples);
  }

  const PinholeCamera& cam0 = map.camera;
  const Eigen::Matrix3d k0_inv = cam0.intrinsics.K().inverse();
  std::vector<ViewGeometry> geos;
  for (const RefinementImage* view : views) {
    if (static_cast<int>(geos.size()) >= std::min(options.max_views, kMaxViews)) {
      break;
    }
    const PinholeCamera& cj = view->camera;
    ViewGeometry g;
    g.m = cj.intrinsics.K() * cj.pose.rotation * cam0.pose.rotation.transpose() * k0_inv;
    g.b = cj.intrinsics.K() * cj.pose.rotation * (cam0.pose.center - cj.pose.center);
    g.data = &view->data;
    geos.push_back(g);
  }
  const auto num_views = static_cast<int>(geos.size());
  const int r = options.radius;
  const int w = map.width();
  const int h = map.height();
  const ImageF32 input = map.depth;  // slopes and validity from the unrefined map

  std::atomic<std::size_t> n_input{0}, n_refined{0};
  parallel_for(0, static_cast<std::size_t>(h), 4, [&](std::size_t begin, std::size_t end) {
    std::size_t c_input = 0, c_refined = 0;
    std::array<double, kMaxSamples> n0{};
    std::array<int, kMaxViews> used{};
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      for (int x = 0; x < w; ++x) {
        const float z0 = input(x, y);
        if (!(z0 > 0.0F)) {
          continue;
        }
        ++c_input;
        map.confidence(x, y) = 0.0F;
        if (num_views == 0 || x < r || y < r || x >= w - r || y >= h - r) {
          continue;
        }
        // Normalised base window.
        double sum = 0.0;
        const auto n = static_cast<int>(samples.size());
        for (int s = 0; s < n; ++s) {
          const Sample& sm = samples[static_cast<std::size_t>(s)];
          n0[static_cast<std::size_t>(s)] = base.data(x + static_cast<int>(sm.dx), y + static_cast<int>(sm.dy), 0);
          sum += n0[static_cast<std::size_t>(s)];
        }
        const double mean = sum / n;
        double norm2 = 0.0;
        for (int s = 0; s < n; ++s) {
          n0[static_cast<std::size_t>(s)] -= mean;
          norm2 += n0[static_cast<std::size_t>(s)] * n0[static_cast<std::size_t>(s)];
        }
        if (norm2 < options.min_texture * options.min_texture * n) {
          continue;
        }
        const double inv_norm = 1.0 / std::sqrt(norm2);
        for (int s = 0; s < n; ++s) {
          n0[static_cast<std::size_t>(s)] *= inv_norm;
        }

        const double w0 = 1.0 / z0;
        const Eigen::Vector3d theta0(w0, r * initial_slope(input, x, y, 1, 0), r * initial_slope(input, x, y, 0, 1));
        const auto xd = static_cast<double>(x);
        const auto yd = static_cast<double>(y);

        // Local view selection at the initial plane.
        int m = 0;
        double s0 = 0.0;
        for (int j = 0; j < num_views; ++j) {
          const ViewResult vr =
              evaluate_view(geos[static_cast<std::size_t>(j)], xd, yd, theta0, samples, n0.data(), false);
          if (vr.valid && vr.ncc >= options.min_ncc) {
            used[static_cast<std::size_t>(m++)] = j;
            s0 += vr.ncc;
          }
        }
        if (m == 0) {
          continue;
        }
        s0 /= m;
        map.confidence(x, y) = static_cast<float>(s0);
        double max_motion = 0.0;  // pixels per unit inverse depth, largest over the used views
        for (int t = 0; t < m; ++t) {
          max_motion =
              std::max(max_motion, centre_motion(geos[static_cast<std::size_t>(used[static_cast<std::size_t>(t)])],
                                                 xd, yd, w0));
        }

        // Mean NCC and Gauss-Newton terms over the used views; -inf if a view drops out.
        auto evaluate = [&](const Eigen::Vector3d& theta, Eigen::Matrix3d& hh, Eigen::Vector3d& gg) {
          hh.setZero();
          gg.setZero();
          double score = 0.0;
          for (int t = 0; t < m; ++t) {
            const ViewResult vr = evaluate_view(geos[static_cast<std::size_t>(used[static_cast<std::size_t>(t)])], xd,
                                                yd, theta, samples, n0.data(), true);
            if (!vr.valid) {
              return -std::numeric_limits<double>::infinity();
            }
            score += vr.ncc;
            hh += vr.h;
            gg += vr.g;
          }
          return score / m;
        };

        Eigen::Matrix3d hh;
        Eigen::Vector3d gg;
        Eigen::Vector3d best = theta0;
        double best_score = evaluate(theta0, hh, gg);
        if (!std::isfinite(best_score)) {
          continue;
        }
        double scale = 1.0;
        for (int it = 0; it < options.iterations; ++it) {
          Eigen::Matrix3d damped = hh;
          damped.diagonal() *= 1.0 + 1e-6;
          damped.diagonal().array() += 1e-12;
          Eigen::Vector3d delta = -damped.ldlt().solve(gg);
          if (!delta.allFinite()) {
            break;
          }
          // At most one pixel of motion per step at the window border.
          const double motion = max_motion * (std::abs(delta[0]) + std::abs(delta[1]) + std::abs(delta[2]));
          if (motion > 1.0) {
            delta /= motion;
          }
          delta *= scale;
          if (max_motion * delta.cwiseAbs().sum() < 1e-3) {
            break;
          }
          Eigen::Matrix3d h2;
          Eigen::Vector3d g2;
          const Eigen::Vector3d cand = best + delta;
          const double score = evaluate(cand, h2, g2);
          if (score > best_score) {
            best = cand;
            best_score = score;
            hh = h2;
            gg = g2;
            scale = 1.0;
          } else {
            scale *= 0.25;
          }
        }
        if (best_score > s0 + options.min_gain && best[0] > 0.0 && max_motion * std::abs(best[0] - w0) <= options.max_shift_px) {
          map.depth(x, y) = static_cast<float>(1.0 / best[0]);
          map.confidence(x, y) = static_cast<float>(best_score);
          ++c_refined;
        }
      }
    }
    n_input += c_input;
    n_refined += c_refined;
  });
  st.input = n_input;
  st.refined = n_refined;
  st.kept = st.input - st.refined;
}

}  // namespace r3d
