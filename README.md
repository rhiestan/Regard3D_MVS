# R3DMVS

Multi-view stereo densification for [OpenMVG](https://github.com/openMVG/openMVG) scenes, developed for
[Regard3D](https://www.regard3d.org). It takes an OpenMVG scene (`sfm_data.bin` or `.json`) and produces a dense, oriented,
scale-annotated point cloud, which is meant to be meshed and textured with external tools (PoissonRecon,
FSSR, texrecon).

Status: **Phase 2** (two-view Census-SGM). Done so far:

- Phase 1: OpenMVG `sfm_data.json` / `.bin` and COLMAP text import, OpenMVG camera models, undistortion,
  image pyramids, stereo pair selection, PLY exports for visual checks.
- Phase 2: calibrated rectification, 9x7 Census, 8-path SGM over the full disparity range with tiling,
  sub-pixel disparities, left-right check and filters (SIMD kernels bit-exact with scalar), Middlebury
  evaluation, and one point cloud per stereo pair.

Design and IP decisions are logged in [docs/DECISIONS.md](docs/DECISIONS.md).

```sh
# Summary, stereo pairs, sparse cloud, camera frustums and undistorted images of a scene:
r3dmvs scene path/to/sfm_data.bin --pairs --sparse-ply sparse.ply --colors --cameras-ply cameras.ply \
       --undistort-dir undistorted

# Rectify and match every view with its best partner; rectified images, disparity maps and a
# coloured point cloud per pair:
r3dmvs pair path/to/sfm_data.bin --all --out-dir pairs

# Match a rectified pair, or score the Middlebury 2014 training set (MiddEval3 layout):
r3dmvs stereo path/to/MiddEval3/trainingH
```

Images are looked up under `root_path`, then next to the `sfm_data` file; `--image-dir` overrides both.
Matcher throughput is measured with `bench_stereo` ([bench/README.md](bench/README.md)).

## Building

Requirements: CMake ≥ 3.25, Ninja, and a C++20 compiler with `<format>`:
MSVC 19.30+ (VS 2022), clang-cl, GCC 13+, Clang 17+, or AppleClang 15+ (macOS 13.3+).
Dependencies are downloaded automatically at configure time (pinned versions, SHA256-verified).
Tests on real scenes run when `R3D_TESTDATA_DIR` (default `../testdata`) contains `<scene>/sfm_data.bin`.
Middlebury 2014 data for `r3dmvs stereo` is available at <https://vision.middlebury.edu/stereo/submit3/>
(`MiddEval3-data-*.zip` and `MiddEval3-GT0-*.zip`).

```sh
cmake --preset gcc-release          # or: msvc-release, clang-cl-release, clang-release, macos-release
cmake --build --preset gcc-release
ctest --preset gcc-release
./build/gcc-release/apps/r3dmvs info
```

On Windows, run these from a Visual Studio developer prompt. Use `-DR3D_USE_SYSTEM_DEPS=ON` to prefer
installed packages (e.g. from vcpkg or a Linux distribution) over downloads.

## Dependencies

All permissively licensed: Eigen (MPL-2.0, built with `EIGEN_MPL2_ONLY`), Google Highway (Apache-2.0),
nlohmann/json (MIT), CLI11 (BSD-3), stb_image (public domain / MIT), doctest (MIT, tests only).
