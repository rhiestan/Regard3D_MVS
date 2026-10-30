# R3DMVS

Multi-view stereo densification for [OpenMVG](https://github.com/openMVG/openMVG) scenes, developed for
[Regard3D](https://www.regard3d.org). It takes an OpenMVG scene (`sfm_data.bin` or `.json`) and produces a dense, oriented,
scale-annotated point cloud, which is meant to be meshed and textured with external tools (PoissonRecon,
FSSR, texrecon).

Status: **Phase 8** (library API and Regard3D integration). Done so far:

- Phase 1: OpenMVG `sfm_data.json` / `.bin` and COLMAP text import, OpenMVG camera models, undistortion,
  image pyramids, stereo pair selection, PLY exports for visual checks.
- Phase 2: calibrated rectification, 9x7 Census, 8-path SGM over the full disparity range with tiling,
  sub-pixel disparities, left-right check and filters (SIMD kernels bit-exact with scalar), Middlebury
  evaluation, and one point cloud per stereo pair.
- Phase 3: hierarchical matching with per-pixel disparity ranges from the coarser level, full
  right-to-left consistency check, tiles admitted by a memory budget (`--memory-budget`); 20+ MP pairs
  with thousands of disparities in seconds.
- Phase 4: `r3dmvs dense`: per-view multi-stereo depth maps from up to 4 pairs, multi-view geometric
  consistency, voxel fusion into a coloured PLY with normals and point scales. ETH3D training, half
  resolution, F1 at 2 cm: courtyard 0.80, office 0.64, pipes 0.61, terrains 0.82 (COLMAP 4.2 at the same
  resolution: 0.77, 0.37, 0.38, 0.72).
- Phase 5: patch-based refinement of every depth as a plane (depth and orientation) by Gauss-Newton on
  the normalised cross-correlation with the neighbouring images, per-pixel view selection, NCC
  confidence. ETH3D F1 at 2 cm: courtyard 0.81, office 0.64, pipes 0.61, terrains 0.82.
- Phase 6: visibility filter against the neighbouring views, octree fusion with leaves of one GSD
  (best resolution wins, optional fold validation `--min-sources 2`, density `--voxel-factor`), and a
  pipeline within a memory budget (`--memory`, `--fusion-memory`; depth maps that do not fit wait in the
  work directory). ETH3D F1 at 2 cm: courtyard 0.80, office 0.64, pipes 0.62, terrains 0.82 (0.85 with
  `--min-sources 2`).
- Phase 7: point layouts for PoissonRecon and FSSR (`--for`), texrecon scene export, the cloud streamed
  to disk while it is fused, and `tools/r3d_mesh.py` (cloud -> Poisson or FSSR -> trimmed mesh ->
  texrecon). Mesh F1 at 2 cm with Poisson: courtyard 0.78, office 0.63, pipes 0.57, terrains 0.81, as good
  as or better than OpenMVS's full pipeline on the same poses (0.78, 0.43, 0.55, 0.70), so there is no
  built-in mesher (D43).
- Phase 8: the pipeline as a library function (`r3d::densify`) with progress callbacks and
  cancellation, presets `fast` / `normal` / `high` / `ultra`, and machine-readable progress and clean
  cancellation (Ctrl+C, SIGTERM, Ctrl+Break) for programs that run `r3dmvs` as a child process.
  Fusion by depth precision (points from small triangulation angles get coarser leaves): ETH3D F1 at 2 cm
  with `normal`: courtyard 0.82, office 0.68, pipes 0.64, terrains 0.84. Depths that no other image
  confirms are removed, and so are halos (depths that extend a surface over the plain background next to
  it); `--weak-segment 0.001` also removes small islands without texture (sky, dark background),
  `--reject-weak` every depth without texture (D51, D52).

Design and IP decisions are logged in [docs/DECISIONS.md](docs/DECISIONS.md). Using R3DMVS from another
program, as a library or as a child process: [docs/API.md](docs/API.md).

```sh
# Summary, stereo pairs, sparse cloud, camera frustums and undistorted images of a scene:
r3dmvs scene path/to/sfm_data.bin --pairs --sparse-ply sparse.ply --colors --cameras-ply cameras.ply \
       --undistort-dir undistorted

# Rectify and match every view with its best partner; rectified images, disparity maps and a
# coloured point cloud per pair:
r3dmvs pair path/to/sfm_data.bin --all --out-dir pairs

# Dense point cloud with a preset (fast, normal (default), high, ultra); options given as well
# override it. Depth maps are cached in the work directory; --reuse skips matching, --reuse-refined
# also the refinement. --progress prints "progress <percent> <stage> <done>/<total>" lines:
r3dmvs dense path/to/sfm_data.bin --preset normal --work-dir work --progress -o dense.ply
# Sparser and cleaner: leaves of two GSD, points seen by two views:
r3dmvs dense path/to/sfm_data.bin --work-dir work --reuse-refined --voxel-factor 2 --min-sources 2 \
       -o dense_clean.ply

# Point attributes for a mesher (--for poisson / fssr, also on 'dense'), and the scene for texrecon:
r3dmvs export --cloud dense.ply --for poisson -o poisson_in.ply
r3dmvs export path/to/sfm_data.bin --texrecon-dir texrecon
# Mesh and texture end to end (PoissonRecon or FSSR, then texrecon), see tools/README.md:
python tools/r3d_mesh.py path/to/sfm_data.bin --out-dir mesh --tools path/to/external_tools -- --voxel-factor 2

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
