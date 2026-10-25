# R3DMVS

Multi-view stereo densification for [OpenMVG](https://github.com/openMVG/openMVG) scenes, developed for
[Regard3D](https://www.regard3d.org). It takes an OpenMVG scene (`sfm_data.bin` or `.json`) and produces a dense, oriented,
scale-annotated point cloud, which is meant to be meshed and textured with external tools (PoissonRecon,
FSSR, texrecon).

Status: **Phase 1** (scene import and pair selection): OpenMVG `sfm_data.json` / `.bin` and COLMAP text
import, OpenMVG camera models, undistortion, image pyramids, stereo pair selection, PLY exports for visual
checks. Design and IP decisions are logged in [docs/DECISIONS.md](docs/DECISIONS.md).

```sh
# Summary, stereo pairs, sparse cloud, camera frustums and undistorted images of a scene:
r3dmvs scene path/to/sfm_data.bin --pairs --sparse-ply sparse.ply --colors --cameras-ply cameras.ply \
       --undistort-dir undistorted
```

Images are looked up under `root_path`, then next to the `sfm_data` file; `--image-dir` overrides both.

## Building

Requirements: CMake ≥ 3.25, Ninja, and a C++20 compiler with `<format>`:
MSVC 19.30+ (VS 2022), clang-cl, GCC 13+, Clang 17+, or AppleClang 15+ (macOS 13.3+).
Dependencies are downloaded automatically at configure time (pinned versions, SHA256-verified).
Tests on real scenes run when `R3D_TESTDATA_DIR` (default `../testdata`) contains `<scene>/sfm_data.bin`.

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
