# R3DMVS

Multi-view stereo densification for [OpenMVG](https://github.com/openMVG/openMVG) scenes, developed for
[Regard3D](https://www.regard3d.org). It takes an OpenMVG `sfm_data.json` and produces a dense, oriented,
scale-annotated point cloud, which is meant to be meshed and textured with external tools (PoissonRecon,
FSSR, texrecon).

Status: **Phase 0** (foundation): core types, image and PLY I/O, thread pool, SIMD dispatch, CI.
See [plan.md](plan.md) for the roadmap and [docs/DECISIONS.md](docs/DECISIONS.md) for design and IP decisions.

## Building

Requirements: CMake ≥ 3.25, Ninja, and a C++20 compiler with `<format>`:
MSVC 19.30+ (VS 2022), clang-cl, GCC 13+, Clang 17+, or AppleClang 15+ (macOS 13.3+).
Dependencies are downloaded automatically at configure time (pinned versions, SHA256-verified).

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
