# Design and IP decision log

Every design decision with IP, licensing or architectural impact is recorded here with its date, so that
the origin of each technique can be traced later. Algorithms are implemented clean-room from the
published papers listed in `plan.md` §1; no code is copied from patented or copyleft implementations.

*Not legal advice. Items marked OPEN must be checked against actual patent claims before the related
code is released.*

## Decisions

| # | Date | Decision | Rationale / source |
|---|---|---|---|
| D1 | 2026-10-24 | Primary matcher: pairwise rectified hierarchical Census-SGM followed by multi-stereo triangulation. Reference-view plane sweep is the secondary/fallback matcher. | Hirschmüller, TPAMI 2008 §2.4 (cost-level multi-view fusion is unstable under occlusion); Wenzel et al., ISPRS 2012 (SURE architecture); Hirschmüller & Scharstein, TPAMI 2009 (Census most robust). |
| D2 | 2026-10-24 | OpenMVG scenes are read by our own `sfm_data.json` parser (nlohmann/json). No link dependency on OpenMVG. The binary `.bin` format is not supported; convert with `openMVG_main_ConvertSfM_DataFormat`. | Light dependency footprint; decided by the project owner. |
| D3 | 2026-10-24 | No built-in mesher. Output is a dense point cloud with normals, scale, confidence and optional visibility, for PoissonRecon / FSSR / texrecon. Revisit only if the Phase 7 benchmark shows meshing is the bottleneck. | Decided by the project owner; see `plan.md` §3.9. |
| D4 | 2026-10-24 | Third-party code is fetched with CMake FetchContent, pinned to exact releases and verified by SHA256; `R3D_USE_SYSTEM_DEPS=ON` prefers installed packages. Header-only libraries are wrapped without running their CMake scripts. | Reproducible builds without requiring vcpkg/Conan; works identically on all CI platforms. |
| D5 | 2026-10-24 | Own thread pool (`r3d::ThreadPool`) instead of oneTBB for now. Chunk boundaries depend only on the range and grain, never on the thread count, and nested `parallel_for` runs serially. | Deterministic results, no extra binary dependency. Revisit (oneTBB, Apache-2.0) if load balancing of nested/irregular work becomes a bottleneck. |
| D6 | 2026-10-24 | Image decoding via stb_image (JPEG, PNG 8/16-bit, PNM, BMP, TGA); own PNG encoder (8/16-bit, adaptive filters) on stb's zlib compressor. | Zero native dependencies. Revisit in Phase 1 if JPEG decoding speed matters (libjpeg-turbo) or TIFF input is required (libtiff). |
| D7 | 2026-10-24 | C++20 with `std::format`. Minimum compilers: MSVC 19.30, GCC 13, Clang 17, AppleClang 15 (macOS deployment target 13.3). | `std::format` removes the need for fmt; these are the "recent compilers" targeted in the plan. |
| D8 | 2026-10-24 | Point positions are stored as float relative to a double-precision `origin`. PLY output writes either relative float coordinates plus an `r3d_origin` comment, or absolute doubles. | Halves memory for 100M+ point clouds while keeping sub-mm precision for georeferenced scenes. |
| D9 | 2026-10-24 | Eigen is compiled with `EIGEN_MPL2_ONLY`. | Rules out the few LGPL-licensed Eigen code paths at compile time. |
| D10 | 2026-10-24 | Pixel convention: integer pixel coordinates are pixel centres; (0, 0) is the centre of the top-left pixel (OpenMVG convention). The COLMAP importer will shift by −0.5. | Single convention throughout the code base. |
| D11 | 2026-10-24 | Highway kernels must be bit-exact with their scalar reference; tests run every compiled and supported target (`ScopedSimdTarget`). | Deterministic output across machines, and SIMD bugs are caught on every CI platform. |
| D12 | 2026-10-25 | Unit tests use doctest (single header, MIT) instead of Catch2. Own `main` routes MSVC CRT assertion reports to stderr; every CTest test has a 300 s timeout. | Decided by the project owner. Fast to compile, thread-safe assertions; the CRT change fixes a Debug run that blocked on an assertion dialog. |

## Avoided techniques (from the earlier design review, `input.txt`)

| Technique | Status |
|---|---|
| eSGM-style storage of per-direction minima only | Avoided: the full aggregated cost volume of each tile is stored. |
| Cost compression / disparity sub-sampling of path costs | Avoided: full label resolution, `uint16` aggregated costs. |
| Tiling that passes boundary path costs between tiles | Avoided: independent overlapping tiles with blending (TPAMI 2008 §2.6). |
| Non-uniform disparity labels | Avoided: uniform disparity (rectified) or uniform inverse depth (plane sweep). |
| PatchMatch randomised search (as in COLMAP/OpenMVS) | Avoided as the core; refinement uses deterministic least squares (Goesele et al. 2007, MPGC). |

## Open IP checks

| Item | Needed before | Notes |
|---|---|---|
| OPEN-1: Patent status of classic SGM (DLR) and its expiry date | Phase 2 release | If a patent was filed in 2005/2006, it reaches the 20-year term in 2025/2026. |
| OPEN-2: Hierarchical per-pixel disparity ranges (SURE / tSGM) vs. cost-compression claims | Phase 3 | Fallback: per-tile ranges (TPAMI 2008 §2.6). |
| OPEN-3: Census transform hardware patents (Interval Research, 1990s) | Phase 2 release | Expected to have expired; record the patent numbers and dates. |
| OPEN-4: PatchMatch patents (Adobe) | only if PatchMatch is ever considered | Not used. |
