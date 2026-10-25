# Third-party dependencies. All permissively licensed (see docs/DECISIONS.md).
#
# Every dependency is pinned to an exact release with a SHA256 hash. With
# R3D_USE_SYSTEM_DEPS=ON an installed package is used when found.
# Header-only libraries are wrapped in INTERFACE targets instead of running
# their own CMake scripts, which keeps configure fast and avoids policy issues.

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# Helper: download a header-only archive and expose `include_subdir` as a
# SYSTEM include directory of INTERFACE target `target`.
function(r3d_fetch_header_only name url sha256 include_subdir target)
  FetchContent_Declare(${name} URL ${url} URL_HASH SHA256=${sha256}
    DOWNLOAD_EXTRACT_TIMESTAMP ON SOURCE_SUBDIR _r3d_no_cmake_)
  FetchContent_MakeAvailable(${name})
  add_library(${target} INTERFACE)
  target_include_directories(${target} SYSTEM INTERFACE "${${name}_SOURCE_DIR}/${include_subdir}")
  set(${name}_SOURCE_DIR "${${name}_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

# --- Eigen 3.4.0 (MPL-2.0) ---------------------------------------------------
if(R3D_USE_SYSTEM_DEPS)
  find_package(Eigen3 3.4 CONFIG QUIET)
endif()
if(NOT TARGET Eigen3::Eigen)
  r3d_fetch_header_only(r3d_eigen
    https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    "" r3d_eigen_headers)
  add_library(Eigen3::Eigen ALIAS r3d_eigen_headers)
endif()

# --- nlohmann/json 3.11.3 (MIT) ---------------------------------------------
if(R3D_USE_SYSTEM_DEPS)
  find_package(nlohmann_json 3.11 CONFIG QUIET)
endif()
if(NOT TARGET nlohmann_json::nlohmann_json)
  r3d_fetch_header_only(r3d_json
    https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    "include" r3d_json_headers)
  add_library(nlohmann_json::nlohmann_json ALIAS r3d_json_headers)
endif()

# --- CLI11 2.4.2 (BSD-3-Clause) ----------------------------------------------
if(R3D_USE_SYSTEM_DEPS)
  find_package(CLI11 2.4 CONFIG QUIET)
endif()
if(NOT TARGET CLI11::CLI11)
  r3d_fetch_header_only(r3d_cli11
    https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.tar.gz
    f2d893a65c3b1324c50d4e682c0cdc021dd0477ae2c048544f39eed6654b699a
    "include" r3d_cli11_headers)
  add_library(CLI11::CLI11 ALIAS r3d_cli11_headers)
endif()

# --- stb (public domain / MIT), pinned commit 2026-08-01 ------------------------
# Only stb_image.h (decoding) and the zlib encoder of stb_image_write.h are used.
r3d_fetch_header_only(r3d_stb
  https://github.com/nothings/stb/archive/2c980bb59875b0d32144a71867fbdebb2f77cd20.tar.gz
  9a955b1b49a4410088a2e0ee2a9c057c3c907d0c1d75454144cb980aca0ba515
  "" r3d_stb_headers)

# --- Google Highway 1.2.0 (Apache-2.0 / BSD-3-Clause) -----------------------
if(R3D_USE_SYSTEM_DEPS)
  find_package(hwy 1.2 CONFIG QUIET)
endif()
if(NOT TARGET hwy::hwy)
  set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(HWY_FORCE_STATIC_LIBS ON CACHE BOOL "" FORCE)
  FetchContent_Declare(r3d_highway
    URL https://github.com/google/highway/archive/refs/tags/1.2.0.tar.gz
    URL_HASH SHA256=7e0be78b8318e8bdbf6fa545d2ecb4c90f947df03f7aadc42c1967f019e63343
    DOWNLOAD_EXTRACT_TIMESTAMP ON SYSTEM)
  FetchContent_MakeAvailable(r3d_highway)
  add_library(hwy::hwy ALIAS hwy)
endif()

# --- doctest 2.5.3 (MIT), tests only ------------------------------------------
if(R3D_BUILD_TESTS)
  if(R3D_USE_SYSTEM_DEPS)
    find_package(doctest 2.4 CONFIG QUIET)
  endif()
  if(TARGET doctest::doctest)
    list(APPEND CMAKE_MODULE_PATH "${doctest_DIR}")
  else()
    r3d_fetch_header_only(r3d_doctest
      https://github.com/doctest/doctest/archive/refs/tags/v2.5.3.tar.gz
      174ebc4e769928959614789c5b4e9c3d0a0f81a62bb608756b127bfebfb21331
      "" r3d_doctest_headers)
    add_library(doctest::doctest ALIAS r3d_doctest_headers)
    list(APPEND CMAKE_MODULE_PATH "${r3d_doctest_SOURCE_DIR}/scripts/cmake")
  endif()
endif()
