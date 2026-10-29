configure_file(r3d/version.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/generated/r3d/version.hpp" @ONLY)

add_library(r3d STATIC
  r3d/core/camera_model.cpp
  r3d/core/color_convert.cpp
  r3d/core/log.cpp
  r3d/core/memory.cpp
  r3d/core/pyramid.cpp
  r3d/core/simd.cpp
  r3d/core/thread_pool.cpp
  r3d/io/colmap_scene.cpp
  r3d/io/image_io.cpp
  r3d/io/mesher_export.cpp
  r3d/io/openmvg_scene.cpp
  r3d/io/pfm.cpp
  r3d/io/ply.cpp
  r3d/io/scene_export.cpp
  r3d/io/stb_impl.cpp
  r3d/mvs/consistency.cpp
  r3d/mvs/visibility.cpp
  r3d/mvs/depth_map.cpp
  r3d/mvs/depth_map_store.cpp
  r3d/mvs/fusion.cpp
  r3d/mvs/multi_stereo.cpp
  r3d/mvs/refinement.cpp
  r3d/rectify/rectification.cpp
  r3d/scene/scene.cpp
  r3d/scene/undistort.cpp
  r3d/select/view_selection.cpp
  r3d/stereo/census.cpp
  r3d/stereo/disparity_eval.cpp
  r3d/stereo/pair_cloud.cpp
  r3d/stereo/sgm_kernels.cpp
  r3d/stereo/stereo_matcher.cpp
)
add_library(R3D::r3d ALIAS r3d)

target_include_directories(r3d PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/generated")

target_link_libraries(r3d
  PUBLIC Eigen3::Eigen hwy::hwy
  PRIVATE r3d_stb_headers nlohmann_json::nlohmann_json)

# Eigen: refuse any non-MPL2 (LGPL) code paths at compile time.
target_compile_definitions(r3d PUBLIC EIGEN_MPL2_ONLY)

find_package(Threads REQUIRED)
target_link_libraries(r3d PUBLIC Threads::Threads)

r3d_set_target_options(r3d)
