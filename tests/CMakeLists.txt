add_executable(r3d_tests
  test_main.cpp
  test_camera.cpp
  test_camera_model.cpp
  test_census.cpp
  test_colmap_scene.cpp
  test_color_convert.cpp
  test_image.cpp
  test_image_io.cpp
  test_mesher_export.cpp
  test_mvs.cpp
  test_openmvg_scene.cpp
  test_ply.cpp
  test_pyramid.cpp
  test_rectification.cpp
  test_stereo.cpp
  test_thread_pool.cpp
  test_undistort.cpp
  test_view_selection.cpp
)
target_link_libraries(r3d_tests PRIVATE R3D::r3d doctest::doctest nlohmann_json::nlohmann_json)
r3d_set_target_options(r3d_tests)

# Optional real scenes (not in the repository): <dir>/<scene>/sfm_data.bin + images.
set(R3D_TESTDATA_DIR "${PROJECT_SOURCE_DIR}/../testdata" CACHE PATH "Directory with optional test scenes")
if(IS_DIRECTORY "${R3D_TESTDATA_DIR}")
  cmake_path(ABSOLUTE_PATH R3D_TESTDATA_DIR NORMALIZE OUTPUT_VARIABLE _r3d_testdata)
  target_compile_definitions(r3d_tests PRIVATE R3D_TESTDATA_DIR="${_r3d_testdata}")
endif()

# One CTest test per doctest test case; the timeout keeps a hung test from blocking CI.
include(doctest)
doctest_discover_tests(r3d_tests PROPERTIES TIMEOUT 300)
