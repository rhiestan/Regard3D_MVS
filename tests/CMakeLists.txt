add_executable(r3d_tests
  test_main.cpp
  test_camera.cpp
  test_color_convert.cpp
  test_image.cpp
  test_image_io.cpp
  test_ply.cpp
  test_thread_pool.cpp
)
target_link_libraries(r3d_tests PRIVATE R3D::r3d doctest::doctest)
r3d_set_target_options(r3d_tests)

# One CTest test per doctest test case; the timeout keeps a hung test from blocking CI.
include(doctest)
doctest_discover_tests(r3d_tests PROPERTIES TIMEOUT 300)
