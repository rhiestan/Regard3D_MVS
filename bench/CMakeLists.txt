add_executable(bench_stereo bench_stereo.cpp)
target_link_libraries(bench_stereo PRIVATE R3D::r3d)
r3d_set_target_options(bench_stereo)
