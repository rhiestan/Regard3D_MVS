add_executable(densify_example densify_example.cpp)
target_link_libraries(densify_example PRIVATE R3D::r3d)
r3d_set_target_options(densify_example)
