add_executable(r3dmvs r3dmvs/main.cpp r3dmvs/dense_command.cpp r3dmvs/stereo_commands.cpp)
target_link_libraries(r3dmvs PRIVATE R3D::r3d CLI11::CLI11)
r3d_set_target_options(r3dmvs)
