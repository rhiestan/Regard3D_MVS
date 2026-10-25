cmake_minimum_required(VERSION 3.25)

project(R3DMVS
  VERSION 0.1.0
  DESCRIPTION "Regard3D multi-view stereo densification"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_property(R3D_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT R3D_IS_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(R3D_BUILD_TESTS "Build unit tests" ${PROJECT_IS_TOP_LEVEL})
option(R3D_BUILD_APPS "Build command line applications" ${PROJECT_IS_TOP_LEVEL})
option(R3D_BUILD_BENCH "Build micro-benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(R3D_WARNINGS_AS_ERRORS "Treat compiler warnings as errors (own targets only)" OFF)
option(R3D_USE_SYSTEM_DEPS "Prefer installed dependencies (find_package) over FetchContent" OFF)
set(R3D_SANITIZERS "" CACHE STRING "Semicolon-separated sanitizers for GCC/Clang, e.g. address;undefined")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(R3DCompilerOptions)
include(R3DDependencies)

add_subdirectory(src)

if(R3D_BUILD_APPS)
  add_subdirectory(apps)
endif()

if(R3D_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(R3D_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
