# Compiler options applied to R3DMVS's own targets (never to third-party code).

if(R3D_SANITIZERS)
  if(MSVC)
    message(FATAL_ERROR "R3D_SANITIZERS is only supported with GCC/Clang")
  endif()
  list(JOIN R3D_SANITIZERS "," _r3d_san)
  add_compile_options(-fsanitize=${_r3d_san} -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=${_r3d_san})
endif()

function(r3d_set_target_options target)
  if(MSVC)
    # Covers both cl.exe and clang-cl.
    target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8 /Zc:__cplusplus)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options(${target} PRIVATE /Zc:preprocessor)  # cl.exe only
    endif()
    target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    if(R3D_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast
      -Wcast-align -Woverloaded-virtual -Wnull-dereference -Wimplicit-fallthrough)
    if(R3D_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()
