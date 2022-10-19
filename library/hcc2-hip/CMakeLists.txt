# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##===----------------------------------------------------------------------===##
#
# Build offloading library libhiprt.so.
#
##===----------------------------------------------------------------------===##

# CMAKE libhiprt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# Add cmake directory to search for custom cmake functions.
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})

# Standalone build or part of LLVM?
set(LIBHIPRT_STANDALONE_BUILD FALSE)
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}" OR
   "${CMAKE_SOURCE_DIR}/libhiprt" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  project(libhiprt C CXX)
  set(LIBHIPRT_STANDALONE_BUILD TRUE)
endif()

if(${LIBHIPRT_STANDALONE_BUILD})
  set(LIBHIPRT_ENABLE_WERROR FALSE CACHE BOOL
    "Enable -Werror flags to turn warnings into errors for supporting compilers.")
  # CMAKE_BUILD_TYPE was not defined, set default to Release
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
else()
  set(LIBOMP_ENABLE_WERROR ${LLVM_ENABLE_WERROR})
endif()

# Compiler flag checks.
include(config-ix)

# Message utilities.
include(LibhiprtUtils)

# Get dependencies for the different components of the project.
include(LibhiprtGetDependencies)

# This is a list of all the targets that are supported/tested right now.
set (LIBHIPRT_ALL_TARGETS "${LIBHIPRT_ALL_TARGETS} amdgcn-amd-hsa")

# Once the plugins for the different targets are validated, they will be added to
# the list of supported targets in the current system.
set (LIBHIPRT_SYSTEM_TARGETS "")

# Set base directories - required for lit to locate the tests.
set(LIBHIPRT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBHIPRT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# We need C++11 support.
if(LIBHIPRT_HAVE_STD_CPP11_FLAG)

  libhiprt_say("Building offloading runtime library libhiprt.")

  # Enable support for C++11.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

  if(LIBHIPRT_ENABLE_WERROR AND LIBHIPRT_HAVE_WERROR_FLAG)
    add_definitions(-Werror)
  endif()

  # If building this library in debug mode, we define a macro to enable
  # dumping progress messages at runtime.
  string( TOLOWER "${CMAKE_BUILD_TYPE}" LIBHIPRT_CMAKE_BUILD_TYPE)
  if(LIBHIPRT_CMAKE_BUILD_TYPE MATCHES debug)
    add_definitions(-DHIPRT_DEBUG)
    add_definitions(-g)
    add_definitions(-O0)
  endif()

  set(src_files
    lib/src/hipLaunchByPtr.cpp
  )

include_directories(
  lib/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  /opt/rocm/include
  )

# Turn on HIP_PLATFORM_CLANG, to build runtime
  add_compile_options(-D__HIPCC__)

  # Build libhiprt library with libdl dependency.
  add_library(hiprt SHARED ${src_files})
  target_link_libraries(hiprt
    dl
    "-lstdc++ -L/opt/rocm/lib -lhip_hcc -lhc_am -Wl,-soname,libhiprt.so")

  # Install libhiprt under the lib destination folder.
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    install(TARGETS hiprt LIBRARY DESTINATION "lib-debug")
  else()
    install(TARGETS hiprt LIBRARY DESTINATION "lib")
  endif()

  install(DIRECTORY include/hip DESTINATION lib/clang/7.0.0/include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
  install(DIRECTORY include/hip2 DESTINATION lib/clang/7.0.0/include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

  # Add tests. (no tests yet)
  # add_subdirectory(test)

else(LIBHIPRT_HAVE_STD_CPP11_FLAG)
  libhiprt_say("Not building offloading runtime library libhiprt: host compiler must have c++11 support.")
endif(LIBHIPRT_HAVE_STD_CPP11_FLAG)

add_subdirectory(libdevice)
