##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
##===----------------------------------------------------------------------===##
#
# hcc2-hip/libdevice/CMakeLists.txt
#
##===----------------------------------------------------------------------===##

#  Try to get LLVM_COMPILER from HIP, then HCC2 , then default /usr/local/hip
set(HIP $ENV{HIP})
if(HIP)
  set(LLVM_COMPILER ${HIP})
else()
  set(HCC2 $ENV{HCC2})
  if(HCC2)
    set(LLVM_COMPILER ${HCC2})
  else()
    set(LLVM_COMPILER "/usr/local/hip")
  endif()
endif()

# Assome rocm-device-libs repository is next to hcc2-hip repository 
set(ROCDL ${CMAKE_CURRENT_SOURCE_DIR}/../../rocm-device-libs)
set(ROCDL_INC_OCKL ${ROCDL}/ockl/inc)
set(ROCDL_INC_OCML ${ROCDL}/ocml/inc)
set(ROCDL_INC_IRIF ${ROCDL}/irif/inc)

set(libname "hiprt")
message("---> Building lib${libname} with Compiler ${LLVM_COMPILER}")
project(${libname})

# mcpus is a list of all gfx processors to build an optimized library for 
set(mcpus $ENV{GFXLIST})
if(NOT mcpus)
  set(mcpus gfx700 gfx701 gfx801 gfx803 gfx900)
endif()

separate_arguments(mcpus)

message("---> Building lib${libname} for gfx processors: ${mcpus}")

file(GLOB h_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
file(GLOB ll_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.ll")
file(GLOB hip_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB ocl_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cl")

#  OpenCL files are no longer gfx specific, so only build them one time
set(cl_cmd ${LLVM_COMPILER}/bin/clang
    -S -emit-llvm
    -DCL_VERSION_2_0=200 -D__OPENCL_C_VERSION__=200
    -Dcl_khr_fp64 -Dcl_khr_fp16
    -Dcl_khr_subgroups -Dcl_khr_int64_base_atomics -Dcl_khr_int64_extended_atomics
    -x cl -Xclang -cl-std=CL2.0 -Xclang -finclude-default-header
    -target amdgcn-amd-amdhsa
    -I${ROCDL_INC_OCKL}
    -I${ROCDL_INC_OCML}
    -I${ROCDL_INC_IRIF}
    -I${CMAKE_CURRENT_SOURCE_DIR}/src
    -I${LLVM_COMPILER}/include)
set(cl_ll_files)
foreach(file ${ocl_sources})
  file(RELATIVE_PATH rfile ${CMAKE_CURRENT_SOURCE_DIR}/src ${file})
  get_filename_component(fname ${rfile} NAME_WE)
  set(ll_filename ${fname}.ll)
  add_custom_command(
    OUTPUT ${ll_filename}
    COMMAND ${cl_cmd} ${file} -o ${ll_filename} 
    DEPENDS ${file})
  list(APPEND cl_ll_files ${ll_filename})
endforeach()

foreach(mcpu ${mcpus})

  set(hip_cmd ${LLVM_COMPILER}/bin/clang++ -x hip --offload-arch=${mcpu} -c -S -emit-llvm -O2 -nocudalib
    -I${CMAKE_CURRENT_SOURCE_DIR}/../include 
    -I${CMAKE_CURRENT_SOURCE_DIR}/src)
  set(hip_ll_files)
  foreach(file ${hip_sources})
    file(RELATIVE_PATH rfile ${CMAKE_CURRENT_SOURCE_DIR}/src ${file})
    get_filename_component(fname ${rfile} NAME_WE)
#   FIXME after we get better file name for device ll
    set(ll_filename ${fname}-cuda-nvptx64-nvidia-cuda-${mcpu}.ll)
    add_custom_command(
      OUTPUT ${ll_filename}
      COMMAND ${hip_cmd} ${file} 
      DEPENDS ${file})
    list(APPEND hip_ll_files ${ll_filename})
  endforeach()

  add_custom_command(
    OUTPUT linkout.${mcpu}.bc
    COMMAND ${LLVM_COMPILER}/bin/llvm-link ${cl_ll_files} ${hip_ll_files} ${ll_sources} -o linkout.${mcpu}.bc
    DEPENDS ${cl_ll_files} ${hip_ll_files} ${ll_sources})
  add_custom_target(linkout-${mcpu}-bc  ALL DEPENDS linkout.${mcpu}.bc)

  set(final_bc_filename lib${libname}-${mcpu}.bc)
  add_custom_command(
    OUTPUT ${final_bc_filename}
    COMMAND ${LLVM_COMPILER}/bin/prepare-builtins linkout.${mcpu}.bc -o ${final_bc_filename}
    DEPENDS linkout.${mcpu}.bc )
  add_custom_target(lib${libname}-${mcpu} ALL DEPENDS ${final_bc_filename})

  if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(OUTPUTDIR ${CMAKE_CURRENT_BINARY_DIR})
  else()
    set(OUTPUTDIR ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  endif()

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    install(FILES ${OUTPUTDIR}/${final_bc_filename} DESTINATION "lib-debug/libdevice/${mcpu}"
      RENAME lib${libname}.bc)
  else()
    install(FILES ${OUTPUTDIR}/${final_bc_filename} DESTINATION "lib/libdevice/${mcpu}"
      RENAME lib${libname}.bc)
  endif()

endforeach()
