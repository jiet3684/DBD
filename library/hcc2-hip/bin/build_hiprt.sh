#!/bin/bash
#
#  File: build_hiprt.sh
#        Build the hip host and device runtimes, 
#        The install option will install components into the hip compiler installation. 
#        The components include:
#          hip headers installed in $HIP/include/hip
#          hip host runtime installed in $HIP/lib/libhiprt.so
#          hip device runtime installed in $HIP/lib/libdevice/libhiprt.<devicetype.bc
#          Debug hip host runtime installed in $HIP/lib-debug/libhiprt.so
#          Debug hip device runtime installed in $HIP/lib/lib-debug/libdevice/libhiprt.<devicetype.bc
#
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

HIP=${HIP:-/usr/local/hip}
HIP_REPOS=${HIP_REPOS:-/home/$USER/git/hip}
BUILD_HIP=${BUILD_HIP:-$HIP_REPOS}
HIPRT_REPO_NAME=${HIPRT_REPO_NAME:-hcc2-hip}
HIP_VERSION=${HIP_VERSION:-"0.5-0"}
GFXLIST=${GFXLIST:-"gfx600 gfx601 gfx700 gfx701 gfx702 gfx703 gfx704 gfx801 gfx803 gfx810 gfx900"}

HIPRT_REPO="$HIP_REPOS/$HIPRT_REPO_NAME"
echo HIPRT_REPO=$HIPRT_REPO

SUDO=${SUDO:-set}
if [ $SUDO == "set" ] ; then
   SUDO="sudo"
else
   SUDO=""
fi

BUILD_DIR=$BUILD_HIP/build_hiprt

INSTALL_DIR="${HIP}_${HIP_VERSION}"
LLVM_BUILD=$INSTALL_DIR

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_hiprt.sh                   cmake, make, NO Install "
  echo "  ./build_hiprt.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_hiprt.sh install           NO Cmake, make install "
  echo " "
  exit
fi

if [ ! -d $HIPRT_REPO ] ; then
   echo "ERROR:  Missing repository $HIPRT_REPO/"
   exit 1
fi

if [ ! -f $HIP/bin/clang ] ; then
   echo "ERROR:  Missing file $HIP/bin/clang"
   echo "        Build the HIP llvm compiler in $HIP first"
   echo "        Suggest you use build_hip.sh"
   echo "        This is needed to build the device libraries"
   echo " "
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then
   $SUDO mkdir -p $INSTALL_DIR
   $SUDO touch $INSTALL_DIR/testfile
   if [ $? != 0 ] ; then
      echo "ERROR: No update access to $INSTALL_DIR"
      exit 1
   fi
   $SUDO rm $INSTALL_DIR/testfile
fi

NUM_THREADS=
if [ ! -z `which "getconf"` ]; then
   NUM_THREADS=$(`which "getconf"` _NPROCESSORS_ONLN)
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then

  BUILDTYPE="Release"
  if [ -d "$BUILD_DIR/build_lib" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo rm -rf $BUILD_DIR/build_lib
     rm -rf $BUILD_DIR
  fi

  MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILDTYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DBUILD_SHARED_LIBS=ON -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm"

  mkdir -p $BUILD_DIR/build_lib
  cd $BUILD_DIR/build_lib
  echo 
  echo " -----Running hiprt cmake ---- "
  echo "   cd $BUILD_DIR/build_lib"
  echo "   cmake $MYCMAKEOPTS $HIPRT_REPO"
  cmake $MYCMAKEOPTS $HIPRT_REPO
  if [ $? != 0 ] ; then
      echo 
      echo "ERROR hiprt cmake failed. Cmake flags:"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi

  BUILDTYPE="Debug"
  if [ -d "$BUILD_DIR/build_debug" ] ; then
    echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
    echo rm -rf $BUILD_DIR/build_debug
    rm -rf $BUILD_DIR/build_debug
  fi

  MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILDTYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DBUILD_SHARED_LIBS=ON -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm"

  mkdir -p $BUILD_DIR/build_debug
  cd $BUILD_DIR/build_debug
  echo " -----Running openmp cmake for debug ---- "
  echo cmake $MYCMAKEOPTS $HIPRT_REPO
  cmake $MYCMAKEOPTS $HIPRT_REPO
  if [ $? != 0 ] ; then
     echo "ERROR openmp debug cmake failed. Cmake flags"
     echo "      $MYCMAKEOPTS"
     exit 1
  fi
fi

cd $BUILD_DIR/build_lib
echo
echo " -----Running make for hiprt ---- "
make -j $NUM_THREADS
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $NUM_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build_lib"
      echo "  make"
      exit 1
fi

cd $BUILD_DIR/build_debug
echo " -----Running make for lib-debug ---- "
make -j $NUM_THREADS
if [ $? != 0 ] ; then
      echo "ERROR make -j $NUM_THREADS failed"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build_debug"
      echo "  make"
      exit 1
else
      echo
      echo "BUILD COMPLETE!  To install run this command:"
      echo  $0 install
      echo
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
      cd $BUILD_DIR/build_lib
      echo " -----Installing to $INSTALL_DIR/lib ----- "
      $SUDO make install
      if [ $? != 0 ] ; then
         echo "ERROR make install failed "
         exit 1
      fi
      cd $BUILD_DIR/build_debug
      echo " -----Installing to $INSTALL_DIR/lib-debug ---- "
      $SUDO make install
      if [ $? != 0 ] ; then
         echo "ERROR make install failed "
         exit 1
      fi
      #  Put all libraries in the compiler installation
      echo
      echo  cp -p /opt/rocm/hip/lib/libhip_hcc.so $INSTALL_DIR/lib/libhip_hcc.so
      $SUDO cp -p /opt/rocm/hip/lib/libhip_hcc.so $INSTALL_DIR/lib/libhip_hcc.so
      echo  cp -p /opt/rocm/hcc/lib/libhc_am.so $INSTALL_DIR/lib/libhc_am.so
      $SUDO cp -p /opt/rocm/hcc/lib/libhc_am.so $INSTALL_DIR/lib/libhc_am.so
      echo  cp -p /opt/rocm/hip/lib/libhip_hcc.so $INSTALL_DIR/lib-debug/libhip_hcc.so
      $SUDO cp -p /opt/rocm/hip/lib/libhip_hcc.so $INSTALL_DIR/lib-debug/libhip_hcc.so
      echo  cp -p /opt/rocm/hcc/lib/libhc_am.so $INSTALL_DIR/lib-debug/libhc_am.so
      $SUDO cp -p /opt/rocm/hcc/lib/libhc_am.so $INSTALL_DIR/lib-debug/libhc_am.so

      echo  cp -p /opt/rocm/hcc/lib/libmcwamp_cpu.so $INSTALL_DIR/lib/libmcwamp_cpu.so
      $SUDO cp -p /opt/rocm/hcc/lib/libmcwamp_cpu.so $INSTALL_DIR/lib/libmcwamp_cpu.so
      echo  cp -p /opt/rocm/hcc/lib/libmcwamp_cpu.so $INSTALL_DIR/lib-debug/libmcwamp_cpu.so
      $SUDO cp -p /opt/rocm/hcc/lib/libmcwamp_cpu.so $INSTALL_DIR/lib-debug/libmcwamp_cpu.so
      echo  cp -p /opt/rocm/hcc/lib/libmcwamp_hsa.so $INSTALL_DIR/lib/libmcwamp_hsa.so
      $SUDO cp -p /opt/rocm/hcc/lib/libmcwamp_hsa.so $INSTALL_DIR/lib/libmcwamp_hsa.so
      echo  cp -p /opt/rocm/hcc/lib/libmcwamp_hsa.so $INSTALL_DIR/lib-debug/libmcwamp_hsa.so
      $SUDO cp -p /opt/rocm/hcc/lib/libmcwamp_hsa.so $INSTALL_DIR/lib-debug/libmcwamp_hsa.so
fi
