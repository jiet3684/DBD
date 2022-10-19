#!/bin/bash
#
#  File: build_libdevice.sh
#        build the rocm-device-libs for multiple gfxids in $HIP/lib/libdevice
#

# Do not change these values. Set the environment variables to override these defaults

HIP=${HIP:-/usr/local/hip}
HIP_REPOS=${HIP_REPOS:-/home/$USER/git/hip}
BUILD_HIP=${BUILD_HIP:-$HIP_REPOS}
HIP_LIBDEVICE_REPO_NAME=${HIP_LIBDEVICE_REPO_NAME:-rocm-device-libs}

HSA_DIR=${HSA_DIR:-/opt/rocm/hsa}
SKIPTEST=${SKIPTEST:-"YES"}
SUDO=${SUDO:-set}
if [ "$SUDO" == "set" ] ; then 
   SUDO="sudo"
else 
   SUDO=""
fi

BUILD_DIR=$BUILD_HIP
if [ "$BUILD_DIR" != "$HIP_REPOS" ] ; then 
   COPYSOURCE=true
fi

INSTALL_DIR="${HIP}/lib/libdevice"

LLVM_BUILD=$HIP
SOURCEDIR=$HIP_REPOS/$HIP_LIBDEVICE_REPO_NAME

MCPU_LIST=${GFXLIST:-"gfx600 gfx601 gfx700 gfx701 gfx702 gfx703 gfx704 gfx801 gfx803 gfx804 gfx810 gfx900"}

MYCMAKEOPTS="-DLLVM_DIR=$LLVM_BUILD -DBUILD_HC_LIB=ON -DROCM_DEVICELIB_INCLUDE_TESTS=OFF -DAMDGPU_TARGET_TRIPLE=amdgcn-amd-amdhsa"

if [ ! -d $HIP/lib ] ; then 
  echo "ERROR: Directory $HIP/lib is missing"
  echo "       HIP must be installed in $HIP to continue"
  exit 1
fi

function gfx2code(){ 
   case "$1" in 
      "gfx600") codename="tahiti"
      ;;
      "gfx601") codename="pitcairn"
      ;;
      "gfx700") codename="kaveri"
      ;;
      "gfx701") codename="hawaii"
      ;;
      "gfx702") codename="r390"
      ;;
      "gfx703") codename="kabini"
      ;;
      "gfx704") codename="bonaire"
      ;;
      "gfx800") codename="iceland"
      ;;
      "gfx801") codename="carrizo"
      ;;
      "gfx802") codename="tonga"
      ;;
      "gfx803") codename="fiji"
      ;;
      "gfx804") codename="polaris"
      ;;
      "gfx810") codename="stoney"
      ;;
      "gfx900") codename="vega"
      ;;
      "gfx901") codename="tbd901"
      ;;
      "gfx902") codename="tbd902"
      ;;
      *) codename="$1" 
      ;;
   esac
   echo $codename
}

NUM_THREADS=
if [ ! -z `which "getconf"` ]; then
   NUM_THREADS=$(`which "getconf"` _NPROCESSORS_ONLN)
fi

export LLVM_BUILD HSA_DIR
export PATH=$LLVM_BUILD/bin:$PATH

if [ "$1" != "install" ] ; then 
   if [ $COPYSOURCE ] ; then 
      if [ -d $BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME ] ; then 
         echo rm -rf $BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME
         rm -rf $BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME
      fi
      mkdir -p $BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME
      echo rsync -a $SOURCEDIR/ $BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME/
      rsync -a $SOURCEDIR/ $BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME/
   fi

   LASTMCPU="fiji"
   sedfile1=$BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME/OCL.cmake
   sedfile2=$BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME/CMakeLists.txt
   origsedfile2=$BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME/CMakeLists.txt.orig
   if [ ! $COPYSOURCE ] ; then 
     cp $sedfile2 $origsedfile2 
   fi
   for MCPU in $MCPU_LIST  ; do 
      builddir_mcpu=$BUILD_DIR/build_libdevice/build_$MCPU
      if [ -d $builddir_mcpu ] ; then 
         echo rm -rf $builddir_mcpu
         rm -rf $builddir_mcpu
      fi
      mkdir -p $builddir_mcpu
      cd $builddir_mcpu
      echo 
      echo DOING BUILD FOR $MCPU in Directory $builddir_mcpu
      echo 
      installdir_gfx="$INSTALL_DIR/$MCPU"
      sed -i -e"s/mcpu=$LASTMCPU/mcpu=$MCPU/" $sedfile1
      sed -i -e"s/mcpu=$LASTMCPU/mcpu=$MCPU/" $sedfile2
      LASTMCPU="$MCPU"

      # check seds worked
      echo CHECK: grep mcpu $sedfile1
      grep mcpu $sedfile1
      echo CHECK: grep mcpu $sedfile2
      grep mcpu $sedfile2
      CC="$LLVM_BUILD/bin/clang"
      export CC
      echo "cmake $MYCMAKEOPTS -DCMAKE_INSTALL_PREFIX=$installdir_gfx $BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME"
      cmake $MYCMAKEOPTS -DCMAKE_INSTALL_PREFIX=$installdir_gfx $BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME
      if [ $? != 0 ] ; then 
         echo "ERROR cmake failed for $MCPU, command was \n"
         echo "      cmake $MYCMAKEOPTS -DCMAKE_INSTALL_PREFIX=$installdir_gfx $BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME"
         if [ ! $COPYSOURCE ] ; then 
            #  Put the cmake files in repository back to original condition.
            cd $BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME
            git checkout $sedfile1
            cp $origsedfile2 $sedfile2
         fi
         exit 1
      fi
      make -j $NUM_THREADS 
      if [ $? != 0 ] ; then 
         echo "ERROR make failed for $MCPU "
         if [ ! $COPYSOURCE ] ; then 
            #  Put the cmake files in repository back to original condition.
            cd $BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME
            git checkout $sedfile1
            cp $origsedfile2 $sedfile2
         fi
         exit 1
      fi
   done
   if [ ! $COPYSOURCE ] ; then 
      #  Put the cmake files in repository back to original condition.
      cd $BUILD_DIR/$HIP_LIBDEVICE_REPO_NAME
      git checkout $sedfile1
      cp $origsedfile2 $sedfile2
      rm $origsedfile2 
   fi
   echo 
   echo "  Done with all makes"
   echo "  Please run ./build_libdevice.sh install "
   echo 

   if [ "$SKIPTEST" != "YES" ] ; then 
      for MCPU in $MCPU_LIST  ; do 
         builddir_mcpu=$BUILD_DIR/build_libdevice/build_$MCPU
         cd $builddir_mcpu
         echo "running tests in $builddir_mcpu"
         make test 
      done
      echo 
      echo "# done with all tests"
      echo 
   fi
fi

if [ "$1" == "install" ] ; then 
   for MCPU in $MCPU_LIST  ; do 
      echo 
      installdir_gfx="$INSTALL_DIR/$MCPU"
      echo mkdir -p $installdir_gfx/include
      $SUDO mkdir -p $installdir_gfx/include
      $SUDO mkdir -p $installdir_gfx/lib
      builddir_mcpu=$BUILD_DIR/build_libdevice/build_$MCPU
      codename=$(gfx2code $MCPU)
      installdir_codename=$INSTALL_DIR/${codename}
      echo "running make install from $builddir_mcpu"
      cd $builddir_mcpu
      echo $SUDO make -j $NUM_THREADS install
      $SUDO make -j $NUM_THREADS install
      if [ -L $installdir_codename ] ; then 
         $SUDO rm $installdir_codename
      fi
      cd $installdir_gfx/..
      echo $SUDO ln -sf $MCPU ${codename}
      $SUDO ln -sf $MCPU ${codename}
   done

   # Make sure the ocl_isa_version returns correct version
   for fixdir in `ls -d $INSTALL_DIR/gfx*` ; do 
      id=${fixdir##*gfx}
      for fixfile in `ls $fixdir/lib/oclc_isa_version_* 2>/dev/null` ; do
         idfile=${fixfile##*isa_version_}
         idfile=${idfile%*.amdgcn.bc}
         if [ "$id" == "$idfile" ] ; then 
            $SUDO mv $fixfile $fixdir/lib/oclc_isa_version.amdgcn.bc
         else
            $SUDO rm $fixfile
         fi
      done
   done

   # rocm-device-lib cmake installs to lib dir, move all bc files up one level
   for MCPU in $MCPU_LIST  ; do 
      installdir_gfx="$INSTALL_DIR/$MCPU"
      echo mv $installdir_gfx/lib/*.bc $installdir_gfx
      $SUDO mv $installdir_gfx/lib/*.bc $installdir_gfx
      echo rmdir $installdir_gfx/lib 
      $SUDO rmdir $installdir_gfx/lib 
   done

   echo 
   echo " $0 Installation complete into $INSTALL_DIR"
   echo 
fi
