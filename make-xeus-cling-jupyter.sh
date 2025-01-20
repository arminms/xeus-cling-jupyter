
#!/usr/bin/env bash
#
# Copyright (c) 2024 Armin Sobhani (arminms@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# A bash script for building/installing xeus-cling 0.15.3 from source in a 
# python virtual environment. The script requires 'realpath' to be installed.
#
usage()
{
    cat << EOF
Usage: $0 [OPTION]... [DIRECTORY(=~/xeus-cling(-env))]

Build and install xeus-cling 0.15.3 from source in DIRECTORY, optionally with a
python virtual environment or as a Docker image.
Needs CMake 3.12+ and 'realpath' to work (e.g. 'sudo apt-get install realpath').

  -b  FOLDER    build directory (default: ./build)
  -c            clean the build directory after installation
  -d            build xeus-cling-jupyter Docker image (requires Docker)
  -g            install xeus-cling CUDA-enabled kernels for NVIDIA GPUs
  -k            install kernels to the current user's kernel registry
  -n  N         number of threads to build cling (default: 2)
  -h            show this help message
  -r            resume the build from the last step
  -s            skip creating python virtual environment
  -x            install extra libraries (xproperty, xwidgets, xtensor)

EOF
}

# setting default values
#
N=2

# parse the command line arguments
#
while getopts ":b:cgkn:rsx" o; do
    case "${o}" in
        b)
            BUILD_DIR=$(realpath ${OPTARG})
            ;;
        c)
            CLEAN=1
            ;;
        d)
            DOCKER=1
            ;;
        g)
            CUDA_KERNELS=1
            ;;
        k)  USER_KERNELS=1
            ;;
        n)
            N=${OPTARG}
            ;;
        h)
            usage
            ;;
        s)
            SKIP_VENV=1
            ;;
        r)
            RESUME=1
            ;;
        x)
            EXTRA_LIBS=1
            ;;
        *)
            usage && exit 1
            ;;
    esac
done
shift $((OPTIND-1))

# exit on error and verbose
#
set -ex

# build the Docker image if requested and exit
if [ "x{$DOCKER}" = 1 ]; then
  if [ -z "${CUDA_KERNELS}" ]; then
    docker build -t xeus-cling-jupyter:0.15.3-cling1.0dev-llvm13-ubuntu20.04 . \
    && cat << EOF

Docker image 'xeus-cling-jupyter:0.15.3-cling1.0dev-llvm13-ubuntu20.04' has been successfully built.
run 'docker run -p 8888:8888 -it --rm xeus-cling-jupyter:0.15.3-cling1.0dev-llvm13-ubuntu20.04' to start jupyter

EOF
  else
    docker build --build-arg CUDA=10 -t -t xeus-cling-jupyter:0.15.3-cling1.0dev-llvm13-cuda10.1-ubuntu20.04 . \
    && cat << EOF

Docker image 'xeus-cling-jupyter:0.15.3-cling1.0dev-llvm13-cuda10.1-ubuntu20.04' has been successfully built.
To run CUDA kernels, 'NVIDIA Container Toolkit' must have been installed on the host:

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

run 'docker run --gpus=all -p 8888:8888 -it --rm xeus-cling-jupyter:0.15.3-cling1.0dev-llvm13-cuda10.1-ubuntu20.04' to start jupyter

EOF
  fi
  exit 0
fi

# set the installation directory
#
if [ -z "${1}" ] ; then
  if [ -z "${SKIP_VENV}" ] ; then
    mkdir -p ~/xeus-cling && INSTALL_DIR=~/xeus-cling
  else
    INSTALL_DIR=~/xeus-cling-env
  fi
else
  INSTALL_DIR=$(realpath $1)
  if [ -z "${SKIP_VENV}" ] ; then
    mkdir -p $INSTALL_DIR
  fi
fi

# set the build directory
#
if [ -z "${BUILD_DIR}" ] ; then
  mkdir -p build && cp -r kernels patches build && cd build && BUILD_DIR=$PWD
else
  cp -r kernels patches $BUILD_DIR && cd $BUILD_DIR
  if [ -z "${CUDA_KERNELS}" ] ; then
    rm -rf kernels/*-cuda
  fi
fi

# create the python virtual environment and install dependencies if needed
#
if [ -z "${SKIP_VENV}" ] ; then
  virtualenv $INSTALL_DIR
  source $INSTALL_DIR/bin/activate
  pip install --upgrade pip && pip install jupyter ipython ipykernel jupyterlab_myst jupyterlab_widgets widgetsnbextension mystmd jupytext
  jupytext-config set-default-viewer
else
  mkdir -p $INSTALL_DIR/bin $INSTALL_DIR/share/jupyter/kernels/
  export PATH=$INSTALL_DIR/bin:$PATH
fi

# nlohmann/json v3.6.1
#
if [ -z "${RESUME}" ] || ([ $RESUME -eq 1 ] && [ ! -f $INSTALL_DIR/include/nlohmann/json.hpp ]); then
  cd $BUILD_DIR
  if [ ! -d "json" ]; then
    git clone https://github.com/nlohmann/json.git
  fi
  cd json/
  git checkout v3.6.1
  cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_BUILD_TYPE=Release -DJSON_BuildTests=OFF
  cmake --build build -j && cmake --install build
fi

# xtl 0.7.5
#
if [ -z "${RESUME}" ] || ([ $RESUME -eq 1 ] && [ !  -d $INSTALL_DIR/include/xtl ]); then
  cd $BUILD_DIR
  if [ ! -d "xtl" ]; then
    git clone https://github.com/xtensor-stack/xtl.git
  fi
  cd xtl/
  git checkout 0.7.5
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
  cmake --build build -j && cmake --install build
fi

# xeus 3.2.0
#
if [ -z "${RESUME}" ] || ([ $RESUME -eq 1 ] && [ ! -f $INSTALL_DIR/include/xeus/xeus.hpp ]); then
  cd $BUILD_DIR
  if [ ! -d "xeus" ]; then
    git clone https://github.com/jupyter-xeus/xeus.git
  fi
  cd xeus/
  git checkout 3.2.0
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
  cmake --build build -j && cmake --install build
fi

# libzmq v4.3.4
#
if [ -z "${RESUME}" ] || ([ $RESUME -eq 1 ] && [ ! -f $INSTALL_DIR/lib/libzmq.so ]); then
  cd $BUILD_DIR
  if [ ! -d "libzmq" ]; then
    git clone https://github.com/zeromq/libzmq.git
  fi
  cd libzmq/
  git checkout v4.3.4
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DWITH_PERF_TOOL=OFF -DZMQ_BUILD_TESTS=OFF
  cmake --build build -j && cmake --install build
fi

# cppzmq v4.8.1
#
if [ -z "${RESUME}" ] || ([ $RESUME -eq 1 ] && [ ! -f $INSTALL_DIR/share/cmake/cppzmq/cppzmqConfig.cmake ]); then
  cd $BUILD_DIR
  if [ ! -d "cppzmq" ]; then
    git clone https://github.com/zeromq/cppzmq.git
  fi
  cd cppzmq/
  git checkout v4.8.1
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCPPZMQ_BUILD_TESTS=NO
  cmake --build build -j && cmake --install build
fi

# xeus-zmq 1.3.0
#
if [ -z "${RESUME}" ] || ([ $RESUME -eq 1 ] && [ ! -f $INSTALL_DIR/lib/libxeus-zmq.so ]); then
  cd $BUILD_DIR
  if [ ! -d "xeus-zmq" ]; then
    git clone https://github.com/jupyter-xeus/xeus-zmq.git
  fi
  cd xeus-zmq/
  git checkout 1.3.0
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
  cmake --build build -j && cmake --install build
fi

# pugixml v1.8.1
#
if [ -z "${RESUME}" ] || ([ $RESUME -eq 1 ] && [ ! -f $INSTALL_DIR/lib/libpugixml.a ]); then
  cd $BUILD_DIR
  if [ ! -d "pugixml" ]; then
    git clone https://github.com/zeux/pugixml.git
  fi
  cd pugixml/
  git checkout v1.8.1
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  cmake --build build -j && cmake --install build
fi

# argparse v2.9
#
if [ -z "${RESUME}" ] || ([ $RESUME -eq 1 ] && [ ! -f $INSTALL_DIR/include/argparse/argparse.hpp ]); then
  cd $BUILD_DIR
  if [ ! -d "argparse" ]; then
    git clone https://github.com/p-ranav/argparse.git
  fi
  cd argparse/
  git checkout v2.9
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
  cmake --build build -j && cmake --install build
fi

# llvm13 and cling v1.0~dev
#
if [ -z "${RESUME}" ] || ([ $RESUME -eq 1 ] && [ ! -f $BUILD_DIR/cling-build/bin/cling ]); then
  cd $BUILD_DIR
  if [ ! -d "llvm-project" ]; then
    git clone https://github.com/root-project/llvm-project.git
  fi
  cd llvm-project/
  git checkout cling-llvm13
  cd $BUILD_DIR
  if [ ! -d "cling" ]; then
    git clone https://github.com/root-project/cling.git
  fi
  cd cling/
  git checkout acb2334131c19ef506767d6d9051b24755a8566b
  # patch to allow redefinition by default
  sed -i -e 's/AllowRedefinition(0)/AllowRedefinition(1)/g' include/cling/Interpreter/RuntimeOptions.h
  mkdir -p $BUILD_DIR/cling-build && cd $BUILD_DIR/cling-build
  cmake -DLLVM_EXTERNAL_PROJECTS=cling \
        -DLLVM_BUILD_LLVM_DYLIB=OFF \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_ENABLE_EH=ON \
        -DLLVM_BUILD_DOCS=OFF \
        -DLLVM_ENABLE_SPHINX=OFF \
        -DLLVM_ENABLE_DOXYGEN=OFF \
        -DLLVM_ENABLE_WERROR=OFF \
        -DLLVM_EXTERNAL_CLING_SOURCE_DIR=$BUILD_DIR/cling/ \
        -DLLVM_ENABLE_PROJECTS="clang" \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=Release \
        $BUILD_DIR/llvm-project/llvm
  cmake --build . --target cling llvm-config -j $N
  make -j install-clang-headers install-clang-resource-headers install-llvm-headers
else
  cd $BUILD_DIR/cling-build
fi
make -j install-clang-headers install-clang-resource-headers install-llvm-headers
cp bin/* $INSTALL_DIR/bin
cp lib/*.a $INSTALL_DIR/lib
cp -r lib/cmake $INSTALL_DIR/lib
CH=$BUILD_DIR/cling/include/cling
mkdir -p $INSTALL_DIR/include/cling
cp -r $CH/Interpreter $CH/MetaProcessor $CH/UserInterface $CH/Utils $INSTALL_DIR/include/cling

# xeus-cling 0.15.3
#
if [ -z "${RESUME}" ] || ([ $RESUME -eq 1 ] && [ ! -f $INSTALL_DIR/bin/xcpp ]); then
  cd $BUILD_DIR
  if [ ! -d "xeus-cling" ]; then
    git clone https://github.com/jupyter-xeus/xeus-cling.git
    cd xeus-cling/
    git checkout 0.15.3
    # fix compilation errors with llvm13
    sed -i -e 's/.getDataLayout();/.getDataLayoutString()/g' src/xmagics/executable.cpp
    sed -i -e 's/getDataLayoutString()/getDataLayoutString();/g' src/xmagics/executable.cpp
    sed -i -e 's/simplisticCastAs/castAs/g' src/xmagics/execution.cpp
    sed -i -e 's/code.str()/std::string(code.str())/g' src/xmime_internal.hpp
    # patch to install new kernel files
    rm -rf share/jupyter/kernels/*
    cp -r $BUILD_DIR/kernels/* share/jupyter/kernels/
    patch -u CMakeLists.txt $BUILD_DIR/patches/kernels.diff
  else
    cd xeus-cling/
  fi
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCling_DIR=$INSTALL_DIR/tools/cling/lib/cmake/cling
  cmake --build build -j && cmake --install build
fi

# install extra libraries if needed
#
if [ ! -z "${EXTRA_LIBS}" ] ; then
  # xproperty 0.11.0
  #
  if [ -z "${RESUME}" ] || ([ $RESUME -eq 1 ] && [ ! -f $INSTALL_DIR/include/xproperty/xproperty.hpp ]); then
    cd $BUILD_DIR
    if [ ! -d "xproperty" ]; then
      git clone https://github.com/jupyter-xeus/xproperty.git
    fi
    cd xproperty/
    git checkout 0.11.0
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
    cmake --build build -j && cmake --install build
  fi
  # xwidgets 0.28.1
  #
  if [ -z "${RESUME}" ] || ([ $RESUME -eq 1 ] && [ ! -f $INSTALL_DIR/include/xwidgets/xwidget.hpp ]); then
    cd $BUILD_DIR
    if [ ! -d "xwidgets" ]; then
      git clone https://github.com/jupyter-xeus/xwidgets.git
    fi
    cd xwidgets/
    git checkout 0.28.1
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
    cmake --build build -j $N && cmake --install build
  fi
  # xtensor 0.25.0
  #
  if [ -z "${RESUME}" ] || ([ $RESUME -eq 1 ] && [ ! -f $INSTALL_DIR/include/xtensor/xtensor.hpp ]); then
    cd $BUILD_DIR
    if [ ! -d "xtensor" ]; then
      git clone https://github.com/xtensor-stack/xtensor.git
    fi
    cd xtensor/
    git checkout 0.25.0
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
    cmake --build build -j && cmake --install build
  fi 
fi

# install kernels to the current user's kernel registry if needed
#
if [ ! -z "${USER_KERNELS}" ] ; then
  cd ${INSTALL_DIR}/share/jupyter/kernels
  for k in */ ; do
    jupyter kernelspec install --user $k
  done
  cd ${BUILD_DIR}
fi

# deactivate the virtual environment if needed
#
if [ -z "${SKIP_VENV}" ] ; then
  deactivate
fi

# print the installation path
#
set +x
if [ ! -z "${CLEAN}" ] ; then
  rm -rf $BUILD_DIR
fi
echo
echo "xeus-cling 0.15.3 has been successfully installed in $INSTALL_DIR"
if [ -z "${SKIP_VENV}" ] ; then
  echo "run 'source $INSTALL_DIR/bin/activate' to activate the virtual environment"
  echo "and then 'jupyter lab' to start jupyter"
fi
if [ -z "${CLEAN}" ] ; then
  echo "run 'rm -rf $BUILD_DIR' to remove the build directory if you don't need it anymore"
fi
echo
