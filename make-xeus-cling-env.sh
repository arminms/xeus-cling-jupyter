
#!/usr/bin/env bash
#
# bash script for building/installing xeus-cling 0.15.3 from source in a virtual
# environment
#
# Copyright (c) 2024 Armin Sobhani (arminms@gmail.com)
#
if [ $# -eq 1 ]; then
  INSTALL_DIR=$1
elif [ $# -eq 0 ]; then
  INSTALL_DIR=~/xeus-cling-env
else
  echo "usage: $0 [VIRTUAL_ENV_PATH=[~/xeus-cling-env]]"
  exit 1
fi

mkdir -p build && cd build
export BUILD_DIR=$PWD

# exit on error and verbose
set -ex

# create the python virtual environment and install dependencies
virtualenv $INSTALL_DIR
source $INSTALL_DIR/bin/activate
pip install --upgrade pip
pip install jupyter
pip install ipython
pip install ipykernel

# llvm13
cd $BUILD_DIR
if [ ! -d "llvm-project" ]; then
  git clone https://github.com/root-project/llvm-project.git
fi
cd llvm-project/
# git checkout cling-latest
git checkout cling-llvm13

# cling v1.0~dev
cd $BUILD_DIR
if [ ! -d "cling" ]; then
  git clone https://github.com/root-project/cling.git
fi
cd cling/
# git checkout v0.9
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
#      -DLLVM_TOOL_OPENMP_BUILD=ON
#      -DCMAKE_SYSTEM_PREFIX_PATH=$EBROOTGENTOO \
#      -DLLVM_REQUIRES_EH=ON \
#      -DLLVM_ENABLE_LIBCXX=ON \
cmake --build . --target cling llvm-config -j 4
make -j install-clang-headers install-clang-resource-headers install-llvm-headers # install-libcling install-cling-cmake-exports install-clang install-libclang install-libclang-headers install-clang-cpp install-llvm-config

#export PATH=$INSTALL_DIR/bin:$PATH

# copying necessary files
# cp -r $BUILD_DIR/llvm-project/clang/include/clang $INSTALL_DIR/include
# cp -r tools/clang/include/clang $INSTALL_DIR/include
cp bin/* $INSTALL_DIR/bin
cp lib/*.a $INSTALL_DIR/lib
cp -r lib/cmake $INSTALL_DIR/lib
CH=$BUILD_DIR/cling/include/cling
mkdir -p $INSTALL_DIR/include/cling
cp -r $CH/Interpreter $CH/MetaProcessor $CH/UserInterface $CH/Utils $INSTALL_DIR/include/cling

# nlohmann/json v3.6.1
cd $BUILD_DIR
git clone https://github.com/nlohmann/json.git
cd json/
git checkout v3.6.1
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_BUILD_TYPE=Release -DJSON_BuildTests=OFF
cmake --build build -j && cmake --install build

# xtl 0.7.5
cd $BUILD_DIR
git clone https://github.com/xtensor-stack/xtl.git
cd xtl/
git checkout 0.7.5
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
cmake --build build -j && cmake --install build

# xeus 3.2.0
cd $BUILD_DIR
git clone https://github.com/jupyter-xeus/xeus.git
cd xeus/
git checkout 3.2.0
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
cmake --build build -j && cmake --install build

# libzmq v4.3.4
cd $BUILD_DIR
git clone https://github.com/zeromq/libzmq.git
cd libzmq/
git checkout v4.3.4
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DWITH_PERF_TOOL=OFF -DZMQ_BUILD_TESTS=OFF
cmake --build build -j && cmake --install build

# cppzmq v4.8.1
cd $BUILD_DIR
git clone https://github.com/zeromq/cppzmq.git
cd cppzmq/
git checkout v4.8.1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCPPZMQ_BUILD_TESTS=NO
cmake --build build -j && cmake --install build

# xeus-zmq 1.3.0
cd $BUILD_DIR
git clone https://github.com/jupyter-xeus/xeus-zmq.git
cd xeus-zmq/
git checkout 1.3.0
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
cmake --build build -j && cmake --install build

# pugixml v1.8.1
cd $BUILD_DIR
git clone https://github.com/zeux/pugixml.git
cd pugixml/
git checkout v1.8.1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build -j && cmake --install build

# argparse v2.9
cd $BUILD_DIR
git clone https://github.com/p-ranav/argparse.git
cd argparse/
git checkout v2.9
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
cmake --build build -j && cmake --install build

# xeus-cling 0.15.3
cd $BUILD_DIR
git clone https://github.com/jupyter-xeus/xeus-cling.git
cd xeus-cling/
git checkout 0.15.3

# fix compilation errors with llvm16 (https://github.com/jupyter-xeus/xeus-cling/pull/524/files)
# sed -i -e 's/"--src-root"//g' CMakeLists.txt
# sed -i -e 's/HeaderSearchOpts,/&CI->getVirtualFileSystem(), HeaderSearchOpts,/g' src/xmagics/executable.cpp
# sed -i -e 's/*Context)/*m_interpreter.getLLVMContext())/g' src/xmagics/executable.cpp
sed -i -e 's/.getDataLayout();/.getDataLayoutString()/g' src/xmagics/executable.cpp
sed -i -e 's/getDataLayoutString()/getDataLayoutString();/g' src/xmagics/executable.cpp
# sed -i -e 's/{llvm::NoneType::None,/std::nullopt,/g' src/xmagics/executable.cpp
sed -i -e 's/simplisticCastAs/castAs/g' src/xmagics/execution.cpp
sed -i -e 's/code.str()/std::string(code.str())/g' src/xmime_internal.hpp

# patch to install new kernel files
rm -rf share/jupyter/kernels/*
cp -r $BUILD_DIR/kernels/* share/jupyter/kernels/
patch -u CMakeLists.txt $BUILD_DIR/patches/kernels.diff

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCling_DIR=$INSTALL_DIR/tools/cling/lib/cmake/cling
cmake --build build -j && cmake --install build

# deactivate the virtual environment
deactivate

# print the installation path
echo "xeus-cling 0.15.3 has been successfully installed in $INSTALL_DIR"
echo "run 'source $INSTALL_DIR/bin/activate' to activate the virtual environment"
echo "and then run 'jupyter lab' to start jupyter lab"
echo "run 'rm -rf $BUILD_DIR' to remove the build directory"