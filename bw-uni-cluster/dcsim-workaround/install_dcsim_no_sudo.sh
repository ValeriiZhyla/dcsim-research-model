#! /usr/bin/bash

module load compiler/gnu/12.1 

git clone https://github.com/HEPCompSim/DCSim.git
this_dir="$( cd "$( dirname "$this_file" )" && pwd )"
pushd DCSim
INSTALL_LOCATION="/home/kit/kastel/jn1292/local"

# pugixml
git clone https://github.com/zeux/pugixml.git
mkdir -p pugixml/build
pushd pugixml/build
git checkout tags/v1.12.1
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" ../
make -j16
make install
popd

# json
git clone https://github.com/nlohmann/json.git
mkdir -p json/build
pushd json/build
git checkout tags/v3.11.2
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" ../
make -j16
make install
popd

# googletest
git clone https://github.com/google/googletest.git
mkdir -p googletest/build
pushd googletest/build
git checkout tags/release-1.12.1
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" ../
make -j16
make install
popd

# simgrid
git clone --depth 1 --branch "v3.34" https://framagit.org/simgrid/simgrid.git
mkdir -p simgrid/build
pushd simgrid/build
cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" ../
make -j16
make install
popd

# wrench
git clone --depth 1 --branch "v2.2" https://github.com/wrench-project/wrench.git
mkdir -p wrench/build
pushd wrench/build
# git checkout tags/v.2.1
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" ../
make -j16
make install
popd


# DCSim
mkdir -p build
pushd build
# if you got an error with "wrench-dev.h" - add 
# include_directories(src/ /home/kit/kastel/jn1292/local/include ${SimGrid_INCLUDE_DIR}/include /usr/local/include /opt/local/include /usr/local/include/wrench ${Boost_INCLUDE_DIR})
# into CMakeLists
cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" ../
make -j16
make install
popd

