#!/usr/bin/env bash
set -eu

CURRENT_DIRPATH=`pwd`
BUILD_DIRNAME="build"

# Boost test
echo '===Boost test===================================================================='
cd "${CURRENT_DIRPATH}/BoostSample_cpp"
if [ ! -d ${BUILD_DIRNAME} ]; then
  mkdir ${BUILD_DIRNAME}
fi
cd ${BUILD_DIRNAME}
cmake ..
make
./BoostSample ${CURRENT_DIRPATH}

# OpenCV test
echo '===OpenCV test===================================================================='
cd "${CURRENT_DIRPATH}/OpencvSample_cpp"
if [ ! -d ${BUILD_DIRNAME} ]; then
  mkdir ${BUILD_DIRNAME}
fi
cd ${BUILD_DIRNAME}
cmake ..
make
./OpencvSample "${CURRENT_DIRPATH}/lenna.png"
