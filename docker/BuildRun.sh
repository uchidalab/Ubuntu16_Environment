#!/usr/bin/env bash
set -eu

IMAGE="snhryt/research-env"

num=`docker images | grep ${IMAGE} | wc -l`
if test num = "0"; then
  docker build -t ${IMAGE} .
fi

PARENT_DIRPATH="/home/snhryt/Research_Master"
docker run runtime=nvidia -it --rm -v ${PARENT_DIRPATH}:/workdir -w /workdir -p 8888:8888 ${IMAGE}
