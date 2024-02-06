#!/bin/bash

py_execute=${1-python3}
CUR_DIR=$(dirname $(readlink -f $0))

cd ${CUR_DIR}/../examples/libtorch_resnet

if [ -f ./resnet_model.pt ]; then
    rm ./resnet_model.pt
fi

$py_execute resnet_trace.py

if [ $? != 0 ]; then
    echo "Failed to trace resnet model."
    exit 1
fi

if [ -d "build" ]; then
    rm -rf ./build
fi

mkdir build && cd build && \
cmake -DCMAKE_PREFIX_PATH=`$py_execute -c 'import torch;print(torch.utils.cmake_prefix_path)'` .. && \
make && ./libtorch_resnet ../resnet_model.pt > out.log

if grep -q "resnet_model run success!" out.log; then
  echo "Success."
  exit 0
else
  echo "Failed."
  exit 1
fi
