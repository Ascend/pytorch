#!/bin/bash

# Copyright (c) 2023 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CUR_DIR=$(dirname $(readlink -f $0))

cd ${CUR_DIR}/../examples/libtorch_resnet

if [ -f ./resnet_model.pt ]; then
    rm ./resnet_model.pt
fi

python3 resnet_trace.py

if [ $? != 0 ]; then
    echo "Failed to trace resnet model."
    exit 1
fi

if [ -d "build" ]; then
    rm -rf ./build
fi

mkdir build && cd build && \
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` .. && \
make && ./libtorch_resnet ../resnet_model.pt
