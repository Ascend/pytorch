#!/bin/bash

# Copyright (c) 2020, Huawei Technologies Co., Ltd
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
ROOT_DIR=$CUR_DIR/..

SRC_DIR=pytorch-develop

# mkdir src
mkdir -p src/aten/src/ATen/native
mkdir -p src/aten/src/ATen/detail
mkdir -p src/aten/src/ATen/templates

mkdir -p src/c10

mkdir -p src/cmake/public

mkdir -p src/third_party

mkdir -p src/torch/contrib
mkdir -p src/torch/csrc/autograd
mkdir -p src/torch/csrc/utils
mkdir -p src/torch/lib/c10d
mkdir -p src/torch/utils

mkdir -p src/tools/autograd

mkdir -p temp/test

# move files
mv $SRC_DIR/aten/src/ATen/native/npu src/aten/src/ATen/native
mv $SRC_DIR/aten/src/THNPU src/aten/src
mv $SRC_DIR/aten/src/ATen/detail/NPU* src/aten/src/ATen/detail
mv $SRC_DIR/aten/src/ATen/npu src/aten/src/ATen
mv $SRC_DIR/aten/src/ATen/templates/NPU* src/aten/src/ATen/templates
cp $SRC_DIR/aten/src/ATen/native/native_functions.yaml src/aten/src/ATen/native
cp $SRC_DIR/tools/autograd/derivatives.yaml src/tools/autograd

mv $SRC_DIR/c10/npu src/c10

mv $SRC_DIR/cmake/public/npu.cmake src/cmake/public

mv $SRC_DIR/third_party/acl src/third_party
mv $SRC_DIR/third_party/hccl src/third_party

mv $SRC_DIR/torch/contrib/npu src/torch/contrib
mv $SRC_DIR/torch/csrc/autograd/profiler_npu.cpp src/torch/csrc/autograd
mv $SRC_DIR/torch/csrc/npu src/torch/csrc
mv $SRC_DIR/torch/csrc/utils/npu_* src/torch/csrc/utils
mv $SRC_DIR/torch/npu src/torch
mv $SRC_DIR/torch/lib/c10d/HCCL* src/torch/lib/c10d
mv $SRC_DIR/torch/lib/c10d/ProcessGroupHCCL* src/torch/lib/c10d

mv $SRC_DIR/env.sh src
mv $SRC_DIR/build.sh src # where

## dump util
mv $SRC_DIR/aten/src/ATen/utils src/aten/src/ATen
mv $SRC_DIR/torch/utils/dumper.py src/torch/utils

# end
mv src temp
mv $SRC_DIR/test/test_npu temp/test
mv $SRC_DIR/access_control_test.py temp
