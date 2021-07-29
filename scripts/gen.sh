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
PT_DIR=$ROOT_DIR/pytorch

function main()
{
    cd $ROOT_DIR
    # patch
    cp $ROOT_DIR/patch/npu.patch $PT_DIR
    cd $PT_DIR
    dos2unix docs/make.bat
    dos2unix scripts/appveyor/install.bat
    dos2unix scripts/appveyor/install_cuda.bat
    dos2unix scripts/build_windows.bat
    dos2unix scripts/proto.ps1
    dos2unix torch/distributions/von_mises.py
    dos2unix torch/nn/modules/transformer.pyi.in
    patch -p1 < npu.patch
    cp -r $ROOT_DIR/src/* $PT_DIR
}

main $@

