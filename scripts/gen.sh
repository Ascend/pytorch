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
PT_DIR=$ROOT_DIR/pytorch_v1.5.0
Version=1.5.0
function main()
{
    cd $ROOT_DIR
    # patch
    if [ $Version = "1.5.0" ]; then
        cp $ROOT_DIR/patch/pytorch1.5.0_npu.patch $PT_DIR
        cd $PT_DIR
        patch -p1 < pytorch1.5.0_npu.patch
        cp -r $ROOT_DIR/pytorch1.5.0/src/* $PT_DIR
    
    elif [ $Version = "1.8.1" ];  then
        cp $ROOT_DIR/patch/pytorch1.8.1_npu.patch $PT_DIR
        cd $PT_DIR
        patch -p1 < pytorch1.8.1_npu.patch
        cp -r $ROOT_DIR/pytorch1.8.1/src/* $PT_DIR
    else
        echo "Usage: Only support version: 1.5.0 or 1.8.1."
        exit 1
    fi     
}

case "$1" in
    -v|--version)
        Version=$2;
    ;;

esac


main