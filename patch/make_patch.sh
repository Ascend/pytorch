#!/bin/bash

# Copyright (c) 2021, Huawei Technologies Co., Ltd
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


OFFICIAL_PYTORCH=$1
PATCH_PYTORCH=$2
OUT_PATCH_FILE=$3

diff -Nur --exclude=.git* --exclude=OWNERS \
    --exclude=access_control_test.py \
    --exclude=build.sh --exclude=third_party \
    --exclude=README* -Nur ${OFFICIAL_PYTORCH}  ${PATCH_PYTORCH} > ${OUT_PATCH_FILE}