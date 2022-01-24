#!/bin/bash

# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
NDIR="$CDIR/.."

cd $NDIR/scripts


cp -f codegen/native_functions.yaml codegen/native_functions.yaml_bk

sed -i '/ _foreach/a\  device_check: NoCheck' codegen/native_functions.yaml   # Only for pytorch 1.8.1

python3 -m codegen.gen_backend_stubs  \
  --output_dir="$NDIR/torch_npu/csrc/aten/" \
  --source_yaml="$NDIR/torch_npu/csrc/aten/npu_native_functions.yaml" \
  --impl_path="$NDIR/torch_npu/csrc/aten"  # Used to double-check the yaml file definitions.

if [ $? -ne 0 ]; then
  echo "Failed to generate NPU backend stubs."
  exit 1
fi

mv -f codegen/native_functions.yaml_bk codegen/native_functions.yaml

python3 -m codegen.gen_python_functions  \
  --output_dir="$NDIR/torch_npu/csrc/" \
  --source_yaml="$NDIR/torch_npu/csrc/aten/npu_native_functions.yaml" \
  --template_path="$NDIR/scripts/codegen/templates"

if [ $? -ne 0 ]; then
  echo "Failed to generate python bindings."
  exit 1
fi