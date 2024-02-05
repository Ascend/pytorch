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

cd $CDIR

py_exec="$1"
pytorch_version="$2"

IFS='.' read -ra version_parts <<< "$pytorch_version"

pytorch_dir="v${version_parts[0]}r${version_parts[1]}"

file=$CDIR/third_party/op-plugin/gencode.sh

if [ -f "${file}" ]; then
  bash ${file} ${pytorch_version} ${py_exec}
fi

# impl_path is used to double-check the yaml file definitions.
# yaml_path is used to load opplugin api
${py_exec} -m codegen.gen_backend_stubs  \
  --output_dir="$CDIR/torch_npu/csrc/aten/" \
  --source_yaml="$CDIR/torch_npu/csrc/aten/npu_native_functions.yaml" \
  --impl_path="$CDIR/torch_npu/csrc/aten" \
  --op_plugin_impl_path="$CDIR/third_party/op-plugin/op_plugin/" \
  --op_plugin_yaml_path="$CDIR/third_party/op-plugin/op_plugin/config/$pytorch_dir/op_plugin_functions.yaml"

${py_exec} -m codegen.autograd.gen_autograd \
  --native_functions_dir="$CDIR/codegen/native_functions.yaml" \
  --out_dir="$CDIR/torch_npu/csrc/aten/" \
  --autograd_dir="$CDIR/codegen/autograd/" \
  --npu_native_function_dir="$CDIR/torch_npu/csrc/aten/npu_native_functions.yaml"

${py_exec} -m codegen.gen_python_functions  \
  --output_dir="$CDIR/torch_npu/csrc/aten/" \
  --source_yaml="$CDIR/torch_npu/csrc/aten/npu_native_functions.yaml" \
  --native_yaml="$CDIR/codegen/native_functions.yaml" \
  --template_path="$CDIR/codegen/templates" \
  --op_plugin_yaml_path="$CDIR/third_party/op-plugin/op_plugin/config/$pytorch_dir/op_plugin_functions.yaml"

if [ -f $CDIR/third_party/op-plugin/codegen/templates/_op_plugin_docs.py ]; then
  cp -f $CDIR/third_party/op-plugin/codegen/templates/_op_plugin_docs.py $CDIR/torch_npu/_op_plugin_docs.py
fi
