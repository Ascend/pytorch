#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd $CDIR

python_execute="$1"
pytorch_version="$2"

IFS='.' read -ra version_parts <<< "$pytorch_version"

pytorch_dir="v${version_parts[0]}r${version_parts[1]}"

file=$CDIR/third_party/op-plugin/gencode.sh

if [ -f "${file}" ]; then
  bash ${file} ${pytorch_version} ${python_execute}
fi

${python_execute} -m codegen.gen_backend_stubs  \
  --output_dir="torch_npu/csrc/aten" \
  --source_yaml="$CDIR/torch_npu/csrc/aten/npu_native_functions.yaml" \
  --impl_path="$CDIR/torch_npu/csrc/aten" \
  --op_plugin_impl_path="$CDIR/third_party/op-plugin/op_plugin/ops" \
  --op_plugin_yaml_path="$CDIR/third_party/op-plugin/op_plugin/config/$pytorch_dir/op_plugin_functions.yaml"

${python_execute} -m codegen.autograd.gen_autograd \
  --out_dir="$CDIR/torch_npu/csrc/aten" \
  --autograd_dir="$CDIR/codegen/autograd" \
  --npu_native_function_dir="$CDIR/torch_npu/csrc/aten/npu_native_functions.yaml"
