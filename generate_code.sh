#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd $CDIR

python_execute="$1"

${python_execute} -m codegen.gen_backend_stubs  \
  --output_dir="torch_npu/csrc/aten" \
  --source_yaml="$CDIR/torch_npu/csrc/aten/npu_native_functions.yaml" \
  --impl_path="$CDIR/torch_npu/csrc/aten" \
  --op_plugin_impl_path="$CDIR/third_party/op-plugin/op_plugin/ops" # Used to double-check the yaml file definitions.

${python_execute} -m codegen.autograd.gen_autograd \
  --out_dir="$CDIR/torch_npu/csrc/aten" \
  --autograd_dir="$CDIR/codegen/autograd" \
  --npu_native_function_dir="$CDIR/torch_npu/csrc/aten/npu_native_functions.yaml"
