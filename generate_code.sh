#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd $CDIR

build_libtorch="$1"

python3 -m codegen.gen_backend_stubs  \
  --output_dir="torch_npu/csrc/aten" \
  --source_yaml="$CDIR/torch_npu/csrc/aten/npu_native_functions.yaml" \
  --impl_path="$CDIR/torch_npu/csrc/aten"  # Used to double-check the yaml file definitions.

if [[ ${build_libtorch} != "True" ]]; then
  python3 -m codegen.gen_python_functions  \
    --output_dir="$CDIR/torch_npu/csrc/aten/" \
    --source_yaml="$CDIR/torch_npu/csrc/aten/npu_native_functions.yaml" \
    --template_path="$CDIR/codegen/templates"
fi
