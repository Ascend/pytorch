#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd $CDIR

python_execute="$1"

${python_execute} -m codegen.gen_backend_stubs  \
  --output_dir="torch_npu/csrc/aten" \
  --source_yaml="$CDIR/torch_npu/csrc/aten/npu_native_functions.yaml" \
  --impl_path="$CDIR/torch_npu/csrc/aten"  # Used to double-check the yaml file definitions.
