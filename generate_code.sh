#!/bin/bash

set -e

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd $CDIR

python_execute="$1"
pytorch_version="$2"

IFS='.' read -ra version_parts <<< "$pytorch_version"

pytorch_dir="v${version_parts[0]}r${version_parts[1]}"

# Fetch ACL headers from submodule paths
ACL_DEST="$CDIR/third_party/acl/inc/acl"
echo " --- Fetching ACL headers from submodules..."

ACL_SRC="$CDIR/third_party/acl_src"

# Copy runtime headers (lower priority, copied first)
if [ -d "$ACL_SRC/runtime/include/external/acl" ]; then
    mkdir -p "$ACL_DEST"
    cp -r "$ACL_SRC/runtime/include/external/acl/"* "$ACL_DEST/"
    echo " --- Copied runtime acl headers"
fi

# Copy ge headers (higher priority, overwrites runtime)
if [ -d "$ACL_SRC/ge/inc/external/acl" ]; then
    mkdir -p "$ACL_DEST"
    cp -r "$ACL_SRC/ge/inc/external/acl/"* "$ACL_DEST/"
    echo " --- Copied ge acl headers"
fi

# Copy super_kernel.h from graph-autofusion
SUPER_KERNEL_SRC="$ACL_SRC/graph-autofusion/super_kernel/include/super_kernel/super_kernel.h"
if [ -f "$SUPER_KERNEL_SRC" ]; then
    cp "$SUPER_KERNEL_SRC" "$ACL_DEST/super_kernel.h"
    echo " --- Copied super_kernel.h"
fi

# Clean up submodule working directories
rm -rf "$ACL_SRC"
echo " --- Cleaned up acl_src submodule directories"
echo " --- ACL headers fetched successfully"

file=$CDIR/third_party/op-plugin/gencode.sh

if [ -f "${file}" ]; then
  bash ${file} ${pytorch_version} ${python_execute}
fi

op_plugin_config_path=$CDIR/third_party/op-plugin/op_plugin/config/$pytorch_dir
source_yaml="$CDIR/torch_npu/csrc/aten/npu_native_functions.yaml"
testing_source_yaml="$CDIR/test/ops_unsupport_list.yaml"

op_plugin_functions_yaml_path="$op_plugin_config_path/npu_native_functions.yaml"

${python_execute} -m torchnpugen.gen_backend_stubs  \
  --output_dir="torch_npu/csrc/aten" \
  --source_yaml="$source_yaml" \
  --impl_path="$CDIR/torch_npu/csrc/aten" \
  --op_plugin_impl_path="$CDIR/third_party/op-plugin/op_plugin/ops" \
  --op_plugin_yaml_path="$op_plugin_config_path/op_plugin_functions.yaml"

${python_execute} -m torchnpugen.autograd.gen_autograd \
  --out_dir="$CDIR/torch_npu/csrc/aten" \
  --autograd_dir="$CDIR/torchnpugen/autograd" \
  --npu_native_function_dir="$source_yaml"

${python_execute} -m torchnpugen.codegen_ops_info

if [ -f $CDIR/third_party/op-plugin/codegen/templates/_op_plugin_docs.py ]; then
  if [ -f $CDIR/torch_npu/_op_plugin_docs.py ]; then
      # remove _op_plugin_docs.py
      rm $CDIR/torch_npu/_op_plugin_docs.py
      echo "Existing _op_plugin_docs.py in torch_npu deleted."
  fi
  cp -f $CDIR/third_party/op-plugin/codegen/templates/_op_plugin_docs.py $CDIR/torch_npu/_op_plugin_docs.py
fi
