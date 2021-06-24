
CUR_DIR=$(dirname $(readlink -f $0))
ROOT_DIR=$CUR_DIR/..

# mkdir src
mkdir -p src/aten/src/ATen/native
mkdir -p src/aten/src/ATen/detail
mkdir -p src/aten/src/ATen/templates

mkdir -p src/c10

mkdir -p src/cmake/public

mkdir -p src/third_party

mkdir -p src/torch/contrib
mkdir -p src/torch/csrc/autograd
mkdir -p src/torch/csrc/utils
mkdir -p src/torch/lib/c10d
mkdir -p src/torch/utils

mkdir -p src/tools/autograd

mkdir -p temp/test

# move files
mv pytorch/aten/src/ATen/native/npu src/aten/src/ATen/native
mv pytorch/aten/src/THNPU src/aten/src
mv pytorch/aten/src/ATen/detail/NPU* src/aten/src/ATen/detail
mv pytorch/aten/src/ATen/npu src/aten/src/ATen
mv pytorch/aten/src/ATen/templates/NPU* src/aten/src/ATen/templates
cp pytorch/aten/src/ATen/native/native_functions.yaml src/aten/src/ATen/native
cp pytorch/tools/autograd/derivatives.yaml src/tools/autograd

mv pytorch/c10/npu src/c10

mv pytorch/cmake/public/npu.cmake src/cmake/public

mv pytorch/third_party/acl src/third_party
mv pytorch/third_party/hccl src/third_party

mv pytorch/torch/contrib/npu src/torch/contrib
mv pytorch/torch/csrc/autograd/profiler_npu.cpp src/torch/csrc/autograd
mv pytorch/torch/csrc/npu src/torch/csrc
mv pytorch/torch/csrc/utils/npu_* src/torch/csrc/utils
mv pytorch/torch/npu src/torch
mv pytorch/torch/lib/c10d/HCCL* src/torch/lib/c10d
mv pytorch/torch/lib/c10d/ProcessGroupHCCL* src/torch/lib/c10d

mv pytorch/env.sh src
mv pytorch/build.sh src # where
mv pytorch/README.en.md src # no need
mv pytorch/README.zh.md src # no need

## fuzzy compile
mv pytorch/aten/src/ATen/native/GlobalStep* src/aten/src/ATen/native

## dump util
mv pytorch/aten/src/ATen/utils src/aten/src/ATen
mv pytorch/torch/utils/dumper.py src/torch/utils

# end
mv src temp
mv pytorch/test/test_npu temp/test
mv pytorch/access_control_test.py temp
