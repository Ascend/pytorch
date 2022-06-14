// Copyright (c) 2020, Huawei Technologies.All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ATen/native/npu/utils/OpAdapter.h"
#include <torch/csrc/autograd/record_function.h>
#include "ATen/native/npu/common/InnerNpuNativeFunction.h"
#include "ATen/native/npu/frame/StorageDescHelper.h"
#include "ATen/native/npu/utils/NpuStorageOffsetGuard.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& stride_copy_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    IntArrayRef shape,
    IntArrayRef stride,
    Scalar storage_offset) {
  if ((result.nbytes() < 32) && (!StorageDescHelper::MetaDataAreMatch(&result))) {
    // [算子限制] 对于1. 小于一个block的数据搬运 2.result不match，Astrided暂不支持。
    copy_kernel_npu(result, self, false);
    return result;
  }
  RECORD_HOST_FUNCTION("npuAsStrided", std::vector<c10::IValue>({self}));
  E2E_RECORD_FUNCTION("npuAsStrided");
  // Set the offset of input discontiguous tensor to be 0.
  // The accurate offset would be provided as a attr to op. 
  OpCommand cmd;
  if (c10::npu::NpuRunMode::IsGraphMode()) {
    NpuStorageOffsetGuard guard_input(const_cast<Tensor &>(self));
    cmd.Name("AsStrided")
        .InputWithoutContiguous(self)
        .Input(shape)
        .Input(stride)
        .Input(storage_offset, at::kLong, CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
        .Output(result)
        .Run();
    return result;
  }
  // (Ascend) Fix multi-compiling of asstrided op by wrapping attr storage_offset as a NPU Tensor instead of GE Const node.
  // If GE Data node can pass vaule of storage_offset to op, we can switch storage_offset to Data node finally.
  cmd.Name("AsStrided")
      .InputWithoutContiguous(self)
      .Input(shape)
      .Input(stride)
      .InputScalarToNPUTensor(at::Scalar(0), at::kLong)
      .Output(result)
      .Run();
  return result;
}

Tensor& stride_copy_out_npu(
    Tensor& result,
    const Tensor& self,
    IntArrayRef shape,
    IntArrayRef stride,
    Scalar storage_offset) {
  stride_copy_out_npu_nocheck(result, self, shape, stride, storage_offset);
  return result;
}

Tensor stride_copy_npu(
    const Tensor& self,
    IntArrayRef shape,
    IntArrayRef stride,
    Scalar storage_offset) {
  // AsStrided OP only supports ND input
  Tensor result = OpPreparation::ApplyTensorWithFormat(
      shape, self.options(), ACL_FORMAT_ND);
  stride_copy_out_npu_nocheck(result, self, shape, stride, storage_offset);
  return result;
}

} // namespace native
} // namespace at