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

#include <ATen/record_function.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/framework/utils/NpuStorageOffsetGuard.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

at::Tensor& stride_copy_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef shape,
    at::IntArrayRef stride,
    at::Scalar storage_offset) {
  if ((result.nbytes() < 32) && (!StorageDescHelper::MetaDataAreMatch(&result))) {
    // [算子限制] 对于1. 小于一个block的数据搬运 2.result不match，Astrided暂不支持。
    copy_kernel_npu(result, self, false);
    return result;
  }
  // Set the offset of input discontiguous tensor to be 0.
  // The accurate offset would be provided as a attr to op. 
  NpuStorageOffsetGuard guard_input(const_cast<at::Tensor &>(self));
  RECORD_FUNCTION("npuAsStrided", std::vector<c10::IValue>({self}));
  OpCommand cmd;
  cmd.Name("AsStrided")
      .InputWithoutContiguous(self)
      .Input(shape)
      .Input(stride)
      .Input(storage_offset, at::kLong, CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::npu_stride_copy_out(
    const at::Tensor& self,
    c10::IntArrayRef shape,
    c10::IntArrayRef stride,
    const c10::Scalar& storage_offset,
    at::Tensor& result) {
  stride_copy_out_npu_nocheck(result, self, shape, stride, storage_offset);
  return result;
}

at::Tensor NPUNativeFunctions::npu_stride_copy(
    const at::Tensor& self,
    c10::IntArrayRef shape,
    c10::IntArrayRef stride,
    const c10::Scalar& storage_offset) {
  // AsStrided OP only supports ND input
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      shape, self.options(), ACL_FORMAT_ND);
  stride_copy_out_npu_nocheck(result, self, shape, stride, storage_offset);
  return result;
}

} // namespace native
} // namespace at_npu