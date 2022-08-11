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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/XLANativeFunctions.h"
#include <ATen/record_function.h>

namespace at_npu {
namespace native {

at::Tensor& stride_copy_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef shape,
    at::IntArrayRef stride,
    const at::Scalar& storage_offset) {
  RECORD_FUNCTION("npuAsStrided", std::vector<c10::IValue>({self}));
  OpCommand cmd;
  cmd.Name("AsStrided")
      .InputWithoutContiguous(self)
      .Input(shape)
      .Input(stride)
      .Input(storage_offset, at::kLong)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& XLANativeFunctions::npu_stride_copy_out(
    const at::Tensor& self,
    c10::IntArrayRef shape,
    c10::IntArrayRef stride,
    const c10::Scalar& storage_offset,
    at::Tensor& result) {
  stride_copy_out_npu_nocheck(result, self, shape, stride, storage_offset);
  return result;
}

at::Tensor XLANativeFunctions::npu_stride_copy(
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