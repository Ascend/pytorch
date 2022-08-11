// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION. 
// All rights reserved.
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

namespace at_npu {
namespace native {

at::Tensor& addcmul_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar value) {
  OpCommand cmd;

  cmd.Name("Addcmul")
    .Input(self)
    .Input(tensor1)
    .Input(tensor2)
    .Input(value, self.scalar_type())
    .Output(result)
    .Run();

  return result;
}

at::Tensor& XLANativeFunctions::addcmul_out(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value,
    at::Tensor& result) {
  auto mulOutputSize = broadcast_ops_npu_output_size(tensor1, tensor2);
  auto outputSize = broadcast_ops_npu_output_size(self.sizes(), mulOutputSize);

  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self, tensor1, tensor2}, {result})
      .Func([&self, &tensor1, &tensor2, &value](at::Tensor& result)
      {addcmul_out_npu_nocheck(result, self, tensor1, tensor2, value);})
      .Call(result);
}

at::Tensor XLANativeFunctions::addcmul(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value) {
  auto mulOutputSize = broadcast_ops_npu_output_size(tensor1, tensor2);
  auto outputSize = broadcast_ops_npu_output_size(self.sizes(), mulOutputSize);

  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  addcmul_out_npu_nocheck(result, self, tensor1, tensor2, value);

  return result;
}

at::Tensor& XLANativeFunctions::addcmul_(
    at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value) {
  OpPreparation::CheckMemory({self}, {self});
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = addcmul_out_npu_nocheck(
        contiguousSelf, contiguousSelf, tensor1, tensor2, value);
    NpuUtils::format_fresh_view(self, result);
  } else {
    addcmul_out_npu_nocheck(self, self, tensor1, tensor2, value);
  }

  return self;
}

} // namespace native
} // namespace at_npu
