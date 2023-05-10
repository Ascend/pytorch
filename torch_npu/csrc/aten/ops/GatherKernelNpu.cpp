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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& gather_out_npu_nocheck(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& result) {
  at::Tensor dtypeCastOfSelf = self;
  at::Tensor dtypeCastOfResult = result;
  if (self.scalar_type() == at::ScalarType::Half) {
    dtypeCastOfSelf = NPUNativeFunctions::npu_dtype_cast(dtypeCastOfSelf, at::ScalarType::Float);
    dtypeCastOfResult = NPUNativeFunctions::npu_dtype_cast(dtypeCastOfResult, at::ScalarType::Float);
  }

  if (self.scalar_type() == at::kLong) {
    TORCH_WARN_ONCE("The oprator of gather is executed, Currently High Accuracy but Low Performance OP"
      "with 64-bit has been used,Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
  }

  OpCommand cmd;
  cmd.Name("GatherElements")
      .Input(dtypeCastOfSelf)
      .Input(index)
      .Attr("dim", dim)
      .Output(dtypeCastOfResult)
      .Run();
  result.copy_(dtypeCastOfResult);
  return result;
}

at::Tensor& NPUNativeFunctions::gather_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& result) {
  auto outputSize = index.sizes();
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);
  return gather_out_npu_nocheck(self, dim, index, sparse_grad, result);
}

at::Tensor& NPUNativeFunctions::gather_out(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& result) {
  auto outputSize = index.sizes();
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);
  return gather_out_npu_nocheck(self, dimname_to_position(self, dim), index, sparse_grad, result);
}

at::Tensor NPUNativeFunctions::gather(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad) {
  auto outputSize = input_same_output_size(index);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  gather_out_npu_nocheck(self, dim, index, sparse_grad, result);
  return result;
}

at::Tensor NPUNativeFunctions::gather(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    bool sparse_grad) {
  return gather(self, dimname_to_position(self, dim), index, sparse_grad);
}
} // namespace native
} // namespace at_npu