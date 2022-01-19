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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& gather_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    bool sparse_grad) {
  Tensor dtypeCastOfSelf = self;
  Tensor resultCopy = result;
  if (self.scalar_type() == ScalarType::Half) {
    dtypeCastOfSelf = dtypeCastOfSelf.to(ScalarType::Float);
    resultCopy = resultCopy.to(ScalarType::Float);
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
      .Output(resultCopy)
      .Run();

  result.copy_(resultCopy);
  return result;
}

Tensor& gather_out_npu(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    bool sparse_grad,
    Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  return gather_out_npu_nocheck(result, self, dim, index, sparse_grad);
}

Tensor& gather_out_dimname_npu(
    const Tensor& self,
    Dimname dim,
    const Tensor& index,
    bool sparse_grad,
    Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  return gather_out_npu_nocheck(result, self, dimname_to_position(self, dim), index, sparse_grad);
}

Tensor gather_npu(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    bool sparse_grad) {
  // calculate the output size
  auto outputSize = input_same_output_size(index);
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  // calculate the output result of the NPU
  gather_out_npu_nocheck(result, self, dim, index, sparse_grad);

  return result;
}

Tensor gather_dimname_npu(
    const Tensor& self,
    Dimname dim,
    const Tensor& index,
    bool sparse_grad) {
  return gather_npu(self, dimname_to_position(self, dim), index, sparse_grad);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("gather.out", TORCH_FN(gather_out_npu));
  m.impl("gather", TORCH_FN(gather_npu));
  m.impl("gather.dimname", TORCH_FN(gather_dimname_npu));
  m.impl("gather.dimname_out", TORCH_FN(gather_out_dimname_npu));
}

} // namespace native
} // namespace at