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

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> gather_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> gather_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> gather_npu_attr(int64_t dim) {
  NPUAttrDesc npuAttrDim = NPUAttrDesc("dim", dim);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrDim};

  return attrs;
}

Tensor& gather_out_npu(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    bool sparse_grad) {
  Tensor formatCastOfSelf = self;
  Tensor formatCastOfIndex = index;
  Tensor resultCopy = result;
  if (self.scalar_type() == ScalarType::Half) {
    formatCastOfSelf = self.to(ScalarType::Float);
    resultCopy = result.to(ScalarType::Float);
  }

  if (self.scalar_type() == at::kLong) {
    TORCH_WARN_ONCE("The oprator of gather is executed, Currently High Accuracy but Low Performance OP with 64-bit has been used,"
      "Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
  }
  
  OpCommand cmd;
  cmd.Name("GatherElements")
      .Input(formatCastOfSelf)
      .Input(formatCastOfIndex)
      .Attr("dim", dim)
      .Output(resultCopy)
      .Run();

  result.copy_(resultCopy);

  return result;
}

Tensor& gather_out_npu(
    Tensor& result,
    const Tensor& self,
    Dimname dim,
    const Tensor& index,
    bool sparse_grad) {
  return gather_out_npu(result, self, dimname_to_position(self, dim), index, sparse_grad);
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
  gather_out_npu(result, self, dim, index, sparse_grad);

  return result;
}

Tensor gather_npu(
    const Tensor& self,
    Dimname dim,
    const Tensor& index,
    bool sparse_grad) {
  return gather_npu(self, dimname_to_position(self, dim), index, sparse_grad);
}

} // namespace native
} // namespace at