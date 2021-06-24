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

#include <ATen/native/IndexingUtils.h>
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> index_npu_input(
    const Tensor& self,
    const Tensor& masksTensor,
    const TensorList& allDefinedIndices) {
  SmallVector<Tensor, N> inputs = {self, masksTensor};
  for (int i = 0; i < allDefinedIndices.size(); i++) {
    inputs.emplace_back(allDefinedIndices[i]);
  }
  return CalcuOpUtil::create_npu_input_tensor_desc(inputs);
}

SmallVector<NPUTensorDesc, N> index_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> index_npu_attr(const Tensor& self) {
  SmallVector<NPUAttrDesc, N> attrs = { };
  return attrs;
}

Tensor& index_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& masksTensor,
    TensorList allDefinedIndices) {
  // constructs the input and output NPUTensorDesc
  auto inputs = index_npu_input(self, masksTensor, allDefinedIndices);
  auto outputs = index_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = index_npu_attr(self);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Index", inputs, outputs, attrs);

  return result;
}

Tensor index_npu(const Tensor& self, TensorList indices) {
  checkIndexTensorTypes(indices);
  Tensor formatCastOfSelf = self.npu_format_cast(ACL_FORMAT_ND);

  // calculate the output size
  auto outputSize = index_npu_output_size(formatCastOfSelf, indices);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, formatCastOfSelf.options(), ACL_FORMAT_ND);

  // masks corresponds to indices. 0 indicates undefined tensor.
  SmallVector<int64_t, N> masks;
  std::vector<Tensor> allDefinedIndices;
  for (int64_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined()) {
      masks.emplace_back(1);
      allDefinedIndices.emplace_back(indices[i]);
    } else {
      masks.emplace_back(0);
    }
  }

  Tensor masksTensor = CalcuOpUtil::copy_tensor_host_to_device(
      from_blob(masks.data(), {masks.size()}, dtype(ScalarType::Long)));

  // calculate the output result of the NPU
  index_out_npu(result, formatCastOfSelf, masksTensor, allDefinedIndices);

  return result;
}
} // namespace native
} // namespace at