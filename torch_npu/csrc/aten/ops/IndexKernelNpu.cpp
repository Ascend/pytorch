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
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& index_out_nocheck_npu(
    const at::Tensor& self,
    const at::Tensor& masksTensor,
    const at::TensorList& allDefinedIndices,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Index")
      .Input(self)
      .Input(masksTensor);
  for (int i = 0; i < allDefinedIndices.size(); i++) {
    cmd.Input(allDefinedIndices[i]);
  }
  cmd.Output(result)
      .Run();
  return result;
}

at::Tensor NPUNativeFunctions::index(const at::Tensor& self, const torch::List<c10::optional<at::Tensor>>& orig) {  
  checkIndexTensorTypes(orig);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
  auto indices = expandTensors(self, orig);
  at::Tensor formatCastOfSelf = self.npu_format_cast(ACL_FORMAT_ND);

  // calculate the output size
  auto outputSize = index_npu_output_size(formatCastOfSelf, indices);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(formatCastOfSelf,  outputSize, ACL_FORMAT_ND);

  // masks corresponds to indices. 0 indicates undefined tensor.
  SmallVector<int64_t, N> masks;
  std::vector<at::Tensor> allDefinedIndices;
  for (int64_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined()) {
      masks.emplace_back(1);
      allDefinedIndices.emplace_back(indices[i]);
    } else {
      masks.emplace_back(0);
    }
  }

  at::Tensor masksTensor = CalcuOpUtil::copy_tensor_host_to_device(
      from_blob(masks.data(), {masks.size()}, dtype(at::ScalarType::Long)));

  // calculate the output result of the NPU
  index_out_nocheck_npu(formatCastOfSelf, masksTensor, allDefinedIndices, result);

  return result;
}

} // namespace native
} // namespace at_npu
