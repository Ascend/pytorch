// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

#include "torch_npu/csrc/framework/utils/AdvancedIndex.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
at::Tensor index_high_dims_op_api(const at::Tensor& self, std::vector<at::Tensor> indices) {
  std::vector<at::Tensor> all_defined_indices;
  at::SmallVector<int64_t, N> zeroSize = {0};
  at::Tensor emptyTensor = OpPreparation::ApplyTensorWithoutFormat(self, zeroSize);
  for (int i = 0; i < indices.size(); i++) {
    if (indices[i].defined()) {
      all_defined_indices.emplace_back(indices[i]);
      continue;
    }
    all_defined_indices.emplace_back(emptyTensor);
  }

  auto output_size = index_npu_output_size(self, indices);
  auto result = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  // calculate the output result of the NPU
  at::TensorList indices_tensor_list = all_defined_indices;
  EXEC_NPU_CMD(aclnnIndex, self, indices_tensor_list, result);

  return result;
}

at::Tensor NPUNativeOpApiFunctions::index(const at::Tensor& self, const torch::List<c10::optional<at::Tensor>>& orig) {
  DO_COMPATIBILITY(aclnnIndex, NPUNativeFunctions::index(self, orig));
  if (self.device().type() == at::kCPU) {
    return at::native::index(self, orig);
  }

  at::native::checkIndexTensorTypes(orig);
  auto indices = AdvanceIndex::npu_expand_tensors(self, orig);

  return index_high_dims_op_api(self, indices);
}

} // namespace native
} // namespace at_npu

