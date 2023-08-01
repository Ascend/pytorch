// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include <ATen/native/IndexingUtils.h>
#include <ATen/native/TypeProperties.h>

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/framework/utils/AdvancedIndex.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace at_npu {
namespace native {
at::Tensor NPUNativeOpApiFunctions::index_put(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    bool accumulate) {
  DO_COMPATIBILITY(aclnnIndexPutImpl,
                   NPUNativeFunctions::index_put(self, indices, value, accumulate));
  return self.clone(at::MemoryFormat::Contiguous)
      .index_put_(indices, value, accumulate);
}

at::Tensor& NPUNativeOpApiFunctions::index_put_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    const bool accumulate) {
  DO_COMPATIBILITY(aclnnIndexPutImpl,
                   NPUNativeFunctions::index_put_(self, indices, value, accumulate));
  return at::_index_put_impl_(
      self, indices, value, accumulate, false);
}

at::Tensor& NPUNativeOpApiFunctions::_index_put_impl_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  DO_COMPATIBILITY(aclnnIndexPutImpl,
                   NPUNativeFunctions::_index_put_impl_(self, indices, value, accumulate, unsafe));
  if (self.device().type() == at::kCPU) {
    return at::native::_index_put_impl_(self, indices, value, accumulate, unsafe);
  }
  auto indices_after = AdvanceIndex::npu_expand_tensors(self, indices, true);
  std::vector<at::Tensor> all_defined_indices;
  at::SmallVector<int64_t, N> zeroSize = {0};
  at::Tensor emptyTensor = OpPreparation::ApplyTensorWithoutFormat(self, zeroSize);
  for (int i = 0; i < indices_after.size(); i++) {
    if (indices_after[i].defined()) {
      all_defined_indices.emplace_back(indices_after[i]);
      continue;
    }
    all_defined_indices.emplace_back(emptyTensor);
  }

  for (auto &all_defined_indice : all_defined_indices) {
    if (all_defined_indice.device() != self.device()) {
      all_defined_indice = all_defined_indice.to(self.device());
    }
  }
  at::TensorList indices_tensor_list = all_defined_indices;
  if (self.numel() != 0 && value.numel() != 0) {
    EXEC_NPU_CMD(aclnnIndexPutImpl, self, indices_tensor_list, value, accumulate, unsafe);
  }
  return self;
}

} // namespace native
} // namespace at_npu

