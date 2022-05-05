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
#include <ATen/native/npu/graph/util/GraphModeGuard.h>
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& index_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& masksTensor,
    TensorList allDefinedIndices) {
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

Tensor index_npu(const Tensor& self, TensorList indices) {
  /**
   * In the cann framework, index operator belongs to the fourth type of
   * operator, which means that the execution of the index operator must go
   * through the dynamic shape execution framework. In this case, constructing
   * a large dynamic shape graph is not beneficial to the overall execution
   * performance, because more dynamic shape operators are introduced.
   * Therefore, when the fourth type of operator is encountered in graph
   * mode, the single op mode is switched to execute by default.
   */
  GraphModeGuard mode_guard(c10::npu::ModeKind::SINGLE_OP_MODE);

  checkIndexTensorTypes(indices);
  Tensor formatCastOfSelf = self.npu_format_cast(ACL_FORMAT_ND);

  // calculate the output size
  auto outputSize = index_npu_output_size(formatCastOfSelf, indices);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensorWithFormat(
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