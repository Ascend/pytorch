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

#include <ATen/native/npu/graph/util/GraphModeGuard.h>
#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/AdvancedIndex.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& index_out_npu(
    Tensor& result,
    const Tensor& self,
    IntArrayRef indexed_sizes,
    IntArrayRef indexed_strides,
    const TensorList& indices) {
  OpCommand cmd;
  cmd.Name("Index")
      .Input(self)
      .Input(indexed_sizes)
      .Input(indexed_strides);
  for (int i = 0; i < indices.size(); i++) {
    string inputName = "indices" + to_string(i);
    cmd.Input(indices[i], inputName);
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
  // Index demands self contiguous and matchs info.indexed_sizes, info.indexed_strides
  Tensor contiguousSelf = NpuUtils::format_contiguous(self);
  auto info = AdvanceIndex::make_info(contiguousSelf, indices);
  auto outputSize = index_npu_output_size(contiguousSelf, indices);
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  index_out_npu(result, contiguousSelf, info.indexed_sizes, info.indexed_strides, info.indices);
  return result;
}

} // namespace native
} // namespace at