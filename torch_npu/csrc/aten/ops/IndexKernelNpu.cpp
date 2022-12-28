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
#include "torch_npu/csrc/framework/utils/AdvancedIndex.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/graph/util/GraphModeGuard.h"

namespace at_npu {
namespace native {

at::Tensor& index_out_nocheck_npu(
    const at::Tensor& self,
    const at::IntArrayRef masks,
    const at::TensorList& indices,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Index")
      .Input(self)
      .Input(masks, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Input(result.sizes(), at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT);
  for (int i = 0; i < indices.size(); i++) {
    std::string name = "indices" + std::to_string(i);
    cmd.Input(indices[i], name);
  }
  cmd.Output(result)
      .Attr("_exclude_engines", (string)"AiCore")
      .Run();
  return result;
}

at::Tensor NPUNativeFunctions::index(const at::Tensor& self, const torch::List<c10::optional<at::Tensor>>& orig) {
  /**
   * In the cann framework, index operator belongs to the fourth type of
   * operator, which means that the execution of the index operator must go
   * through the dynamic shape execution framework. In this case, constructing
   * a large dynamic shape graph is not beneficial to the overall execution
   * performance, because more dynamic shape operators are introduced.
   * Therefore, when the fourth type of operator is encountered in graph
   * mode, the single op mode is switched to execute by default.
   */
  GraphModeGuard mode_guard(c10_npu::ModeKind::SINGLE_OP_MODE);

  at::native::checkIndexTensorTypes(orig);
  // masks corresponds to indices. 0 indicates undefined tensor.
  at::SmallVector<int64_t, N> masks;
  std::vector<at::Tensor> allDefinedIndices;
  std::vector<at::Tensor> allValuedIndices;
  for (c10::optional<at::Tensor> index_opt : orig) {
    if (index_opt.has_value()) {
      at::Tensor index = std::move(*index_opt);
      allValuedIndices.emplace_back(index);
      if (index.defined()) {
        allDefinedIndices.emplace_back(index);
        masks.emplace_back(1);
      } else {
        masks.emplace_back(0);
      }
    } else {
      masks.emplace_back(0);
    }
  }
  at::Tensor formatCastOfSelf = NPUNativeFunctions::npu_format_cast(self, ACL_FORMAT_ND);
  auto outputSize = index_npu_output_size(formatCastOfSelf, allValuedIndices);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(formatCastOfSelf,  outputSize, ACL_FORMAT_ND);

  // calculate the output result of the NPU
  index_out_nocheck_npu(formatCastOfSelf, masks, allDefinedIndices, result);

  return result;
}

} // namespace native
} // namespace at_npu
