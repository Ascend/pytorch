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

namespace at_npu {
namespace native {

at::Tensor& index_out_nocheck_npu(
    const at::Tensor& self,
    at::IntArrayRef indexed_sizes,
    at::IntArrayRef indexed_strides,
    const at::TensorList& indices,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Index")
      .Input(self)
      .Input(indexed_sizes)
      .Input(indexed_strides);
  for (int i = 0; i < indices.size(); i++) {
    std::string name = "indices" + std::to_string(i);
    cmd.Input(indices[i], name);
  }
  cmd.Output(result)
      .Run();
  return result;
}

at::Tensor NPUNativeFunctions::index(const at::Tensor& self, const torch::List<c10::optional<at::Tensor>>& orig) {  
  // Index demands self contiguous and matchs info.indexed_sizes, info.indexed_strides
  at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
  auto info = AdvanceIndex::make_info(contiguousSelf, orig);
  auto indices = at::native::expandTensors(contiguousSelf, orig);
  auto outputSize = index_npu_output_size(contiguousSelf, indices);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  index_out_nocheck_npu(contiguousSelf, info.indexed_sizes, info.indexed_strides, info.indices, result);
  return result;
}

} // namespace native
} // namespace at_npu
