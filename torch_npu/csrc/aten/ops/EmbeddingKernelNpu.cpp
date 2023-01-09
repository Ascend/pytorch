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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& embedding_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& weight,
    const at::Tensor& indices) {
  c10::SmallVector<int64_t, N> dimVec = {0};
  int64_t batch_dims = 0;

  OpCommand cmd;
  cmd.Name("GatherV2")
     .Input(weight)
     .Input(indices)
     .Input(dimVec)
     .Output(result)
     .Attr("batch_dims", batch_dims)
     .Run();

return result;

}

at::Tensor NPUNativeFunctions::embedding(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  // calculate the output size
  auto outputSize = array_to_small_vector(indices.sizes());
  outputSize.emplace_back(weight.size(weight.dim() - 1));
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      weight.options(),
      CalcuOpUtil::get_tensor_npu_format(weight));

  // calculate the output resugt of the NPU
  embedding_out_npu_nocheck(result, weight, indices);
  return result;
}
} // namespace native
} // namespace at_npu
