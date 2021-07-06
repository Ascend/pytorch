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
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "c10/npu/OptionsManager.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& embedding_out_npu(
    Tensor& result,
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  SmallVector<int64_t, N> dimVec = {0};
  if (!c10::npu::OptionsManager::CheckDynamicEnable()) {
    OpCommand cmd;
    cmd.Name("GatherV2")
        .Input(weight)
        .Input(indices)
        .Input(dimVec, at::kInt)
        .Output(result)    
        .Run();
  } else {
    OpDynamicCommand cmd;
    cmd.Name("GatherV2D")
        .Input(weight)
        .Input(indices)
        .Output(result)
        .Attr("axis", (int64_t)0);
    Tensor dimTensor_cpu = from_blob((void*)dimVec.data(), {1}, at::kLong).to(at::kInt);
    Tensor dimTensor_npu = CalcuOpUtil::copy_tensor_host_to_device(dimTensor_cpu);
    cmd.DynamicName("GatherV2")
        .DynamicInput(weight)
        .DynamicInput(indices)
        .DynamicInput(dimTensor_npu, "axis")
        .DynamicOutput(result)
        .DynamicOpRun();
  }
  return result;
}

Tensor embedding_npu(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  // calculate the output size
  auto outputSize = array_to_small_vector(indices.sizes());
  outputSize.emplace_back(weight.size(weight.dim() - 1));
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize,
      weight.options(),
      CalcuOpUtil::get_tensor_npu_format(weight));

  // calculate the output resugt of the NPU
  embedding_out_npu(
      result, weight, indices, padding_idx, scale_grad_by_freq, sparse);
  return result;
}

} // namespace native
} // namespace at
