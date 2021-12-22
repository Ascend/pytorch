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

namespace at {
namespace native {
using namespace at::native::npu;

namespace {
SmallVector<int64_t, SIZE> _embedding_bag_npu_output_size(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets) {
  SmallVector<int64_t, SIZE> outputSize = {};
  if (indices.dim() == 1) {
    outputSize = {offsets.size(0), weight.size(1)};
  } else {
    outputSize = {indices.size(0), weight.size(1)};
  }
  return outputSize;
} // _embedding_bag_npu_output_size

string get_mode_str(bool mode) {
  string modeStr = "mean";
  if (mode == 0) {
    modeStr = "sum";
  } else if (mode == 1) {
    modeStr = "mean";
  } else {
    modeStr = "max";
  }
  return modeStr;
} // get_mode_str

} // namespace

tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_npu(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const Tensor& per_sample_weights,
    bool include_last_offset) {
  Tensor weight_cpu = weight.to("cpu").requires_grad_();
  Tensor indices_cpu = indices.to("cpu");
  Tensor offsets_cpu = offsets.to("cpu");
  Tensor per_sample_weights_cpu = per_sample_weights;
  if (per_sample_weights_cpu.defined()) {
    Tensor per_sample_weights_cpu = per_sample_weights_cpu.to("cpu");
  }
  
  auto result = _embedding_bag_cpu(weight_cpu, indices_cpu, offsets_cpu, scale_grad_by_freq, mode, sparse, per_sample_weights_cpu, include_last_offset);
  
  Tensor output = std::get<0>(result).to(weight.device());
  Tensor offset2bag = std::get<1>(result).to(weight.device());
  Tensor bag_size = std::get<2>(result).to(weight.device());
  Tensor max_indices = std::get<3>(result).to(weight.device());
  
  return std::tie(output, offset2bag, bag_size, max_indices);
}

} // namespace native
} // namespace at