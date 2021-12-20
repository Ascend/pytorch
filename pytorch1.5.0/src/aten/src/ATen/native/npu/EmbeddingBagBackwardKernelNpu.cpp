// Copyright (c) 2021 Huawei Technologies Co., Ltd
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

Tensor _embedding_bag_backward_npu(
    const Tensor& grad,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& offset2bag,
    const Tensor& bag_size,
    const Tensor& maximum_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const Tensor& per_sample_weights) {

  Tensor grad_cpu = grad.to("cpu");
  Tensor indices_cpu = indices.to("cpu");
  Tensor offsets_cpu = offsets.to("cpu");
  Tensor offset2bag_cpu = offset2bag.to("cpu");
  Tensor bag_size_cpu = bag_size.to("cpu");
  Tensor maximum_indices_cpu = maximum_indices.to("cpu");
  Tensor per_sample_weights_cpu = per_sample_weights;
  if (per_sample_weights_cpu.defined()) {
    Tensor per_sample_weights_cpu = per_sample_weights_cpu.to("cpu");
  }

  Tensor result = at::_embedding_bag_backward(
      grad_cpu, indices_cpu, offsets_cpu, offset2bag_cpu, bag_size_cpu, 
      maximum_indices_cpu, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights_cpu);
  
  result = at::native::sparse_to_dense(result);
  result = result.to(indices.device());

  return result;
}

} // namespace native
} // namespace at