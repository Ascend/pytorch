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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& scatter_out_npu(
    Tensor& output,
    const Tensor& self,
    const Tensor& indices,
    const Tensor& updates,
    int64_t dim) {
  OpCommand cmd;
  cmd.Name("ArgMaxGrad")
      .Input(self)
      .Input(indices)
      .Input(updates)
      .Output(output)
      .Attr("dimension", dim)
      .Run();
  
  return output;
}

Tensor scatter_npu(const Tensor& self, const Tensor& indices, const Tensor& updates, int64_t dim) {
  Tensor outputs = at::empty_with_format(
      self.sizes(), self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  scatter_out_npu(outputs, self, indices, updates, dim);

  return outputs;
}


} // namespace native
} // namespace at
