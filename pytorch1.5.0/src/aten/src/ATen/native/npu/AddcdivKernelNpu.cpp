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

Tensor& addcdiv_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  
  OpCommand cmd;
  if (c10::npu::OptionsManager::CheckDynamicOptimizer("ADDCDIV")) {
    cmd.Name("Addcdiv")
      .Input(self)
      .Input(tensor1)
      .Input(tensor2)
      .Input(value, self.scalar_type(), MemoryType::MEMORY_HOST)
      .Output(result)
      .Run();
  } else {
    cmd.Name("Addcdiv")
      .Input(self)
      .Input(tensor1)
      .Input(tensor2)
      .Input(value, self.scalar_type())
      .Output(result)
      .Run();
  }

  return result;
}

Tensor addcdiv_npu(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  auto divOutputSize = broadcast_ops_npu_output_size(tensor1, tensor2);
  auto outputSize = broadcast_ops_npu_output_size(self.sizes(), divOutputSize);
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  addcdiv_out_npu(result, self, tensor1, tensor2, value);

  return result;
}

Tensor& addcdiv_npu_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  OpPreparation::CheckMemory({self, tensor1, tensor2}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = addcdiv_out_npu(contiguousSelf, contiguousSelf, tensor1, tensor2, value);
    NpuUtils::format_fresh_view(self, result);
  } else {
    addcdiv_out_npu(self, self, tensor1, tensor2, value);
  }
  return self;
}

} // namespace native
} // namespace at