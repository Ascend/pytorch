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

Tensor& atan_out_npu(Tensor& result, const Tensor& self) { 
  OpCommand cmd;
  cmd.Name("Atan")
     .Input(self)
     .Output(result)
     .Run();
  return result;  
}
 
Tensor atan_npu(const Tensor& self) { 
  Tensor result = OpPreparation::ApplyTensor(self);
  //calculate the output result of the NPU 
  atan_out_npu(result, self);  
  return result; 
} 
 
Tensor& atan_npu_(Tensor& self) { 
  OpPreparation::CheckMemory({self}, {self});
  if (!NpuUtils::check_match(&self)) { 
    Tensor contiguousSelf = NpuUtils::format_contiguous(self); 
    Tensor result = atan_out_npu(contiguousSelf, contiguousSelf); 
    NpuUtils::format_fresh_view(self, result); 
  } else {
    atan_out_npu(self, self); 
  }
  return self;
}
 
}} // namespace at::native