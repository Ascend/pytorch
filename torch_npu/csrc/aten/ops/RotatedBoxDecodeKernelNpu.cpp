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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
    
at::Tensor NPUNativeFunctions::npu_rotated_box_decode(
    const at::Tensor& self, 
    const at::Tensor& deltas, 
    const at::Tensor& weight){
  at::Tensor result = OpPreparation::ApplyTensor(self);
  at::Tensor weightContiguous = weight.to(at::Device(at::kCPU), at::kFloat);
  at::ArrayRef<float> weightList(weightContiguous.data_ptr<float>(), weightContiguous.numel());  
  
  OpCommand cmd;
  cmd.Name("RotatedBoxDecode")
      .Input(self)
      .Input(deltas)
      .Output(result)
      .Attr("weight", weightList)
      .Run();   
  return result;  
}    
} // namespace native
} // namespace at_npu