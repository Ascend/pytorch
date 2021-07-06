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

#include "c10/npu/OptionsManager.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at { 
namespace native {
using namespace at::native::npu;

Tensor flip_npu(const Tensor& self, IntArrayRef dims){
    Tensor result = OpPreparation::ApplyTensor(self);
    SmallVector<int64_t,N> dimVec = array_to_small_vector(dims);
    if (!c10::npu::OptionsManager::CheckDynamicEnable()) {  
      OpCommand cmd;
      cmd.Name("ReverseV2") 
        .Input(self) 
        .Input(dimVec, at::kLong) 
        .Output(result) 
        .Run();
    } else {
      OpDynamicCommand cmd;
      cmd.Name("ReverseV2D")
        .Input(self)
        .Output(result)
        .Attr("axis", dims);
      cmd.DynamicName("ReverseV2")
        .DynamicInput(self)
        .DynamicInput(dimVec, at::kLong, at::kInt, "axis")
        .DynamicOutput(result)
        .DynamicOpRun();
    }
    return result;
}

}
}