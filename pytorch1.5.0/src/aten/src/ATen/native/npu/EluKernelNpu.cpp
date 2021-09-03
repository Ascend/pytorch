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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& elu_out_npu(Tensor& result, const Tensor& self, Scalar alpha, Scalar scale, Scalar input_scale) 
{
    float alphaValue = CalcuOpUtil::get_scalar_float_value(alpha);
    float scaleValue = CalcuOpUtil::get_scalar_float_value(scale);
    float inputScaleValue = CalcuOpUtil::get_scalar_float_value(input_scale);

    OpCommand cmd;
    cmd.Name("Elu")
        .Input(self)
        .Output(result)
        .Attr("alpha", alphaValue)
        .Attr("scale", scaleValue)
        .Attr("input_scale", inputScaleValue)
        .Run();

    return result;
}

Tensor elu_npu(const Tensor& self, Scalar alpha, Scalar scale, Scalar input_scale) 
{
    // calculate the output size
    auto outputSize = input_same_output_size(self);

    // construct the output tensor of the NPU
    Tensor result = at::empty_with_format(outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  
    // calculate the output result of the NPU
    elu_out_npu(result, self, alpha, scale, input_scale);

    return result;
}

Tensor& elu_npu_(Tensor& self, Scalar alpha, Scalar scale, Scalar input_scale)
{
    SmallVector<Tensor, N> inputs = {self};
    SmallVector<Tensor, N> outputs = {self};
    CalcuOpUtil::check_memory_over_laps(inputs, outputs);
    
    if (!NpuUtils::check_match(&self)) {
        Tensor contiguousSelf = NpuUtils::format_contiguous(self);
        Tensor result = elu_out_npu(contiguousSelf, contiguousSelf, alpha, scale, input_scale);
        NpuUtils::format_fresh_view(self, result);
    } else {
        elu_out_npu(self, self, alpha, scale, input_scale);
    }

    return self;
}

} // namespace native
} // namespace at
