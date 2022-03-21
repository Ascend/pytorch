// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& elu_backward_out_npu(at::Tensor& grad_input, const at::Tensor& grad_output, at::Scalar alpha, at::Scalar scale, at::Scalar input_scale, const at::Tensor& output) {
    float value = CalcuOpUtil::get_scalar_float_value(alpha);
    OpCommand cmd;
    cmd.Name("EluGradV2")
       .Input(grad_output)
       .Input(output)
       .Output(grad_input)
       .Attr("alpha", value)
       .Run();
    return grad_input;
}
at::Tensor NPUNativeFunctions::elu_backward(const at::Tensor& grad_output, at::Scalar alpha, at::Scalar scale, at::Scalar input_scale, bool is_result, const at::Tensor& output) {
    at::Tensor result = OpPreparation::ApplyTensor(grad_output);
    elu_backward_out_npu(result, grad_output, alpha, scale, input_scale, output);
    return result;
}
} // namespace native
} // namespace at_npu
