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
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& lerp_out_npu(Tensor& result, const Tensor& self, const Tensor& end, const Tensor& weight) {
    OpCommand cmd;
    cmd.Name("Lerp")
    .Input(self)
    .Input(end)
    .Input(weight)
    .Output(result)
    .Run();

    return result;
}

Tensor& lerp_out_npu(Tensor& result, const Tensor& self, const Tensor& end, Scalar weight) {
    OpCommand cmd;
    cmd.Name("Lerp")
    .Input(self)
    .Input(end)
    .Input(weight, self.scalar_type())
    .Output(result)
    .Run();

    return result;
}

Tensor lerp_npu(const Tensor& start, const Tensor& end, const Tensor& weight) {
    // calculate the output size
    auto outputSize = input_same_output_size(start);

    // construct the output tensor of the NPU
    Tensor result = at::empty_with_format(
        outputSize, start.options(), CalcuOpUtil::get_tensor_npu_format(start));

    // calculate the output result of the NPU
    lerp_out_npu(result, start, end, weight);

    return result;
}

Tensor lerp_npu(const Tensor& start, const Tensor& end, Scalar weight) {
    // calculate the output size
    auto outputSize = input_same_output_size(start);

    // construct the output tensor of the NPU
    Tensor result = at::empty_with_format(
        outputSize, start.options(), CalcuOpUtil::get_tensor_npu_format(start));

    // calculate the output result of the NPU
    lerp_out_npu(result, start, end, weight);

    return result;
}

Tensor& lerp_npu_(Tensor& self, const Tensor& end, const Tensor& weight) {
    SmallVector<Tensor, N> inputs = {self, end, weight};
    SmallVector<Tensor, N> outputs = {self};
    CalcuOpUtil::check_memory_over_laps(inputs, outputs);

    lerp_out_npu(self, self, end, weight);
    return self;
}

Tensor& lerp_npu_(Tensor& self, const Tensor& end, Scalar weight) {
    SmallVector<Tensor, N> inputs = {self, end};
    SmallVector<Tensor, N> outputs = {self};
    CalcuOpUtil::check_memory_over_laps(inputs, outputs);

    lerp_out_npu(self, self, end, weight);
    return self;
}

} // namespace native
} // namespace at
