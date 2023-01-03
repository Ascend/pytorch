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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& NPUNativeFunctions::linspace_out(const at::Scalar& start, const at::Scalar& end, int64_t steps, at::Tensor& result) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  at::Tensor r = result.is_contiguous() ? result : result.contiguous();
  r = NPUNativeFunctions::npu_dtype_cast(r, at::kFloat);
  if(steps == 0){
    // skip
  } else if (steps == 1) {
    r.fill_(start);
  } else {
    c10::SmallVector<int64_t, N> sizeVec = {steps};
    OpCommand cmd;
    cmd.Name("LinSpace")
        .Input(start, at::ScalarType::Float)
        .Input(end, at::ScalarType::Float)
        .Input(sizeVec, at::ScalarType::Int)
        .Output(r)
        .Run();
  }

  if(r.dtype() != result.dtype()) {
    r = r.to(result.dtype());
  }

  return result.copy_(r);
}

at::Tensor NPUNativeFunctions::linspace(const at::Scalar& start,const at::Scalar& end,
    int64_t steps,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt
) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  auto device =  device_or_default(device_opt);
  at::TensorOptions option;
  option = option.dtype(dtype_opt)
                 .layout(layout_opt)
                 .device(device)
                 .pinned_memory(pin_memory_opt);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat({steps}, option, ACL_FORMAT_ND);
  at::Tensor resultCast = NPUNativeFunctions::npu_dtype_cast(result, at::kFloat);

  // calculate the output result of the NPU
  NPUNativeFunctions::linspace_out(start, end, steps, resultCast);

  if(option.dtype() != resultCast.dtype()) {
    resultCast = resultCast.to(option.dtype());
  }

  return resultCast;
}
} // namespace native
} // namespace at_npu
