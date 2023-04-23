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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& logspace_out_npu_nocheck(
    at::Scalar start,
    at::Scalar end,
    int64_t steps,
    double base,
    at::Tensor& result) {
  TORCH_CHECK(steps >= 0, "logspace requires non-negative steps, given steps is ", steps);
  if ((base <= 0) && ((!start.isIntegral(false)) || (!end.isIntegral(false)))) {
    std::cout << "Warning: start and end in logspace should both be int when base <= 0, "
              << "get type "
              << start.type()
              << " and"
              << end.type()
              << std::endl;
  }

  at::Tensor inputs;
  int64_t dtype = 0;
  auto result_type = result.scalar_type();
  if (result_type == at::ScalarType::Half) {
    inputs = NPUNativeFunctions::npu_dtype_cast(
        at::arange(0, steps, at::device(at_npu::key::NativeDeviceType)),
        at::kHalf);
    dtype = 0;
  } else if (result_type == at::ScalarType::Float) {
    inputs = at::arange(0, steps, at::device(at_npu::key::NativeDeviceType).dtype(at::kFloat));
    dtype = 1;
  } else {
    TORCH_CHECK(false, "logspace only support float32 and float16, given type is ", result_type);
  }

  OpCommand cmd;
  cmd.Name("LogSpaceD")
      .Input(inputs)
      .Output(result)
      .Attr("start", start)
      .Attr("end", end)
      .Attr("steps", steps)
      .Attr("base", static_cast<float>(base))
      .Attr("dtype", dtype)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::logspace_out(
    const at::Scalar& start,
    const at::Scalar& end,
    int64_t steps,
    double base,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      { },
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      {steps});
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({},{result})
    .Func([&start, &end, &steps, &base](at::Tensor& result)
    {logspace_out_npu_nocheck(start, end, steps, base, result);})
    .Call(result);
  return result;
}

at::Tensor NPUNativeFunctions::logspace(
    const at::Scalar& start,
    const at::Scalar& end,
    int64_t steps,
    double base,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  auto device = device_or_default(device_opt);
  at::TensorOptions options;
  options = options.dtype(dtype_opt)
                   .layout(layout_opt)
                   .device(device)
                   .pinned_memory(pin_memory_opt);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat({steps}, options, ACL_FORMAT_ND);
  return logspace_out_npu_nocheck(start, end, steps, base, result);
}
}
}
