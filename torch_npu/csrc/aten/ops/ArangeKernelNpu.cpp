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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

// bool inputs are considered integral
static inline bool allIntegral(
    std::initializer_list<std::reference_wrapper<at::Scalar>> l) {
  for (at::Scalar& s : l) {
    if (!s.isIntegral(true)) {
      return false;
    }
  }
  return true;
}


at::Tensor& arange_out_npu_nocheck(
    at::Tensor& result,
    at::Scalar start,
    at::Scalar end,
    at::Scalar step) {
  OpCommand cmd;
  cmd.Name("Range")
     .Input(start, result.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
     .Input(end, result.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
     .Input(step, result.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
     .Output(result)
     .Run();

  return result;
}

at::Tensor NPUNativeFunctions::arange(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {

  c10::TensorOptions option = c10::TensorOptions().dtype(dtype_opt)
                                          .device(device_opt)
                                          .layout(layout_opt)
                                          .pinned_memory(pin_memory_opt);

  float start_value = CalcuOpUtil::get_scalar_float_value(start);
  float end_value = CalcuOpUtil::get_scalar_float_value(end);
  float step_value = CalcuOpUtil::get_scalar_float_value(step);

  // Check step start end
  TORCH_CHECK(step_value != 0, "step must be nonzero");
  TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
      "upper bound and larger bound inconsistent with step sign");
  at::Scalar start_opt = start;
  at::Scalar end_opt = end;
  at::Scalar step_opt = step;
  bool set_to_integral_dtype =
      !option.has_dtype() && allIntegral({start_opt, end_opt, step_opt});

  // check start == end
  if (set_to_integral_dtype) {
    option = option.dtype(at::ScalarType::Long);
  }
  at::Tensor result_check = OpPreparation::ApplyTensorWithFormat({0}, option, ACL_FORMAT_ND);

  if (start_value == end_value) {
    return result_check;
  }

  // calculate the output size
  double size_arange = std::ceil(static_cast<double>(end.toDouble() - start.toDouble())
                                 / step.toDouble());
  int64_t size_value = static_cast<int64_t>(size_arange);
  at::SmallVector<int64_t, SIZE> outputSize = {size_value};
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(outputSize, option, ACL_FORMAT_ND);

  if(option.dtype() == at::kHalf) {
    result = result.to(at::kFloat);
  }

  arange_out_npu_nocheck(result, start, end, step);

  if(option.dtype() == at::kHalf) {
    result = result.to(at::kHalf);
  }

  return result;
}

at::Tensor NPUNativeFunctions::arange(
    const at::Scalar& start, 
    const at::Scalar& end, 
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {

  return NPUNativeFunctions::arange(start, end, 1, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}


at::Tensor NPUNativeFunctions::arange(
    const at::Scalar& end, 
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {

  return NPUNativeFunctions::arange(0, end, dtype_opt, layout_opt, device_opt, pin_memory_opt);  // start = 0
}

at::Tensor& NPUNativeFunctions::arange_out(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& result) {
  float start_value = CalcuOpUtil::get_scalar_float_value(start);
  float end_value = CalcuOpUtil::get_scalar_float_value(end);
  float step_value = CalcuOpUtil::get_scalar_float_value(step);

  // Check step start end
  TORCH_CHECK(step_value != 0, "step must be nonzero");
  TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
      "upper bound and larger bound inconsistent with step sign");

  // calculate the output size
  double size_arange = std::ceil(static_cast<double>(end.toDouble() - start.toDouble())
                                 / step.toDouble());
  int64_t size_value = static_cast<int64_t>(size_arange);
  at::SmallVector<int64_t, SIZE> outputSize = {size_value};
  result.resize_(outputSize);

  arange_out_npu_nocheck(result, start, end, step);

  return result;
}

at::Tensor& arange_other_out_npu(at::Scalar start, at::Scalar end, at::Tensor& result) {
  at::Scalar step = 1;
  return NPUNativeFunctions::arange_out(start, end, step, result);
}

at::Tensor& NPUNativeFunctions::arange_out(const at::Scalar& end, at::Tensor& result) {
  return arange_other_out_npu(0, end, result);
}

at::Tensor NPUNativeFunctions::_dim_arange(const at::Tensor& self, int64_t dim) {
  c10::optional<at::ScalarType> dtype_opt(at::kInt);
  c10::optional<at::Layout> layout_opt(self.options().layout());
  c10::optional<at::Device> device_opt(self.options().device());
  c10::optional<bool> pin_memory_opt(self.options().pinned_memory());

  at::Tensor result = NPUNativeFunctions::arange(self.size(dim), dtype_opt, layout_opt, device_opt, pin_memory_opt);
  return result;
}

} // namespace native
} // namespace at_npu
