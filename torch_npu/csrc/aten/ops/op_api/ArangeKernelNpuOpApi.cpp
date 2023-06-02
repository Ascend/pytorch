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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

// bool inputs are considered integral
static inline bool allIntegral(std::initializer_list<std::reference_wrapper<at::Scalar>> l) {
  for (at::Scalar& s : l) {
    if (!s.isIntegral(true)) {
      return false;
    }
  }
  return true;
}

static at::Tensor& ArangeOutOpApi(at::Scalar start, at::Scalar end, at::Scalar step, at::Tensor& result) {
  EXEC_NPU_CMD(aclnnArange, start, end, step, result);
  return result;
}

static bool IsTensorFloat(at::Tensor& result) {
  if (isFloatingType(result.scalar_type())) {
    return true;
  } else {
    return false;
  }
}

static int64_t GetResultSize(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
                             at::Tensor& result) {
  double size_arange = 0;
  // calculate the output size
  if (IsTensorFloat(result)) {
    if (step.toDouble() != 0) {
      size_arange = std::ceil(static_cast<double>(end.toDouble() - start.toDouble()) / step.toDouble());
    }
  } else {
    if (step.toLong() != 0) {
      size_arange = std::ceil(static_cast<double>(end.toLong() - start.toLong()) / step.toLong());
    }
  }
  return static_cast<int64_t>(size_arange);
}

at::Tensor NPUNativeOpApiFunctions::arange(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
                                           c10::optional<at::ScalarType> dtype_opt,
                                           c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
                                           c10::optional<bool> pin_memory_opt) {
  DO_COMPATIBILITY(aclnnArange,
                   NPUNativeFunctions::arange(start, end, step, dtype_opt, layout_opt, device_opt, pin_memory_opt));
  c10::TensorOptions option =
      c10::TensorOptions().dtype(dtype_opt).device(device_opt).layout(layout_opt).pinned_memory(pin_memory_opt);

  at::Scalar start_opt = start;
  at::Scalar end_opt = end;
  at::Scalar step_opt = step;
  bool set_to_integral_dtype = !option.has_dtype() && allIntegral({start_opt, end_opt, step_opt});

  // check start == end
  if (set_to_integral_dtype) {
    option = option.dtype(at::ScalarType::Long);
  }
  at::Tensor result_check = OpPreparation::ApplyTensorWithFormat({0}, option, ACL_FORMAT_ND);

  int64_t size_value = GetResultSize(start, end, step, result_check);
  at::SmallVector<int64_t, SIZE> outputSize = {size_value};
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(outputSize, option, ACL_FORMAT_ND);

  if (option.dtype() == at::kHalf) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kFloat);
  }

  ArangeOutOpApi(start, end, step, result);

  if (option.dtype() == at::kHalf) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kHalf);
  }

  return result;
}

at::Tensor NPUNativeOpApiFunctions::arange(const at::Scalar& start, const at::Scalar& end,
                                           c10::optional<at::ScalarType> dtype_opt,
                                           c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
                                           c10::optional<bool> pin_memory_opt) {
  DO_COMPATIBILITY(aclnnArange,
                   NPUNativeFunctions::arange(start, end, dtype_opt, layout_opt, device_opt, pin_memory_opt));
  return NPUNativeOpApiFunctions::arange(start, end, 1, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor NPUNativeOpApiFunctions::arange(const at::Scalar& end, c10::optional<at::ScalarType> dtype_opt,
                                           c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
                                           c10::optional<bool> pin_memory_opt) {
  DO_COMPATIBILITY(aclnnArange, NPUNativeFunctions::arange(end, dtype_opt, layout_opt, device_opt, pin_memory_opt));
  return NPUNativeOpApiFunctions::arange(0, end, dtype_opt, layout_opt, device_opt, pin_memory_opt);  // start = 0
}

at::Tensor& NPUNativeOpApiFunctions::arange_out(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
                                                at::Tensor& result) {
  DO_COMPATIBILITY(aclnnArange, NPUNativeFunctions::arange_out(start, end, step, result));

  int64_t size_value = GetResultSize(start, end, step, result);
  at::SmallVector<int64_t, SIZE> outputSize = {size_value};
  result.resize_(outputSize);

  ArangeOutOpApi(start, end, step, result);

  return result;
}

at::Tensor& ArangeStartEndOut(at::Scalar start, at::Scalar end, at::Tensor& result) {
  at::Scalar step = 1;
  return NPUNativeOpApiFunctions::arange_out(start, end, step, result);
}

at::Tensor& NPUNativeOpApiFunctions::arange_out(const at::Scalar& end, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnArange, NPUNativeFunctions::arange_out(end, result));
  return ArangeStartEndOut(0, end, result);
}

}  // namespace native
}  // namespace at_npu
