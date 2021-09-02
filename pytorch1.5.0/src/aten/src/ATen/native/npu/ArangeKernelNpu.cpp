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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

// bool inputs are considered integral
static inline bool allIntegral(
    std::initializer_list<std::reference_wrapper<Scalar>> l) {
  for (Scalar& s : l) {
    if (!s.isIntegral(true)) {
      return false;
    }
  }
  return true;
}


Tensor& arange_out_npu_nocheck(
    Tensor& result,
    Scalar start,
    Scalar end,
    Scalar step) {
  OpCommand cmd;
  cmd.Name("Range")
     .Input(start, result.scalar_type())  //start
     .Input(end, result.scalar_type())  //limit
     .Input(step, result.scalar_type())  //delta
     .Output(result)
     .Run();

  return result;
}

Tensor arange_npu(Scalar end, const TensorOptions& options) {
  return arange_npu(/*start=*/0, end, options);
}

Tensor arange_npu(Scalar start, Scalar end, const TensorOptions& options) {
  return arange_npu(start, end, /*step=*/1, options);
}

Tensor arange_npu(
    Scalar start,
    Scalar end,
    Scalar step,
    const TensorOptions& options) {
  float start_value = CalcuOpUtil::get_scalar_float_value(start);
  float end_value = CalcuOpUtil::get_scalar_float_value(end);
  float step_value = CalcuOpUtil::get_scalar_float_value(step);

  //Check step start end
  TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero");
  TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
      "upper bound and larger bound inconsistent with step sign");

  bool set_to_integral_dtype =
      !options.has_dtype() && allIntegral({start, end, step});

  //check start == end
  Tensor result_check = set_to_integral_dtype
      ? at::empty_with_format(
            {0}, options.dtype(at::ScalarType::Int), ACL_FORMAT_ND)
      : at::empty_with_format({0}, options, ACL_FORMAT_ND);
  if (start_value == end_value) {
    return result_check;
  }

  // calculate the output size
  double size_arange = std::ceil(static_cast<double>(end.toDouble() - start.toDouble())
                                 / step.toDouble());
  int64_t size_value = static_cast<int64_t>(size_arange);
  SmallVector<int64_t, SIZE> outputSize = {size_value};

  Tensor result = set_to_integral_dtype
      ? at::empty_with_format(
            outputSize, options.dtype(at::ScalarType::Int), ACL_FORMAT_ND)
      : at::empty_with_format(outputSize, options, ACL_FORMAT_ND);

  if(options.dtype() == at::kHalf) {
    result = result.to(at::kFloat);
  }

  arange_out_npu_nocheck(result, start, end, step);

  if(options.dtype() == at::kHalf) {
    result = result.to(at::kHalf);
  }

  return result;
}

Tensor& arange_out_npu(Tensor& result, Scalar end) {
  return arange_out_npu(result, /*start=*/0, end);
}

Tensor& arange_out_npu(Tensor& result, Scalar start, Scalar end) {
  return arange_out_npu(result, start, end, /*step=*/1);
}

Tensor& arange_out_npu(
    Tensor& result,
    Scalar start,
    Scalar end,
    Scalar step) {
  float start_value = CalcuOpUtil::get_scalar_float_value(start);
  float end_value = CalcuOpUtil::get_scalar_float_value(end);
  float step_value = CalcuOpUtil::get_scalar_float_value(step);

  //Check step start end
  TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero");
  TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
      "upper bound and larger bound inconsistent with step sign");

  // calculate the output size
  double size_arange = std::ceil(static_cast<double>(end.toDouble() - start.toDouble())
                                 / step.toDouble());
  int64_t size_value = static_cast<int64_t>(size_arange);
  SmallVector<int64_t, SIZE> outputSize = {size_value};

  OpPreparation::CheckOut(
      { },
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      outputSize);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({},{result})
   .Func([&start, &end, &step](Tensor& result){arange_out_npu_nocheck(result, start, end, step);})
   .Call(result);

  //arange_out_npu_nocheck(result, start, end, step);

  return result;
}

Tensor _dim_arange_npu(const Tensor& self, int64_t dim) {
  Tensor result = at::arange(self.size(dim), self.options().dtype(at::kInt));
  return result;
}

} // namespace native
} // namespace at
