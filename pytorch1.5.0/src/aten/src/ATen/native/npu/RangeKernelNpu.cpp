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

Tensor range_npu(Scalar start, Scalar end, const TensorOptions& options) {
  return range_npu(start, end, 1, options);
}

Tensor range_npu(
    Scalar start,
    Scalar end,
    Scalar step,
    const TensorOptions& options) {
  float start_value = CalcuOpUtil::get_scalar_float_value(start);
  float end_value = CalcuOpUtil::get_scalar_float_value(end);
  float step_value = CalcuOpUtil::get_scalar_float_value(step);

  // Check step start end
  TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero");
  TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
      "upper bound and larger bound inconsistent with step sign");

  // calculate the output size
  auto outputSize = range_npu_output_size(start_value, end_value, step_value);

  Tensor result = at::empty_with_format(outputSize, options, ACL_FORMAT_NCHW);

  return range_out_npu(result, start, end, step);
}

Tensor& range_out_npu(
    Tensor& result,
    Scalar start,
    Scalar end,
    Scalar step) {
  // generate x assistant tensor
  int value = result.size(0);
  vector<int> tmp_vector = {};
  for (int i = 0; i < value; i++) {
    tmp_vector.emplace_back(i);
  }
  Tensor assistDimInfo = from_blob(tmp_vector.data(), {value}, at::kInt);
  Tensor assistTensor = CalcuOpUtil::copy_tensor_host_to_device(assistDimInfo);
  assistTensor = assistTensor.npu_dtype_cast(result.scalar_type());

  OpCommand cmd;
  cmd.Name("RangeD")
     .Input(assistTensor)
     .Output(result)
     .Attr("start", start)
     .Attr("limit", end)
     .Attr("delta", step)
     .Run();

  return result;
}

} // namespace native
} // namespace at
