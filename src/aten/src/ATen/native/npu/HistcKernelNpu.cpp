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

#define FLT_EPSILON     1.19209290E-07F

bool is_zero(float x)
{
	if(x > -FLT_EPSILON && x < FLT_EPSILON){
		return true;
	}
	else{
		return false;
	}
}

Tensor& histc_out_npu(
    Tensor& result,
    const Tensor& self,
    int64_t bins,
    Scalar min,
    Scalar max) {
  OpCommand cmd;
  float max_value = CalcuOpUtil::get_scalar_float_value(max);
  float min_value = CalcuOpUtil::get_scalar_float_value(min);

  if(max_value == min_value && is_zero(max_value)){
    // Execute reduce_max_d and reduce_min_d to get the min and max value
    Tensor res_max = at::max(self);
    Tensor res_min = at::min(self);
    
    max_value = CalcuOpUtil::get_scalar_float_value(res_max.item());
    min_value = CalcuOpUtil::get_scalar_float_value(res_min.item());
  }
  cmd.Name("HistogramD")
      .Input(self)
      .Attr("bins", bins)
      .Attr("min", min_value)
      .Attr("max", max_value)
      .Output(result)
      .Run();
  
  return result;
}

Tensor histc_npu(const Tensor& self, int64_t bins, Scalar min, Scalar max) {
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      {bins}, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  histc_out_npu(result, self, bins, min, max);

  return result;
}

} // namespace native
} // namespace at
