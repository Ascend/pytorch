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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<int64_t, SIZE> renorm_npu_output_size(
    const Tensor& self,
    int64_t dim) {
  SmallVector<int64_t, SIZE> outSize;
  for(int64_t i=0; i < self.dim(); i++) {
    if(i != dim) {
      outSize.emplace_back(1);
    } else {
      outSize.emplace_back(self.sizes()[i]);
    }
  }
  return outSize;
}

Tensor& renorm_compute(   
    Tensor& result, 
    const Tensor& self,
    Scalar p, 
    int64_t dim, 
    Scalar maxnorm) {
  float p_value = CalcuOpUtil::get_scalar_float_value(p);
  float maxnorm_value = CalcuOpUtil::get_scalar_float_value(maxnorm);
  OpCommand cmd;
  cmd.Name("Renorm")
      .Input(self)
      .Output(result)
      .Attr("p", p_value)
      .Attr("dim", dim)
      .Attr("maxnorm", maxnorm_value)
      .Run();
  return result;
}

Tensor& renorm_out_npu_nocheck(   
    Tensor& result, 
    const Tensor& self,
    Scalar p, 
    int64_t dim, 
    Scalar maxnorm) {
  auto ori_type = self.scalar_type();
  if(ori_type != c10::ScalarType::Half && ori_type != c10::ScalarType::Float) {
    AT_ERROR("Renorm only support float16 or float32 type.");
  }
  if(result.scalar_type() != ori_type) {
    AT_ERROR("result's type must be equal to input's.");
  }
  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  auto outputSize = renorm_npu_output_size(self, dim);
  Tensor result_bak = OpPreparation::ApplyTensor(
      outputSize,
      self.options().dtype(at::kFloat), 
      self);
  if(ori_type == c10::ScalarType::Half) {
    Tensor self_no_name = self.rename(nullopt);
    Tensor result_no_name = result.rename(nullopt);
    self_no_name = self_no_name.npu_dtype_cast(c10::ScalarType::Float);
    result_no_name = result_no_name.npu_dtype_cast(c10::ScalarType::Float);
    renorm_compute(   
        result_bak, 
        self_no_name,
        p, 
        dim, 
        maxnorm);
    // broadcast and mul
    Tensor result_broadcast = at::npu_broadcast(result_bak, self.sizes());
    at::mul_out(result_no_name, result_broadcast, self_no_name);
    result.npu_dtype_cast_(result_no_name);
  } else {
    renorm_compute(   
        result_bak, 
        self,
        p, 
        dim, 
        maxnorm);
    // broadcast and mul
    Tensor result_broadcast = at::npu_broadcast(result_bak, self.sizes());
    at::mul_out(result, result_broadcast, self);
  }
  return result;
}

Tensor& renorm_out_npu(   
    Tensor& result, 
    const Tensor& self,
    Scalar p, 
    int64_t dim, 
    Scalar maxnorm) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    Tensor contiguousResult = NpuUtils::format_contiguous(result);
    renorm_out_npu_nocheck(contiguousResult, self, p, dim, maxnorm);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    renorm_out_npu_nocheck(result, self, p, dim, maxnorm);
  }
    return result;
}

Tensor renorm_npu(const Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
  Tensor result = OpPreparation::ApplyTensor(self);
  renorm_out_npu_nocheck(result, self, p, dim, maxnorm);
  return result;
}

Tensor& renorm_npu_(Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
  renorm_out_npu(self, self, p, dim, maxnorm);
  return self;
}

} // namespace na tive
} // namespace at