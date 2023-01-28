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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

c10::SmallVector<int64_t, SIZE> renorm_npu_output_size(
    const at::Tensor& self,
    int64_t dim) {
  c10::SmallVector<int64_t, SIZE> outSize;
  for(int64_t i=0; i < self.dim(); i++) {
    if(i != dim) {
      outSize.emplace_back(1);
    } else {
      outSize.emplace_back(self.sizes()[i]);
    }
  }
  return outSize;
}

at::Tensor& renorm_compute(
    at::Tensor& result,
    const at::Tensor& self,
    at::Scalar p,
    int64_t dim,
    at::Scalar maxnorm) {
  float p_value = CalcuOpUtil::GetScalarFloatValue(p);
  float maxnorm_value = CalcuOpUtil::GetScalarFloatValue(maxnorm);

  OpCommand cmd;
  cmd.Name("Renorm")
    .Input(self)
    .Output(result)
    .Attr("p", p_value)
    .Attr("maxnorm", maxnorm_value)
    .Attr("dim", dim)
    .Run();
  return result;
}

at::Tensor& renorm_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::Scalar p,
    int64_t dim,
    at::Scalar maxnorm) {
  auto ori_type = self.scalar_type();
  if(ori_type != c10::ScalarType::Half && ori_type != c10::ScalarType::Float) {
    AT_ERROR("Renorm only support float16 or float32 type.");
  }
  if(result.scalar_type() != ori_type) {
    AT_ERROR("result's type must be equal to input's.");
  }
  dim = CalcuOpUtil::MakeWrapDim(dim, self.dim());
  auto outputSize = renorm_npu_output_size(self, dim);
  at::Tensor result_bak = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      self.options().dtype(at::kFloat),
      CalcuOpUtil::GetTensorNpuFormat(self));
  if(ori_type == c10::ScalarType::Half) {
    at::Tensor self_no_name = self.rename(c10::nullopt);
    at::Tensor result_no_name = result.rename(c10::nullopt);
    self_no_name = NPUNativeFunctions::npu_dtype_cast(self_no_name, c10::ScalarType::Float);
    result_no_name = NPUNativeFunctions::npu_dtype_cast(result_no_name, c10::ScalarType::Float);
    renorm_compute(
        result_bak,
        self_no_name,
        p,
        dim,
        maxnorm);

    at::Tensor result_broadcast = NPUNativeFunctions::npu_broadcast(result_bak, self.sizes());
    at::mul_out(result_no_name, result_broadcast, self_no_name);
    NPUNativeFunctions::npu_dtype_cast_(result, result_no_name);
  } else {
    renorm_compute(
        result_bak,
        self,
        p,
        dim,
        maxnorm);

    at::Tensor result_broadcast = NPUNativeFunctions::npu_broadcast(result_bak, self.sizes());
    at::mul_out(result, result_broadcast, self);
  }
  return result;
}

at::Tensor& NPUNativeFunctions::renorm_out(
    const at::Tensor& self,
    const at::Scalar& p,
    int64_t dim,
    const at::Scalar& maxnorm,
    at::Tensor& result) {
  auto outputSize = input_same_output_size(self);
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);

  c10::SmallVector<at::Tensor, N> inputs = {self};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor checkresult = renorm_out_nocheck(contiguousSelf, contiguousSelf, p, dim, maxnorm);
    NpuUtils::format_fresh_view(result, checkresult);
  } else {
    renorm_out_nocheck(result, self, p, dim, maxnorm);
  }

  return result;
}

at::Tensor NPUNativeFunctions::renorm(const at::Tensor& self, const at::Scalar& p, int64_t dim, const at::Scalar& maxnorm) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensor(self);

  return renorm_out_nocheck(result, self, p, dim, maxnorm);
}

at::Tensor& NPUNativeFunctions::renorm_(at::Tensor& self, const at::Scalar& p, int64_t dim, const at::Scalar& maxnorm) {

    return renorm_out(self, p, dim, maxnorm, self);
}

} // namespace native
} // namespace at_npu