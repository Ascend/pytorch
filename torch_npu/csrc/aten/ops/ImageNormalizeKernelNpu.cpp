// Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

at::Tensor &image_normalize_out(
    const at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> mean,
    c10::optional<c10::ArrayRef<double>> variance,
    int64_t dtype,
    at::Tensor &result)
{
  TORCH_CHECK(mean.has_value() && variance.has_value(),
      "[mean] and [variance] should be mandatory");
  TORCH_CHECK(dtype == 0 || dtype == 1,
      "output data type should be float16 or float32");
  std::vector<int64_t> para_shape = {1, 3, 1, 1};
  OpCommand cmd;
  cmd.Name("NormalizeV2")
      .Input(self)
      .Input(mean.value(), para_shape, at::kFloat)
      .Input(variance.value(), para_shape, at::kFloat)
      .Output(result)
      .Attr("dtype", dtype)
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::image_normalize(
    const at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> mean,
    c10::optional<c10::ArrayRef<double>> variance,
    int64_t dtype)
{
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  at::Tensor result;
  if (dtype == 0) {
    result = OpPreparation::ApplyTensorWithFormat(
        outputSize,
        self.options().dtype(at::kFloat),
        CalcuOpUtil::GetTensorNpuFormat(self));
  } else {
    result = OpPreparation::ApplyTensorWithFormat(
        outputSize,
        self.options().dtype(at::kHalf),
        CalcuOpUtil::GetTensorNpuFormat(self));
  }

  // calculate the output result of the NPU
  image_normalize_out(self, mean, variance, dtype, result);

  return result;
}

at::Tensor& NPUNativeFunctions::image_normalize_(
    at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> mean,
    c10::optional<c10::ArrayRef<double>> variance,
    int64_t dtype)
{
  TORCH_CHECK(self.scalar_type() == at::kFloat || self.scalar_type() == at::kHalf,
      "inplace image normalize can only support float16 or float32");
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = image_normalize_out(contiguousSelf, mean, variance, dtype, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    image_normalize_out(self, mean, variance, dtype, self);
  }

  return self;
}

} // namespace native
} // namespace at_npu