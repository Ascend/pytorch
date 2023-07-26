// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {
at::Tensor& avg_pool2d_out_npu_nocheck_opapi(at::Tensor& result,
                                             const at::Tensor& self,
                                             at::IntArrayRef kernel,
                                             at::IntArrayRef stride,
                                             at::IntArrayRef padding,
                                             bool ceil_mode,
                                             bool count_include_pad,
                                             c10::optional<int64_t> divisor_override)
{
  int64_t s_divisor_override = 0;
  if (divisor_override.has_value()) {
    s_divisor_override = divisor_override.value();
    TORCH_CHECK(s_divisor_override != 0, "divisor must be not zero.");
  }

  const int8_t cube_math_type = 1;
  EXEC_NPU_CMD(aclnnAvgPool2d, self, kernel, stride, padding, ceil_mode, count_include_pad, s_divisor_override,
      cube_math_type, result);

  return result;
}

c10::SmallVector<int64_t, SIZE> calc_output_size_with_generalized_attrs(const at::Tensor& self,
                                                                        at::IntArrayRef kernel_size,
                                                                        at::IntArrayRef stride,
                                                                        at::IntArrayRef padding,
                                                                        bool ceil_mode,
                                                                        bool count_include_pad,
                                                                        c10::optional<int64_t> divisor_override)
{
  // generalize kernels, strides and paddings to 2D-shape
  const int64_t k_h = kernel_size[0];
  const int64_t k_w = kernel_size.size() == 1 ? k_h : kernel_size[1];
  c10::SmallVector<int64_t, SIZE> kernel_sizes = {k_h, k_w};
  at::IntArrayRef kernels = at::IntArrayRef(kernel_sizes);

  const int64_t s_h = stride.empty() ? k_h : stride[0];
  const int64_t s_w = stride.empty() ? k_w : stride.size() == 1 ? s_h : stride[1];
  c10::SmallVector<int64_t, SIZE> stride_sizes = {s_h, s_w};
  at::IntArrayRef strides = at::IntArrayRef(stride_sizes);

  const int64_t pad_h = padding[0];
  const int64_t pad_w = padding.size() == 1 ? pad_h : padding[1];
  c10::SmallVector<int64_t, SIZE> padding_sizes = {pad_h, pad_w};
  at::IntArrayRef paddings = at::IntArrayRef(padding_sizes);

  auto output_size = avg_pool2d_npu_output_size(self, kernels, strides, paddings, ceil_mode, count_include_pad,
      divisor_override);

  return output_size;
}

at::Tensor& NPUNativeOpApiFunctions::avg_pool2d_out(const at::Tensor& self,
                                                    at::IntArrayRef kernel_size,
                                                    at::IntArrayRef stride,
                                                    at::IntArrayRef padding,
                                                    bool ceil_mode,
                                                    bool count_include_pad,
                                                    c10::optional<int64_t> divisor_override,
                                                    at::Tensor& result)
{
  c10::SmallVector<int64_t, SIZE> output_size = calc_output_size_with_generalized_attrs(self, kernel_size, stride,
      padding, ceil_mode, count_include_pad, divisor_override);
  OpPreparation::CheckOut({self}, result, self, output_size);

  DO_COMPATIBILITY(aclnnAvgPool2d,
      NPUNativeFunctions::avg_pool2d_out(self, kernel_size, stride, padding, ceil_mode, count_include_pad,
          divisor_override, result));

  avg_pool2d_out_npu_nocheck_opapi(result, self, kernel_size, stride, padding, ceil_mode, count_include_pad,
      divisor_override);

  return result;
}

at::Tensor NPUNativeOpApiFunctions::avg_pool2d(const at::Tensor& self,
                                               at::IntArrayRef kernel_size,
                                               at::IntArrayRef stride,
                                               at::IntArrayRef padding,
                                               bool ceil_mode,
                                               bool count_include_pad,
                                               c10::optional<int64_t> divisor_override)
{
  c10::SmallVector<int64_t, SIZE> output_size = calc_output_size_with_generalized_attrs(self, kernel_size, stride,
      padding, ceil_mode, count_include_pad, divisor_override);

  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, output_size);

  DO_COMPATIBILITY(aclnnAvgPool2d,
      NPUNativeFunctions::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad,
          divisor_override));

  avg_pool2d_out_npu_nocheck_opapi(result, self, kernel_size, stride, padding, ceil_mode, count_include_pad,
      divisor_override);

  return result;
}

} // namespace native
} // namespace at_npu
