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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::upsample_bilinear2d_out(
    const at::Tensor& self_ex,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& result){
  at::Tensor self = self_ex;
  TORCH_CHECK(self.scalar_type() != at::ScalarType::Double,
      "upsample_binlinear_2d not support torch.fp64 dtypes");
  auto outputSize = upsample_bilinear2d_npu_output_size(
      self, output_size, align_corners, scales_h, scales_w);

  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);

  double scales_h_attr = scales_h.has_value() ? scales_h.value() : 1;
  double scales_w_attr = scales_w.has_value() ? scales_w.value() : 1;
  EXEC_NPU_CMD(aclnnUpsampleBilinear2D, self, output_size, align_corners,
               scales_h_attr, scales_w_attr, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::upsample_bilinear2d(
    const at::Tensor& self_ex,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  at::Tensor self = self_ex;
  TORCH_CHECK(self.scalar_type() != at::ScalarType::Double,
      "upsample_binlinear_2d not support torch.fp64 dtypes");
  
  double scales_h_attr = scales_h.has_value() ? scales_h.value() : 1;
  double scales_w_attr = scales_w.has_value() ? scales_w.value() : 1;
  auto outputSize = upsample_bilinear2d_npu_output_size(
      self, output_size, align_corners, scales_h, scales_w);
  at::Tensor result = OpPreparation::ApplyTensor(outputSize, self.options(), self);

  EXEC_NPU_CMD(aclnnUpsampleBilinear2D, self, output_size, align_corners,
               scales_h_attr, scales_w_attr, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::upsample_bilinear2d(
    const at::Tensor& self_ex,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  at::Tensor self = self_ex;
  auto osize = CalcuOpUtil::ComputeOutputSize(self_ex.sizes(), output_size, scale_factors);
  auto scales_h = CalcuOpUtil::GetScaleValue(scale_factors, 0);
  auto scales_w = CalcuOpUtil::GetScaleValue(scale_factors, 1);

  TORCH_CHECK(self.scalar_type() != at::ScalarType::Double,
      "upsample_binlinear_2d not support torch.fp64 dtypes");
  
  auto outputSize = upsample_bilinear2d_npu_output_size(
      self, osize, align_corners, scales_h, scales_w);
  at::Tensor result = OpPreparation::ApplyTensor(outputSize, self.options(), self);
  
  auto outputsize = at::IntArrayRef(osize);
  double scales_h_attr = scales_h.has_value() ? scales_h.value() : 1;
  double scales_w_attr = scales_w.has_value() ? scales_w.value() : 1;
  EXEC_NPU_CMD(aclnnUpsampleBilinear2D, self, outputsize, align_corners,
               scales_h_attr, scales_w_attr, result);
  return result;
}

} // namespace native
} // namespace at_npu
