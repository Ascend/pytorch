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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::upsample_linear1d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnUpsampleLinear1d, NPUNativeFunctions::upsample_linear1d_out(self, output_size,
                                                                                    align_corners,
                                                                                    scales,
                                                                                    result));
  auto outsize = upsample_linear1d_npu_output_size(self, output_size, align_corners, scales);

  OpPreparation::CheckOut({self}, result, self, outsize);

  double scales_h_attr = scales.has_value() ? scales.value() : -1;
  EXEC_NPU_CMD(aclnnUpsampleLinear1d, self, output_size, align_corners, scales_h_attr, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::upsample_linear1d(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales) {
  DO_COMPATIBILITY(aclnnUpsampleLinear1d, NPUNativeFunctions::upsample_linear1d(self, output_size,
                                                                                align_corners,
                                                                                scales));
  auto outsize = upsample_linear1d_npu_output_size(self, output_size, align_corners, scales);

  double scales_h_attr = scales.has_value() ? scales.value() : -1;
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(outsize, self.options());

  EXEC_NPU_CMD(aclnnUpsampleLinear1d, self, output_size, align_corners, scales_h_attr, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::upsample_linear1d(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  DO_COMPATIBILITY(aclnnUpsampleLinear1d, NPUNativeFunctions::upsample_linear1d(self, output_size,
                                                                                align_corners,
                                                                                scale_factors));
  auto osize = CalcuOpUtil::ComputeOutputSize(self.sizes(), output_size, scale_factors);
  auto scales = CalcuOpUtil::GetScaleValue(scale_factors, 0);
  auto outsize = at::IntArrayRef(osize);
  auto out_size = upsample_linear1d_npu_output_size(self, outsize, align_corners, scales);

  double scales_h_attr = scales.has_value() ? scales.value() : -1;
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(out_size, self.options());

  EXEC_NPU_CMD(aclnnUpsampleLinear1d, self, outsize, align_corners, scales_h_attr, result);
  return result;
}

} // namespace native
} // namespace at_npu