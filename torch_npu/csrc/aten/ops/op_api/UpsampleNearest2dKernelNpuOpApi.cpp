// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {

at::SmallVector<int64_t, SIZE> upsample_nearest2d_output_size_npu(
    const at::Tensor& input,
    at::IntArrayRef output_size){
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t H = output_size[0];
  int64_t W = output_size[1];
  at::SmallVector<int64_t, SIZE> outputSize = {N, C, H, W};

  return outputSize;
}

at::Tensor& NPUNativeOpApiFunctions::upsample_nearest2d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnUpsampleNearest2d,
                   NPUNativeFunctions::upsample_nearest2d_out(self, output_size, scales_h, scales_w, result));
  at::SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_output_size_npu(self, output_size);
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);
  EXEC_NPU_CMD(aclnnUpsampleNearest2d, self, output_size, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::upsample_nearest2d(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  DO_COMPATIBILITY(aclnnUpsampleNearest2d,
                   NPUNativeFunctions::upsample_nearest2d(self, output_size, scales_h, scales_w));
  at::SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_output_size_npu(self, output_size);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  EXEC_NPU_CMD(aclnnUpsampleNearest2d, self, output_size, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::upsample_nearest2d(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  DO_COMPATIBILITY(aclnnUpsampleNearest2d,
                   NPUNativeFunctions::upsample_nearest2d(input, output_size, scale_factors));
  auto osize = CalcuOpUtil::ComputeOutputSize(input.sizes(), output_size, scale_factors);
  at::SmallVector<int64_t, SIZE> output_size_vec = upsample_nearest2d_output_size_npu(input, osize);
  at::Tensor result = OpPreparation::ApplyTensor(input, output_size_vec);
  auto outputSize = at::IntArrayRef(osize);
  EXEC_NPU_CMD(aclnnUpsampleNearest2d, input, outputSize, result);
  return result;
}

} // namespace native
} // namespace at_npu