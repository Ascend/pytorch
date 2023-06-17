// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::range_out(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
                                               at::Tensor& result) {
  DO_COMPATIBILITY(aclnnRange, NPUNativeFunctions::range_out(start, end, step, result));

  float start_value = CalcuOpUtil::GetScalarFloatValue(start);
  float end_value = CalcuOpUtil::GetScalarFloatValue(end);
  float step_value = CalcuOpUtil::GetScalarFloatValue(step);

  TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
      "upper bound and larger bound inconsistent with step sign");

  auto outputSize = range_npu_output_size(start_value, end_value, step_value);
  OpPreparation::CheckOut({ }, result, result.scalar_type(), outputSize);

  EXEC_NPU_CMD(aclnnRange, start, end, step, result);
  return result;
}

}  // namespace native
}  // namespace at_npu