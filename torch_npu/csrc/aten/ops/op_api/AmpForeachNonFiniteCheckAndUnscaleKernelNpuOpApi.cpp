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
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace at_npu {
namespace native {

const int FLOAT_STATUS_OP_DIMS_SIZE = 8;

void NPUNativeOpApiFunctions::_amp_foreach_non_finite_check_and_unscale_(at::TensorList scaled_grads,
                                                                         at::Tensor& found_inf,
                                                                         const at::Tensor& inv_scale) {
  TORCH_NPU_WARN_ONCE("Non finite check and unscale on NPU device!");

  TORCH_CHECK(at_npu::key::isDeviceTensor(inv_scale), "inv_scale must be NPU-Tensor");
  TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor");
  TORCH_CHECK(inv_scale.scalar_type() == at::ScalarType::Float, "inv_scale must be a float tensor");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor");

  if (scaled_grads.empty()) {
    return;
  }

  bool is_finite = true;
  if (c10_npu::IsSupportInfNan()) {
    for (const auto& scaled_grad : scaled_grads) {
      auto res = NPUNativeOpApiFunctions::sum(scaled_grad, at::ScalarType::Float);
      float cpu_sum = res.item().toFloat();
      if (!std::isfinite(cpu_sum)) {
        is_finite = false;
        break;
      }
    }
  } else {
    is_finite = !NPUNativeFunctions::_amp_foreach_non_finite_check_(scaled_grads);
  }

  if (!is_finite) {
    NPUNativeOpApiFunctions::ones_out(1, found_inf);
  }

  auto expected_device = scaled_grads[0].device();
  auto expected_dtype = scaled_grads[0].dtype();
  for (auto& t : scaled_grads) {
    TORCH_CHECK(at_npu::key::isDeviceTensor(t), "one of scaled_grads was not a NPU tensor.");
    TORCH_CHECK(t.device() == expected_device, "scaled_grads must be on the same device.");
    TORCH_CHECK(t.dtype() == expected_dtype, "scaled_grads must have the same dtype.");
    TORCH_CHECK(t.layout() == at::kStrided, "one of scaled_grads was not a strided tensor.");

    NPUNativeOpApiFunctions::mul_out(t, inv_scale, const_cast<at::Tensor&>(t));
  }
}

}  // namespace native
}  // namespace at_npu
