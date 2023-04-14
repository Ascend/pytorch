// Copyright (c) 2022 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

const int FLOAT_STATUS_OP_DIMS_SIZE = 8;

bool NPUNativeFunctions::_amp_foreach_non_finite_check(at::TensorList scaled_grads) {
    TORCH_WARN_ONCE("Non finite check on NPU device!");

    auto options = at::TensorOptions(at_npu::key::NativeDeviceType).dtype(at::kFloat);
    at::Tensor float_status = at::zeros({FLOAT_STATUS_OP_DIMS_SIZE}, options);
    auto ans = NPUNativeFunctions::npu_get_float_status(float_status);

    auto result = float_status[0].item().to<bool>();

    if(result == true) {
        auto ans_clear = NPUNativeFunctions::npu_clear_float_status(float_status);
    }
    
    return result;
}

void NPUNativeFunctions::_amp_foreach_non_finite_check_and_unscale_(at::TensorList scaled_grads,
                                                                    at::Tensor& found_inf,
                                                                    const at::Tensor& inv_scale) {
    TORCH_WARN_ONCE("Non finite check and unscale on NPU device!");
    TORCH_CHECK(at_npu::key::isDeviceTensor(inv_scale), "inv_scale must be NPU-Tensor");
    TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor");
    TORCH_CHECK(inv_scale.scalar_type() == at::ScalarType::Float, "inv_scale must be a float tensor");

    if (scaled_grads.size() == 0) {
        return;
    }

    if (NPUNativeFunctions::_amp_foreach_non_finite_check(scaled_grads) == 0) {
        auto expected_device = scaled_grads[0].device();
        auto expected_dtype = scaled_grads[0].dtype();
        for (auto t : scaled_grads) {
            TORCH_CHECK(at_npu::key::isDeviceTensor(t), "one of scaled_grads was not a NPU tensor.");
            TORCH_CHECK(t.device() == expected_device, "scaled_grads must be on the same device.");
            TORCH_CHECK(t.dtype() == expected_dtype, "scaled_grads must have the same dtype.");
            TORCH_CHECK(t.layout() == at::kStrided, "one of scaled_grads was not a strided tensor.");

            t.mul_(inv_scale);
        }
    } else {
        found_inf.add_(1);
    }
}
}
}