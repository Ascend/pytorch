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

#pragma once
#ifndef __SCRIPTS_CODEGEN_TEMPLATES_DISPATCHKEYNATIVEFUNCTOPNS__
#define __SCRIPTS_CODEGEN_TEMPLATES_DISPATCHKEYNATIVEFUNCTOPNS__


#include <ATen/Tensor.h>
#include <ATen/ATen.h>
namespace at_npu {
namespace key {
static constexpr c10::DeviceType NativeDeviceType = c10::DeviceType::NPU;
static constexpr c10::DispatchKey NativeDispatchKey = c10::DispatchKey::NPU;
static constexpr c10::DispatchKey NativeAutogradDispatchKey = c10::DispatchKey::AutogradNPU;
static constexpr c10::Backend NativeBackend = c10::Backend::NPU;

static bool isDeviceTensor(const at::Tensor &tensor) {
  return tensor.is_npu();
}
} // namespace key
} // namespace at_npu

// ${generated_comment}
namespace ${cpp_namespace} {

struct ${class_name} {

${dispatch_declarations}

};
}  // namespace ${cpp_namespace}

#endif