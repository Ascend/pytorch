// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2022, Facebook CORPORATION.
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
#ifndef __SCRIPTS_CODEGEN_TEMPLATES_DISPATCHKEYNATIVEFUNCTIONS__
#define __SCRIPTS_CODEGEN_TEMPLATES_DISPATCHKEYNATIVEFUNCTIONS__


#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include <c10/core/Device.h>

namespace at_npu {
namespace key {

static bool isDeviceTensor(const at::Tensor &tensor) {
  return tensor.device().type() == c10::DeviceType::PrivateUse1;
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