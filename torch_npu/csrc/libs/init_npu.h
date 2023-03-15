// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"


namespace torch_npu {

// device init related funcs
void init_npu(const c10::DeviceIndex device_index=0);
void init_npu(const std::string& device_str);
void init_npu(const at::Device& device);

// device finalize related funcs
void finalize_npu();

bool is_npu_device(const at::Device& device);
c10::DeviceIndex current_device();

} // namespace torch_npu
