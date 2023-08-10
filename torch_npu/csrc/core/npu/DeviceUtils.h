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

#pragma once
#include <c10/core/TensorOptions.h>
#include <c10/core/Device.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace torch_npu {
namespace utils {

static bool is_npu(const at::Tensor& tensor) {
  return tensor.is_xla();
}

static bool is_npu(const at::TensorOptions& options) {
  return options.device().type() == c10::DeviceType::XLA;
}

static bool is_npu(const at::Device& device) {
  return device.type() == c10::DeviceType::XLA;
}

inline c10::DeviceType get_npu_device_type() {
  return c10::DeviceType::XLA;
}

} // namespace utils
} // namespace torch_npu
