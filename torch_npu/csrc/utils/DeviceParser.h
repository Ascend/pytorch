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

#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/tensor/python_tensor.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/utils/Device.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace at_npu {
namespace key {
static at::Device parse_npu_device(PyObject* obj) {
  if (!obj || obj == Py_None) {
    return at::Device(c10::backendToDeviceType(c10::dispatchKeyToBackend(torch::tensors::get_default_dispatch_key())));
  }
  if (THPUtils_checkLong(obj)) {
    const auto device_index = THPUtils_unpackLong(obj);
    TORCH_CHECK(device_index >= 0, "Device index must not be negative", PTA_ERROR(ErrCode::VALUE));
    return at::Device(at_npu::key::NativeDeviceType, device_index);
  }
  if (THPUtils_checkString(obj)) {
    std::string device_str = THPUtils_unpackString(obj);
    if (device_str.find(at_npu::key::npu_device_str) != std::string::npos) {
      device_str = device_str.replace(device_str.find(at_npu::key::npu_device_str),
                                      at_npu::key::npu_device_str.length(),
                                      at_npu::key::default_device_str);
    }
    return at::Device(device_str);
  }

  if (THPDevice_Check(obj)) {
    const auto device = reinterpret_cast<THPDevice*>(obj);
    return device->device;
  }
  const auto device = reinterpret_cast<TNPDevice*>(obj);
  return device->device;
}

static c10::optional<at::Device>  parse_npu_device_optional(PyObject* obj) {
  if (!obj) {
    return c10::nullopt;
  }
  return parse_npu_device(obj);
}

static at::Device  parse_npu_device_with_default(PyObject* obj, const at::Device& default_device) {
  if (!obj) return default_device;
  return parse_npu_device(obj);
}

} // namespace key
} // namespace at_npu
