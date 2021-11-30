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

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/Optional.h>
#include "ATen/native/npu/utils/OpAdapter.h"
#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace at {
namespace native {

// Take a Device that may not have device_index set (i.e., having it as -1
// representing the current device) and return the corresponding Device
// according to the actual device at the time of this function call.  No-op
// if the device_index is set.
static inline Device ensure_has_index(Device device) {
  if (device.is_cpu() || device.has_index()) {
    return device;
  }
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl(device.type());
  return impl->getDevice();
}

static inline Tensor to_impl_npu(
    const Tensor& self,
    const TensorOptions& options,
    bool non_blocking,
    bool copy) {
  auto memory_format = options.memory_format_opt().value_or(
      MemoryFormat::Contiguous); // Here cpu's default value is Preserve

  if (self.dtype() == options.dtype() && self.layout() == options.layout() &&
      self.device() == options.device() && !copy &&
      (memory_format == MemoryFormat::Preserve ||
       self.suggest_memory_format() == memory_format)) {
    return self;
  }

  if (memory_format == MemoryFormat::Preserve) {
    if (self.is_non_overlapping_and_dense()) {
      // Copy all strides
      auto r = at::empty_strided(
          self.sizes(), self.strides(), options.memory_format(c10::nullopt));
      r.copy_(self, non_blocking);
      return r;
    } else {
      memory_format = self.suggest_memory_format();
    }
  }
  // See Note [Explicit nullopt MemoryFormat argument]
  auto r = at::empty(
      self.sizes(), options.memory_format(memory_format), c10::nullopt);
  r.copy_(self, non_blocking);
  return r;
}

Tensor to_npu(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "NPU not support specify memory_format.");
  TensorOptions options_;
  options_ = options_.dtype(dtype)
                  .layout(layout)
                  .device(device);
  TORCH_CHECK(
      !(options_.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");
  auto options =
      options_.merge_in(TensorOptions().memory_format(optional_memory_format));

  TORCH_CHECK(
      options.requires_grad_opt() == c10::nullopt,
      "to(options) expects unset requires_grad flag, but got "
      "options.requires_grad set as ",
      options.requires_grad());

  TORCH_CHECK(
      !options.has_layout() || self.layout() == options.layout(),
      "to(options) doesn't support converting to a different layout, "
      "but got self.layout being ",
      self.layout(),
      " and options.layout set as ",
      options.layout());

  if (options.has_device()) {
    options = options.device(ensure_has_index(options.device()));
  }
  auto specified_options = self.options().merge_in(options);
  return to_impl_npu(self, specified_options, non_blocking, copy);
}

Tensor to_device_npu(
    const Tensor& self,
    Device device,
    ScalarType dtype,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  device = ensure_has_index(device);
  return to_impl_npu(
      self,
      self.options().device(device).dtype(dtype).memory_format(
          optional_memory_format),
      non_blocking,
      copy);
}

Tensor to_dtype_npu(
    const Tensor& self,
    ScalarType dtype,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  if (self.dtype() == dtype) {
    return self;
  }
  if (ScalarType::Double == dtype) {
    TORCH_WARN_ONCE("Unsupport Double dtype now, replace with float.");
  }
  dtype = (ScalarType::Double == dtype) ? ScalarType::Float : dtype;
  return to_impl_npu(self, 
      self.options().dtype(dtype).memory_format(optional_memory_format), non_blocking, copy);
}

Tensor to_other_npu(
    const Tensor& self,
    const Tensor& other,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto options = other.options();
  return to_impl_npu(
      self, options.memory_format(optional_memory_format), non_blocking, copy);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("to.dtype_layout", TORCH_FN(to_npu));
  m.impl("to.other", TORCH_FN(to_other_npu));
  m.impl("to.dtype", TORCH_FN(to_dtype_npu));
  m.impl("to.device", TORCH_FN(to_device_npu));
}
} // namespace native
} // namespace at
