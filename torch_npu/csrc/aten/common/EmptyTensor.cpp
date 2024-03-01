#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/CPUAllocator.h>
#include <ATen/EmptyTensor.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/THNPUCachingHostAllocator.h"

namespace at_npu {
namespace native {

static c10::Allocator* GetCPUAllocatorMaybePinned(bool pin_memory) {
  if (pin_memory) {
    return getPinnedMemoryAllocator();
  }
  return c10::GetCPUAllocator();
}

at::TensorBase empty_cpu(c10::IntArrayRef size, at::ScalarType dtype, bool pin_memory,
                         c10::optional<c10::MemoryFormat> memory_format_opt) {
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return at::detail::empty_generic(size, allocator, cpu_ks, dtype, memory_format_opt);
}

at::TensorBase empty_strided_cpu(c10::IntArrayRef size, c10::IntArrayRef stride,
                                 at::ScalarType dtype, bool pin_memory) {
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return at::detail::empty_strided_generic(
      size, stride, allocator, cpu_ks, dtype);
}

at::TensorBase empty_cpu(
    c10::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == at::DeviceType::CPU, OPS_ERROR(ErrCode::PARAM));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == at::Layout::Strided, OPS_ERROR(ErrCode::PARAM));

  auto pin_memory = c10::pinned_memory_or_default(pin_memory_opt);
  auto dtype = dtype_or_default(dtype_opt);
  return empty_cpu(size, dtype, pin_memory, memory_format_opt);
}

at::TensorBase empty_cpu(
    c10::IntArrayRef size, const at::TensorOptions &options) {
  return empty_cpu(
      size,
      c10::optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

at::TensorBase empty_strided_cpu(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == at::DeviceType::CPU, OPS_ERROR(ErrCode::PARAM));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == at::Layout::Strided, OPS_ERROR(ErrCode::PARAM));

  auto pin_memory = c10::pinned_memory_or_default(pin_memory_opt);
  auto dtype = dtype_or_default(dtype_opt);
  return empty_strided_cpu(size, stride, dtype, pin_memory);
}

at::TensorBase empty_strided_cpu(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    const at::TensorOptions &options) {
  return empty_strided_cpu(
      size,
      stride,
      c10::optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

at::Tensor empty_memory_format(c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt, c10::optional<c10::MemoryFormat> memory_format_opt) {
  return empty_cpu(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
}

at::Tensor empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt,
                         c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt) {
  return empty_strided_cpu(size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("empty.memory_format", TORCH_FN(empty_memory_format));
  m.impl("empty_strided", TORCH_FN(empty_strided));
}
}
}
