#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor true_divide_Tensor(const at::Tensor& self, const at::Tensor& divisor);

at::Tensor& true_divide_out_Tensor(const at::Tensor& self, const at::Tensor& divisor, at::Tensor& result);

at::Tensor& true_divide__Tensor(at::Tensor& self, const at::Tensor& divisor);

bool _has_compatible_shallow_copy_type(const at::Tensor &self, const at::Tensor &from);

at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt);

at::Tensor empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt);
bool is_pinned(const at::Tensor& self, c10::optional<at::Device> device);

at::Tensor _pin_memory(const at::Tensor& self, c10::optional<at::Device> device);

at::Tensor _to_copy(
    const at::Tensor& self,
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<c10::MemoryFormat> optional_memory_format);
}
}
