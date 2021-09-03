
#ifndef __NATIVE_NPU_TENSOR_INTERFACE__
#define __NATIVE_NPU_TENSOR_INTERFACE__
#include "ATen/native/npu/common/FormatCastHelper.h"
#include "ATen/native/npu/frame/FormatHelper.h"
#include<torch/script.h>
#include <tuple>

namespace at {
namespace native {
using namespace at::native::npu;

Tensor npu_format_cast(const Tensor& src, int64_t acl_format);

Tensor& npu_format_cast_(Tensor& self, int64_t acl_format);

Tensor& npu_format_cast_(Tensor& self, const Tensor& src);

Tensor& npu_dtype_cast_(Tensor& self, const Tensor& src);

Tensor npu_dtype_cast(const Tensor& self, ScalarType dtype);

Tensor& copy_memory_(Tensor& self, const Tensor& src, bool non_blocking);

Tensor npu_broadcast(const Tensor& self, IntArrayRef size);

Tensor& npu_broadcast_out(const Tensor& self, IntArrayRef size, Tensor& result);


Tensor npu_indexing(const Tensor& self,
    IntArrayRef begin,
    IntArrayRef end,
    IntArrayRef strides);

Tensor& npu_indexing_out(const Tensor& self,
    IntArrayRef begin,
    IntArrayRef end,
    IntArrayRef strides,
    Tensor& result);


Tensor npu_slice(const Tensor& self,
    IntArrayRef begin,
    IntArrayRef end,
    IntArrayRef strides);

Tensor& npu_slice_out(const Tensor& self,
    IntArrayRef begin,
    IntArrayRef end,
    IntArrayRef strides,
    Tensor& result);

Tensor npu_transpose_to_contiguous(const Tensor& self);

Tensor npu_transpose(const Tensor& self, IntArrayRef perm);

Tensor npu_stride_add(  const Tensor& self,
    const Tensor& other,
    Scalar offset1,
    Scalar offset2,
    Scalar c1_len);

Tensor empty_with_format(IntArrayRef size,
              c10::optional<DimnameList> names,
              c10::optional<ScalarType> dtype_opt,
              c10::optional<Layout> layout_opt,
              c10::optional<Device> device_opt,
              c10::optional<bool> pin_memory_opt,
              int64_t dst_format);

Tensor empty_with_format(IntArrayRef size,
              c10::optional<ScalarType> dtype_opt,
              c10::optional<Layout> layout_opt,
              c10::optional<Device> device_opt,
              c10::optional<bool> pin_memory_opt,
              int64_t dst_format);

Tensor empty_like(
    const Tensor& self,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> optional_memory_format);

Tensor& npu_transpose_out(const Tensor& self,
    IntArrayRef perm,
    Tensor& result);

Tensor& one_(Tensor& self);

Tensor ones_like(const Tensor& self,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    optional<c10::MemoryFormat> optional_memory_format);

Tensor ones(IntArrayRef size,     
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt);

Tensor zeros(IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt);

Tensor& zero_(Tensor& self);

Tensor zeros_like(
    const Tensor& self,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> optional_memory_format);

Tensor npu_conv2d(
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups);

Tensor& npu_conv2d_out(const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    Tensor& result);

std::tuple<Tensor, Tensor, Tensor> npu_conv2d_backward(
    const Tensor& input,
    const Tensor& grad,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> grad_input_mask);


Tensor& npu_conv3d_out(const Tensor &input,
                       const Tensor &weight, 
                       const optional<Tensor> &bias_opt,
                       IntArrayRef stride, 
                       IntArrayRef padding,
                       IntArrayRef dilation, 
                       int64_t groups,
                       Tensor &result);

Tensor npu_conv3d(const Tensor &input, 
                  const Tensor &weight, 
                  const optional<Tensor> &bias_opt,
                  IntArrayRef stride, 
                  IntArrayRef padding, 
                  IntArrayRef dilation,
                  int64_t groups);

std::tuple<Tensor, Tensor, Tensor>
npu_conv3d_backward(const Tensor &input, const Tensor &grad,
                    const Tensor &weight, IntArrayRef stride,
                    IntArrayRef padding, IntArrayRef dilation, int64_t groups,
                    std::array<bool, 3> grad_input_mask);

Tensor npu_conv_transpose2d(
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups);
}
}
#endif