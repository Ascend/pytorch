#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include <ATen/native/Pool.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::max_pool2d_with_indices_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices,
    at::Tensor& grad_input) {
  at::Tensor self_cp = self;
  at::Tensor grad_output_cp = grad_output;
  at::Tensor indices_cp = indices;
  if (self.dim() == 3) {
    self_cp = self.unsqueeze(0);
    grad_output_cp = grad_output.unsqueeze(0);
    indices_cp = indices.unsqueeze(0);
    grad_input.unsqueeze_(0);
  }
  int64_t stride_h = 1;
  int64_t stride_w = 1;
  if (stride.empty()) {
    stride_h = kernel_size[0];
    stride_w = kernel_size[1];
  } else {
    stride_h = stride[0];
    stride_w = stride[1];
  }

  c10::SmallVector<int64_t, N> kernel_sizes = {1, kernel_size[0], kernel_size[1], 1};
  c10::SmallVector<int64_t, N> strides_size = {1, stride_h, stride_w, 1};
  c10::SmallVector<int64_t, N> paddings = {1, padding[0], padding[1], 1};
  c10::SmallVector<int64_t, N> dilations = {1, dilation[0], dilation[1], 1};
  OpCommand cmd;
  cmd.Name("MaxPoolGradWithArgmaxV1")
      .Input(self_cp, "x", ACL_FORMAT_NCHW)
      .Input(grad_output_cp, "grad", ACL_FORMAT_NCHW)
      .Input(indices_cp, "argmax", ACL_FORMAT_NCHW, "uint16")
      .Output(grad_input, "y", ACL_FORMAT_NCHW)
      .Attr("ksize", kernel_sizes)
      .Attr("strides", strides_size)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("ceil_mode", ceil_mode)
      .Run();

  if (self.dim() == 3) {
    grad_input.squeeze_(0);
  }
  return grad_input;
}

at::Tensor NPUNativeFunctions::max_pool2d_with_indices_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices) {
  TORCH_CHECK((kernel_size.size() == 1 || kernel_size.size() == 2),
      "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  TORCH_CHECK((stride.size() == 0 || stride.size() == 1 || stride.size() == 2),
      "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  TORCH_CHECK((padding.size() == 1 || padding.size() == 2),
      "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  TORCH_CHECK((dilation.size() == 1 || dilation.size() == 2),
      "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  TORCH_CHECK((self.ndimension() == 3 || self.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  const int k_h = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
  const int k_w = kernel_size.size() == 1 ? k_h : at::native::safe_downcast<int, int64_t>(kernel_size[1]);
  c10::SmallVector<int64_t, SIZE> kernel_sizes = {k_h, k_w};
  at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  const int d_h = stride.empty() ? k_h : at::native::safe_downcast<int, int64_t>(stride[0]);
  const int d_w = stride.empty() ? k_w : stride.size() == 1 ? d_h : at::native::safe_downcast<int, int64_t>(stride[1]);
  c10::SmallVector<int64_t, SIZE> strides = {d_h, d_w};
  at::IntArrayRef stridess = at::IntArrayRef(strides);

  const int pad_h = at::native::safe_downcast<int, int64_t>(padding[0]);
  const int pad_w = padding.size() == 1 ? pad_h : at::native::safe_downcast<int, int64_t>(padding[1]);
  c10::SmallVector<int64_t, SIZE> paddings = {pad_h, pad_w};
  at::IntArrayRef padss = at::IntArrayRef(paddings);

  const int dilation_h = at::native::safe_downcast<int, int64_t>(dilation[0]);
  const int dilation_w = dilation.size() == 1 ? dilation_h : at::native::safe_downcast<int, int64_t>(dilation[1]);
  c10::SmallVector<int64_t, SIZE> dilations = {dilation_h, dilation_w};
  at::IntArrayRef dilationss = at::IntArrayRef(dilations);

  at::Tensor grad_input = OpPreparation::ApplyTensor(self);
  NPUNativeFunctions::max_pool2d_with_indices_backward_out(
      grad_output,
      self,
      kernel_sizess,
      stridess,
      padss,
      dilationss,
      ceil_mode,
      indices,
      grad_input);

  return grad_input;
}
} // namespace native
} // namespace at_npu
