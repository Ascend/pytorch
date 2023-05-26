#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include <ATen/native/Pool.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::max_pool2d_with_indices_out(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    at::Tensor& output,
    at::Tensor& indices) {
  at::Tensor self_cp = self.dim() == 3 ? self.unsqueeze(0) : self;
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
  cmd.Name("MaxPoolWithArgmaxV1")
      .Input(self_cp, "x", ACL_FORMAT_NCHW)
      .Output(output, "y", ACL_FORMAT_NCHW)
      .Output(indices, "argmax", ACL_FORMAT_NCHW, "uint16")
      .Attr("ksize", kernel_sizes)
      .Attr("strides", strides_size)
      .Attr("pads", paddings)
      .Attr("dilation", dilations)
      .Attr("ceil_mode", ceil_mode)
      .Run();
  if (self.dim() == 3) {
    output.squeeze_(0);
    indices.squeeze_(0);
  }
  return tuple<at::Tensor&, at::Tensor&>(output, indices);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::max_pool2d_with_indices(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
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

  /* sizes */
  const int64_t nbatch = self.ndimension() == 4 ? self.size(-4) : 1;
  const int64_t n_input_plane = self.size(-3);
  const int64_t input_height = self.size(-2);
  const int64_t input_width = self.size(-1);

  const int64_t output_height = at::native::pooling_output_shape<int64_t>(input_height, k_h, pad_h, d_h, dilation_h, ceil_mode);
  const int64_t output_width = at::native::pooling_output_shape<int64_t>(input_width, k_w, pad_w, d_w, dilation_w, ceil_mode);

  at::native::pool2d_shape_check(self, k_h, k_w, d_h, d_w, pad_h, pad_w, dilation_h, dilation_w,
      n_input_plane, input_height, input_width, output_height, output_width, self.suggest_memory_format());

  c10::SmallVector<int64_t, SIZE> output_size = {nbatch, n_input_plane, output_height, output_width};

  const int64_t BLOCKSIZE = 16;
  int64_t mask_h = kernel_size[0] * kernel_size[1];
  int64_t mask_w = (CeilDiv(output_height * output_width, BLOCKSIZE) + 1);
  c10::SmallVector<int64_t, SIZE> indices_size = {nbatch, n_input_plane, mask_h, mask_w};

  at::Tensor output = OpPreparation::ApplyTensor(self, output_size);
  at::Tensor indices = OpPreparation::ApplyTensorWithFormat(self, indices_size, ACL_FORMAT_NC1HWC0, true);

  NPUNativeFunctions::max_pool2d_with_indices_out(
      self, kernel_sizess, stridess, padss, dilationss, ceil_mode, output, indices);

  // The semantics of indices do not match the native API, and the issue has been documented.
  return tuple<at::Tensor, at::Tensor>(output, indices);
}
} // namespace native
} // namespace at_npu
