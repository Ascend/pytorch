// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include <ATen/native/Resize.h>

static inline void resize_out(const at::Tensor &out, at::IntArrayRef sizes,
                              at::IntArrayRef strides, const at::TensorOptions &options) {
  TORCH_CHECK(options.dtype() == out.dtype(), "Expected out Tensor to have dtype ",
              options.dtype(), ", but got ", out.dtype(), " instead");
  TORCH_CHECK(options.device() == out.device(), "Expected out Tensor to have device ",
              options.device(), ", but got ", out.device(), " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      at::native::as_strided_(out, sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
    }
  }
}

namespace at_npu {
namespace native {

struct structured_gcd_out_out final : public at::native::structured_gcd_out {
  structured_gcd_out_out(at::Tensor& out0) : outputs_{ std::ref(out0) } {}

  void set_output(int64_t output_idx, at::IntArrayRef sizes, at::IntArrayRef strides,
                  at::TensorOptions options, at::DimnameList names) override {
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    if (!names.empty()) {
      at::namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_gcd_out::set_output(output_idx, sizes, strides, options, names);
  }

  const at::Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx];
  }
  std::array<std::reference_wrapper<at::Tensor>, 1> outputs_;
};

at::Tensor& NPUNativeFunctions::gcd_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out) {
  // convert args to cpu in order to use at::native kernel
  TORCH_NPU_WARN_ONCE("Warning: kernel [gcd.out] is not supported by NPU currently. Now this kernel is running on CPU.");
  const auto self_cpu = self.cpu();
  const auto other_cpu = other.cpu();
  auto out_cpu = out.cpu();

  structured_gcd_out_out op(out_cpu);
  op.meta(self_cpu, other_cpu);
  op.impl(self_cpu, other_cpu, op.outputs_[0]);
  out.copy_(out_cpu);
  return out;
}
}  // namespace native
}  // namespace at_npu
