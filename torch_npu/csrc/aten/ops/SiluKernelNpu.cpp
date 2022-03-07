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

#include <torch/csrc/autograd/custom_function.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

at::Tensor& silu_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Swish")
     .Input(self)
     .Output(result)
     .Run();
  return result;
}

at::Tensor& silu_out_npu(const at::Tensor& self, at::Tensor& out){
  OpPreparation::CheckOut(
      {self},
      out,
      self);
    
  if (!NpuUtils::check_match(&out)) {
    at::Tensor contiguousOut = NpuUtils::format_contiguous(out);
    at::Tensor newOut = silu_out_npu_nocheck(contiguousOut, self);
    NpuUtils::format_fresh_view(out, newOut);
  } else {
    silu_out_npu_nocheck(out, self);
  }

  return out;
}

at::Tensor silu_kerner_npu(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);

  silu_out_npu_nocheck(result, self);

  return result;
}

at::Tensor& NPUNativeFunctions::npu_silu_(at::Tensor& self) {
  silu_out_npu(self, self);
  return self;
}

at::Tensor& silu_backward_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& grad_output,
    const at::Tensor& x0, 
    const at::Tensor& x1) {

  OpCommand cmd;
  cmd.Name("SwishGrad")
    .Input(grad_output)
    .Input(x0)
    .Input(x1)
    .Output(result)
    .Run();

  return result;
}

at::Tensor NPUNativeFunctions::npu_silu_backward(const at::Tensor& grad_output, const at::Tensor& x0, const at::Tensor& x1) {
  // construct the output tensor of the NPU
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output);

  // calculate the output result of the NPU
  silu_backward_out_npu_nocheck(grad_input, grad_output, x0, x1);

  return grad_input;
}

class NPUSiluFunction : public torch::autograd::Function<NPUSiluFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
    const at::Tensor& self) {
      at::AutoNonVariableTypeMode g;
      at::Tensor result = silu_kerner_npu(self);
      ctx->save_for_backward({self, result});

      return result;
    }
    
  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
      auto saved = ctx->get_saved_variables();
      auto input = saved[0];
      auto result = saved[1];

      at::Tensor output = NPUNativeFunctions::npu_silu_backward(grad_outputs[0], input, result);
      tensor_list outputlist = {output};

      return outputlist;
    }
};

at::Tensor NPUNativeFunctions::npu_silu(const at::Tensor& self) {
  return NPUSiluFunction::apply(self);
}

} // namespace native
} // namespace at_npu