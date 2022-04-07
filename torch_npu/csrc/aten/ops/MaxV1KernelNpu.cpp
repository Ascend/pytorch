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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

std::tuple<at::Tensor&, at::Tensor&> max_v1_out_npu_nocheck(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices) {
  OpCommand cmd;
  cmd.Name("ArgMaxWithValue")
      .Input(self)
      .Output(indices)      
      .Output(output)
      .Attr("dimension", dim)
      .Attr("keep_dims", keepdim)
      .Run();
  return std::tie(output, indices);
}

std::tuple<at::Tensor, at::Tensor> npu_max_npu(const at::Tensor& self, int64_t dim, bool keepdim) {
  c10::SmallVector<int64_t, SIZE> dims = {dim};
  c10::SmallVector<int64_t, SIZE> outputSize =
      reduce_ops_npu_output_size(self, dims, keepdim);
  c10::SmallVector<int64_t, SIZE> indicesSize =
      reduce_ops_npu_output_size(self, dims, keepdim);
  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
  if (outputSize.empty()) {
    npu_format = ACL_FORMAT_NCHW;
  }
  at::Tensor outputs = OpPreparation::ApplyTensorWithFormat(
      outputSize, self.options(), npu_format);
  at::Tensor indices = OpPreparation::ApplyTensorWithFormat(
      indicesSize, self.options().dtype(at::kInt), ACL_FORMAT_NCHW);
  max_v1_out_npu_nocheck(self, dim, keepdim, outputs, indices);
  return std::tie(outputs, indices);
}

at::Tensor NPUNativeFunctions::npu_max_backward(
    const at::Tensor& grad, 
    int64_t dim, 
    const at::Tensor& indices, 
    at::IntArrayRef sizes, bool keepdim) {
  at::Tensor new_grad = grad;
  at::Tensor new_indices = indices;
  if (keepdim && sizes.size() > 0) {
    new_grad = grad.squeeze(dim);
    new_indices = indices.squeeze(dim);
  }
  if (new_indices.dtype() == at::kLong) {
    new_indices = NPUNativeFunctions::npu_dtype_cast(new_indices, at::kInt);
  }
  auto grad_input = NPUNativeFunctions::npu_scatter(at::zeros(sizes, new_grad.options()), new_indices, new_grad, dim);
  return grad_input;
}

class NPUMaxFunction : public torch::autograd::Function<NPUMaxFunction> {
public:
  static tensor_list forward(AutogradContext *ctx,
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["shape"] = self.sizes();
    ctx->saved_data["keepdim"] = keepdim;
    at::AutoNonVariableTypeMode g;
    auto result = npu_max_npu(self, dim, keepdim);
    auto indices = std::get<1>(result);
    ctx->save_for_backward({indices});
    tensor_list result_list = {std::get<0>(result), indices};
    return result_list;
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto dim = ctx->saved_data["dim"].toInt();
    auto sizes = ctx->saved_data["shape"].toIntVector();
    auto keepdim = ctx->saved_data["keepdim"].toBool();
    auto saved = ctx->get_saved_variables();
    auto indices = saved[0];
    at::Tensor result = NPUNativeFunctions::npu_max_backward(grad_outputs[0], dim, indices, sizes, keepdim);

    tensor_list output = {result,
                          at::Tensor(),
                          at::Tensor()};
    return output;
  }
};

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_max(const at::Tensor& self, int64_t dim, bool keepdim) {
  auto output = NPUMaxFunction::apply(self, dim, keepdim);
  std::tuple<at::Tensor, at::Tensor> result(output[0], output[1]);
  return result;
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_max(const at::Tensor& self, at::Dimname dim, bool keepdim) {
  return npu_max(self, dimname_to_position(self, dim), keepdim);
}

} // namespace native
} // namespace at_npu
