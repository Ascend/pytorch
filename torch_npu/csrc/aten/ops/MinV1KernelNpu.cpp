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
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

tuple<at::Tensor&, at::Tensor&> min_v1_out_npu_nocheck(
    at::Tensor& output,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  OpCommand cmd;
  cmd.Name("ArgMinWithValue")
    .Input(self)
    .Output(indices)
    .Output(output)
    .Attr("dimension", dim)
    .Attr("keep_dims", keepdim)
    .Run();

  return std::tie(output, indices);
}

tuple<at::Tensor, at::Tensor> min_v1_npu(const at::Tensor& self, int64_t dim, bool keepdim) {
  c10::SmallVector<int64_t, SIZE> dims = {dim};
  c10::SmallVector<int64_t, SIZE> outputSize =
      reduce_ops_npu_output_size(self, dims, keepdim);
  c10::SmallVector<int64_t, SIZE> indicesSize =
      reduce_ops_npu_output_size(self, dims, keepdim);

  int64_t npuFormat = CalcuOpUtil::get_tensor_npu_format(self);
  if (outputSize.empty()) {
    npuFormat = ACL_FORMAT_NCHW;
  }

  at::Tensor outputs = OpPreparation::ApplyTensorWithFormat(outputSize, self.options(), npuFormat);
  at::Tensor indices = OpPreparation::ApplyTensorWithFormat(indicesSize, self.options().dtype(at::kInt), npuFormat);

  min_v1_out_npu_nocheck(outputs, indices, self, dim, keepdim);
  return std::tie(outputs, indices);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_min(const at::Tensor& self, at::Dimname dim, bool keepdim) {
  return min_v1_npu(self, dimname_to_position(self, dim), keepdim);
}

at::Tensor NPUNativeFunctions::npu_min_backward(
    const at::Tensor& grad,
    int64_t dim,
    const at::Tensor& indices,
    at::IntArrayRef sizes,
    bool keepdim) {
  at::Tensor newGrad = grad;
  at::Tensor newIndices = indices;
  if (keepdim && sizes.size() > 0) {
    newGrad = grad.squeeze(dim);
    newIndices = indices.squeeze(dim);
  }
  auto gradInput = NPUNativeFunctions::npu_scatter(
      at::native::zeros(sizes, newGrad.options()), newIndices, newGrad, dim);
  return gradInput;
}

class NPUMinFunction : public torch::autograd::Function<NPUMinFunction> {
public:
  static tensor_list forward(AutogradContext *ctx,
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["keepdim"] = keepdim;
    ctx->saved_data["size"] = self.sizes();
    at::AutoNonVariableTypeMode g;
    auto result = min_v1_npu(self, dim, keepdim);
    auto indices = std::get<1>(result);
    ctx->save_for_backward({indices});
    tensor_list result_list = {std::get<0>(result), indices};
    return result_list;
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto dim = ctx->saved_data["dim"].toInt();
    auto keepdim = ctx->saved_data["keepdim"].toBool();
    auto size = ctx->saved_data["size"].toIntVector();
    auto saved = ctx->get_saved_variables();
    auto indices = saved[0];
    at::Tensor result = NPUNativeFunctions::npu_min_backward(
        grad_outputs[0], dim, indices, size, keepdim);
    tensor_list output = {result, at::Tensor(), at::Tensor()};
    return output;
  }
};

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_min(const at::Tensor& self, int64_t dim, bool keepdim) {
  auto result = NPUMinFunction::apply(self, dim, keepdim);
  std::tuple<at::Tensor, at::Tensor> output(result[0], result[1]);
  return output;
}

} // namespace native
} // namespace at_npu