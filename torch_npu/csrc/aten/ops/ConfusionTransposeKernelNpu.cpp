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

at::Tensor confusion_transpose_npu(
    const at::Tensor& self,
    at::IntArrayRef perm,
    at::IntArrayRef shape,
    bool transpose_first) {
  c10::SmallVector<int64_t, SIZE> output_size;
  if (transpose_first){
    output_size = array_to_small_vector(shape);
  } else {
    for (int i = 0; i < perm.size(); i++){
      output_size.emplace_back(shape[perm[i]]);
    }
  }

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self, output_size);
  OpCommand cmd;
  cmd.Name("ConfusionTransposeD")
      .Input(self)
      .Output(result)
      .Attr("perm", perm)
      .Attr("shape", shape)
      .Attr("transpose_first", transpose_first)
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::npu_confusion_transpose_backward(
    const at::Tensor& grad,
    at::IntArrayRef perm,
    at::IntArrayRef shape,
    bool transpose_first) {
  c10::SmallVector<int64_t, SIZE> svec_shape;
  if (transpose_first){
    svec_shape = array_to_small_vector(shape);
  } else {
    for (int i = 0; i < perm.size(); i++){
      svec_shape.emplace_back(shape[perm[i]]);
    }
  }
  std::vector<int64_t> vec_perm;
  int64_t perm_len =  perm.size();
  int64_t temp_perm[perm_len] = {0};
  for (int64_t i = 0; i < perm_len; i++){
    temp_perm[perm[i]] = i;
  }
  vec_perm = std::vector<int64_t>(temp_perm, temp_perm+perm_len);
  perm = at::IntArrayRef(vec_perm);

  at::Tensor result = OpPreparation::ApplyTensor(grad, shape);

  OpCommand cmd;
  cmd.Name("ConfusionTransposeD")
      .Input(grad)
      .Output(result)
      .Attr("perm", perm)
      .Attr("shape", svec_shape)
      .Attr("transpose_first", transpose_first)
      .Run();

  return result;
}

class NPUConfusionTransposeFunction : public torch::autograd::Function<NPUConfusionTransposeFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
    const at::Tensor& self,
    at::IntArrayRef perm,
    at::IntArrayRef shape,
    bool transpose_first) {
    ctx->saved_data["perm"] = perm;
    ctx->saved_data["shape"] = self.sizes();
    ctx->saved_data["transpose_first"] = !transpose_first;
    at::AutoNonVariableTypeMode g;
    return confusion_transpose_npu(self, perm, shape, transpose_first);
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto perm = ctx->saved_data["perm"].toIntVector();
    auto shape = ctx->saved_data["shape"].toIntVector();
    auto transpose_first = ctx->saved_data["transpose_first"].toBool();
    at::Tensor result = NPUNativeFunctions::npu_confusion_transpose_backward(grad_outputs[0], perm, shape, transpose_first);

    tensor_list output = {result,
                          at::Tensor(),
                          at::Tensor(),
                          at::Tensor()};
    return output;
  }
};

at::Tensor NPUNativeFunctions::npu_confusion_transpose(const at::Tensor& self,
    at::IntArrayRef perm,
    at::IntArrayRef shape,
    bool transpose_first) {
  return NPUConfusionTransposeFunction::apply(self, perm, shape, transpose_first);
}

} // namespace native
} // namespace at_npu