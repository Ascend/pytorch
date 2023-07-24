// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

const int8_t ALLOW_FP32_DOWN_PRECISION = 1;
const int8_t KEEP_DTYPE = 0;

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;

static c10::SmallVector<int64_t, SIZE> get_npu_output_size(const at::Tensor &tensor1,
                                                           const at::Tensor &tensor2) {
  c10::SmallVector<int64_t, SIZE> output_size;
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();

  TORCH_CHECK(dim_tensor1 > 0 && dim_tensor2 > 0,
              "matmul got error dimentions: ", "(", dim_tensor1, ", ",
              dim_tensor2, ")");

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    output_size = {};

  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    output_size = {tensor1.size(0)};

  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    output_size = {tensor2.size(1)};

  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    output_size = {tensor1.size(0), tensor2.size(1)};

  } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
    // t1:(N, n, m) * t2:(m, p)
    auto size1 = tensor1.sizes();
    auto tmp = std::vector<int64_t>{tensor2.size(0), 1};
    auto size2 = dim_tensor2 == 1 ? tmp : tensor2.sizes();
    output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
    if (dim_tensor2 > 1) {
      output_size.push_back(size2[dim_tensor2 - 1]);
    }

  } else if ((dim_tensor1 == 1 || dim_tensor1 == 2) && dim_tensor2 >= 3) {
    auto tmp = c10::SmallVector<int64_t, SIZE>{1, tensor1.size(0)};
    auto size1 = dim_tensor1 == 1 ? tmp : tensor1.sizes();
    auto size2 = tensor2.sizes();
    output_size.insert(output_size.end(), size2.begin(), size2.end() - 2);
    if (dim_tensor1 > 1) {
      output_size.push_back(size1[0]);
    }
    output_size.push_back(size2[dim_tensor2 - 1]);

  } else if (dim_tensor1 >= 3 && dim_tensor2 >= 3) {
    // t1:(b1, n, m1) * t2:(x2, m2, p)
    int64_t n = tensor1.size(-2);
    at::IntArrayRef batch_tensor1(tensor1.sizes().data(), dim_tensor1 - 2);
    int64_t p = tensor2.size(-1);
    at::IntArrayRef batch_tensor2(tensor2.sizes().data(), dim_tensor2 - 2);
    std::vector<int64_t> expand_batch_portion =
        at::infer_size(batch_tensor1, batch_tensor2);
    std::vector<int64_t> output_expand_size(expand_batch_portion);
    output_expand_size.insert(output_expand_size.end(), {n, p});
    output_size = output_expand_size;

  } else {
    TORCH_CHECK(false, "matmul got error sizes: ", "(", dim_tensor1, ", ",
                dim_tensor2, ")");
  }

  return output_size;
}

static at::Tensor matmul_forward_npu(c10::optional<at::Tensor> out_opt,
                                     const at::Tensor &tensor1,
                                     const at::Tensor &tensor2) {

  at::NoNamesGuard guard;
  at::Tensor out;
  auto output_size = get_npu_output_size(tensor1, tensor2);
  if (out_opt.has_value()) {
    out = out_opt.value();
    OpPreparation::CheckOut({tensor1, tensor2}, out, tensor1, output_size);
  } else {
    out =
        OpPreparation::ApplyTensorWithoutFormat(output_size, tensor1.options());
  }

  // allow dicrease precision
  int8_t cube_math_type = ALLOW_FP32_DOWN_PRECISION;
  EXEC_NPU_CMD(aclnnMatmul, tensor1, tensor2, out, cube_math_type);
  return out;
}

std::tuple<at::Tensor, at::Tensor> matmul_backward_npu(const at::Tensor &grad,
                                                       const at::Tensor &self,
                                                       const at::Tensor &other) {
  if (!grad.defined()) {
    return std::make_tuple(at::Tensor(), at::Tensor());
  }

  auto mat1 = self;
  auto mat2 = other;
  auto dim_tensor1 = self.dim();
  auto dim_tensor2 = other.dim();
  std::vector<int64_t> new_shape(grad.sizes().begin(), grad.sizes().end());
  // unsqueese mat1 & mat2
  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    mat1 = self.view({1, self.size(0)});
    mat2 = other.view({other.size(0), 1});
    new_shape = {1, 1};
  }
  // unsqueese mat1
  if (dim_tensor1 == 1 && dim_tensor2 > 1) {
    mat1 = self.view({1, self.size(0)});
    new_shape.insert(new_shape.begin() + new_shape.size() - 1, 1);
  }
  // unsqueese mat2
  if (dim_tensor1 > 1 && dim_tensor2 == 1) {
    mat2 = other.view({other.size(0), 1});
    new_shape.insert(new_shape.end(), 1);
  }

  // backward input 0
  auto grad_self = matmul_forward_npu(c10::nullopt, grad.view(new_shape),
                                      mat2.transpose(-2, -1));
  // backward input 1
  auto grad_other = matmul_forward_npu(c10::nullopt, mat1.transpose(-2, -1),
                                       grad.view(new_shape));
  if (grad_other.size(-1) == 1 && other.dim() == 1) {
    grad_other = grad_other.squeeze(-1);
  }

  return std::make_tuple(grad_self, grad_other);
}

class NPUMatmulOpApiFunction
    : public torch::autograd::Function<NPUMatmulOpApiFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx, const at::Tensor &self,
                            const at::Tensor &other) {
    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({self, other});
    auto result = matmul_forward_npu(c10::nullopt, self, other);
    return result;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto self = saved[0];
    auto other = saved[1];

    auto grads = matmul_backward_npu(grad_outputs[0], self, other);
    tensor_list output = {std::get<0>(grads), std::get<1>(grads)};

    return output;
  }
};

at::Tensor NPUNativeOpApiFunctions::matmul(const at::Tensor &tensor1,
                                           const at::Tensor &tensor2) {
  DO_COMPATIBILITY(aclnnMatmul, NPUNativeFunctions::matmul(tensor1, tensor2));
  auto maybe_outnames =
      at::namedinference::compute_matmul_outnames(tensor1, tensor2);
  auto result = NPUMatmulOpApiFunction::apply(tensor1, tensor2);
  at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

at::Tensor &NPUNativeOpApiFunctions::matmul_out(const at::Tensor &tensor1,
                                                const at::Tensor &tensor2,
                                                at::Tensor &result) {
  DO_COMPATIBILITY(aclnnMatmul,
                   NPUNativeFunctions::matmul_out(tensor1, tensor2, result));
  auto maybe_outnames =
      at::namedinference::compute_matmul_outnames(tensor1, tensor2);
  // matmul_out don't support
  matmul_forward_npu(c10::optional<at::Tensor>(result), tensor1, tensor2);
  at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

} // namespace native
} // namespace at_npu
