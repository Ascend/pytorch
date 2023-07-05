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

#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include <ATen/NamedTensorUtils.h>

namespace at_npu {
namespace native {

const int8_t ALLOW_FP32_DOWN_PRECISION = 1;
const int8_t KEEP_DTYPE = 0;

static c10::SmallVector<int64_t, SIZE> get_npu_output_size(const at::Tensor &tensor1,
                                                           const at::Tensor &tensor2) {
  c10::SmallVector<int64_t, SIZE> output_size;
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();

  TORCH_CHECK(dim_tensor1 > 0 && dim_tensor2 > 0, "matmul got error dimentions: ", "(", dim_tensor1,
              ", ", dim_tensor2, ")");

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    output_size = {};

  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    auto size1 = c10::SmallVector<int64_t, SIZE>{tensor1.size(0), 1};
    auto size2 = c10::SmallVector<int64_t, SIZE>{1};
    output_size = tensor1.size(0) > 1 ? size1 : size2;

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
    TORCH_CHECK(false, "matmul got error sizes: ", "(", dim_tensor1, ", ", dim_tensor2, ")");
  }

  return output_size;
}

static at::Tensor matmul_opt_npu(c10::optional<at::Tensor> out_opt,
                                 const at::Tensor &tensor1,
                                 const at::Tensor &tensor2) {

  at::NoNamesGuard guard;
  at::Tensor out;
  auto output_size = get_npu_output_size(tensor1, tensor2);
  if (out_opt.has_value()) {
    out = out_opt.value();
    OpPreparation::CheckOut({tensor1, tensor2}, out, tensor1, output_size);
  } else {
    out = OpPreparation::ApplyTensorWithoutFormat(output_size, tensor1.options());
  }

  // allow dicrease precision
  int8_t cube_math_type = ALLOW_FP32_DOWN_PRECISION;
  EXEC_NPU_CMD(aclnnMatmul, tensor1, tensor2, out, cube_math_type);
  return out;
}

at::Tensor NPUNativeOpApiFunctions::matmul(const at::Tensor &tensor1,
                                           const at::Tensor &tensor2) {
  DO_COMPATIBILITY(aclnnMatmul, NPUNativeFunctions::matmul(tensor1, tensor2));
  auto maybe_outnames = at::namedinference::compute_matmul_outnames(tensor1, tensor2);
  auto result = matmul_opt_npu(c10::nullopt, tensor1, tensor2);
  at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

at::Tensor &NPUNativeOpApiFunctions::matmul_out(const at::Tensor &tensor1,
                                                const at::Tensor &tensor2,
                                                at::Tensor &result) {
  DO_COMPATIBILITY(aclnnMatmul, NPUNativeFunctions::matmul_out(tensor1, tensor2, result));
  auto maybe_outnames = at::namedinference::compute_matmul_outnames(tensor1, tensor2);
  matmul_opt_npu(c10::optional<at::Tensor>(result), tensor1, tensor2);
  at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

} // namespace native
} // namespace at_npu
