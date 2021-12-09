// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> cdist_npu_input(
    const Tensor& x1,
    const Tensor& x2) {
  SmallVector<NPUTensorDesc, N> inputs = CalcuOpUtil::create_npu_input_tensor_desc({x1, x2});
  return inputs;
}

SmallVector<NPUAttrDesc, N> cdist_npu_attr(const float p) {
  NPUAttrDesc npuAttrValue = NPUAttrDesc("p", p);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrValue};
  return attrs;
}

SmallVector<NPUTensorDesc, N> cdist_npu_output(const Tensor& result) {
  return CalcuOpUtil::create_npu_output_tensor_desc({result});
}

Tensor cdist_npu(
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    c10::optional<int64_t> compute_mode) {
  return at::_cdist_forward(x1, x2, p, compute_mode);
}

Tensor& cdist_out_npu(
    Tensor &result,
    const Tensor& x1,
    const Tensor& x2,
    const float p_float) {
  auto inputs = cdist_npu_input(x1, x2);
  auto attrs = cdist_npu_attr(p_float);
  auto outputs = cdist_npu_output({result});

  // Executing the NPU operator.
  CalcuOpUtil::execute_npu_operate("Cdist", inputs, outputs, attrs);

  return result;
}

Tensor _cdist_forward_npu(
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    c10::optional<int64_t> compute_mode) {
  TORCH_CHECK(x1.dim() >= 2, "cdist only supports at least 2D tensors, X1 got: ", x1.dim(), "D");
  TORCH_CHECK(x2.dim() >= 2, "cdist only supports at least 2D tensors, X2 got: ", x2.dim(), "D");
  TORCH_CHECK(x1.size(-1) == x2.size(-1), "X1 and X2 must have the same number of columns. X1: ", x1.size(-1), " X2: ", x2.size(-1));
  TORCH_CHECK(at::isFloatingType(x1.scalar_type()), "cdist only supports floating-point dtypes, X1 got: ", x1.scalar_type());
  TORCH_CHECK(at::isFloatingType(x1.scalar_type()), "cdist only supports floating-point dtypes, X2 got: ", x2.scalar_type());
  TORCH_CHECK(p >= 0, "cdist only supports non-negative p values");

  // Since double is not supported in NPU, the type of P needs to be converted from double to float.
  float p_float;
  if (std::isinf(p)) {
    p_float = std::numeric_limits<float>::infinity();
  }
  else {
    TORCH_CHECK(p <= std::numeric_limits<float>::max(), "npu dose not support float64" );
    p_float = static_cast<float>(p);
  }

  int64_t mode = compute_mode.value_or(0);
  TORCH_CHECK(mode >= 0 && mode <= 2, "possible modes: 0, 1, 2, but was: ", mode);
  
  // Broadcast
  int64_t c1 = x1.size(-1);
  int64_t c2 = x2.size(-1);
  int64_t r1 = x1.size(-2);
  int64_t r2 = x2.size(-2);  
  auto dim1 = x1.dim();
  auto dim2 = x2.dim();

  IntArrayRef batch_tensor1(x1.sizes().data(), dim1 - 2);
  IntArrayRef batch_tensor2(x2.sizes().data(), dim2 - 2);
  std::vector<int64_t> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);
  std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
  tensor1_expand_size.insert(tensor1_expand_size.end(), {r1, c1});
  std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
  tensor2_expand_size.insert(tensor2_expand_size.end(), {r2, c2});

  int expand_batch_product = std::accumulate(expand_batch_portion.begin(), expand_batch_portion.end(), 1, std::multiplies<int64_t>());
  std::vector<int64_t> tensor1_view{expand_batch_product, r1, 1, c1};
  std::vector<int64_t> tensor2_view{expand_batch_product, 1, r2, c2};
  std::vector<int64_t> result_size{expand_batch_product, r1, r2};
  std::vector<int64_t> tensor_broadcast_size = infer_size(tensor1_view, tensor2_view);

  // Broadcast batch dim.
  Tensor tensor1_expanded = x1.expand(tensor1_expand_size).contiguous().view(tensor1_view);
  Tensor tensor2_expanded = x2.expand(tensor2_expand_size).contiguous().view(tensor2_view);
  
  // Broadcast r1 and r2.
  Tensor tensor1_broadcast = tensor1_expanded.expand(tensor_broadcast_size).contiguous();
  Tensor tensor2_broadcast = tensor2_expanded.expand(tensor_broadcast_size).contiguous();
  
  // Executing the NPU operator.
  auto output_size = cdist_npu_output_size(x1, x2);

  Tensor result = at::empty_with_format(
      result_size,
      tensor1_broadcast.options(),
      CalcuOpUtil::get_tensor_npu_format(tensor1_broadcast));

  cdist_out_npu(result, tensor1_broadcast, tensor2_broadcast, p_float);

  return result.view(output_size);
}

} // namespace native
} // namespace at