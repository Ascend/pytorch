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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

static void check_cdist_backward_input(    
    const at::Tensor& grad,
    const at::Tensor& x1,
    const at::Tensor& x2,
    const double p,
    const at::Tensor& cdist) {
  TORCH_CHECK(x1.is_contiguous(), "_cdist_backward requires X1 to be contiguous");
  TORCH_CHECK(x2.is_contiguous(), "_cdist_backward requires X2 to be contiguous");
  TORCH_CHECK(cdist.is_contiguous(), "_cdist_backward requires dist to be contiguous");
  TORCH_CHECK(grad.is_contiguous(), "_cdist_backward requires grad to be contiguous");
  auto device1 = x1.device().type();
  TORCH_CHECK(device1 == at::kCPU || device1 == at::kCUDA || device1 == at::kNPU, "_cdist_backward only supports CPU, CUDA and NPU devices, X1 got: ", device1);
  auto device2 = x2.device().type();
  TORCH_CHECK(device2 == at::kCPU || device2 == at::kCUDA || device2 == at::kNPU, "_cdist_backward only supports CPU, CUDA and NPU devices, X2 got: ", device2);
  TORCH_CHECK(p <= std::numeric_limits<float>::max(), "npu dose not support float64" );
}

at::Tensor NPUNativeFunctions::_cdist_backward(
    const at::Tensor& grad,
    const at::Tensor& x1,
    const at::Tensor& x2,
    const double p,
    const at::Tensor& cdist) {
  check_cdist_backward_input(grad, x1, x2, p, cdist);
  float p_float;
  if (std::isinf(p)) {
    p_float = std::numeric_limits<float>::infinity();
  }
  else {
    p_float = static_cast<float>(p);
  }
 
  auto dim1 = x1.dim();
  auto dim2 = x2.dim();
  c10::SmallVector<int64_t, SIZE> tensor1_expand_size = array_to_small_vector(x1.sizes());
  tensor1_expand_size.insert(tensor1_expand_size.begin() + (dim1 - 1), 1);
  c10::SmallVector<int64_t, SIZE> tensor2_expand_size = array_to_small_vector(x2.sizes());
  tensor2_expand_size.insert(tensor2_expand_size.begin() + (dim2 - 2), 1);
  c10::SmallVector<int64_t, SIZE> grad_expand_size = array_to_small_vector(grad.sizes());
  grad_expand_size.insert(grad_expand_size.end(), 1);
  c10::SmallVector<int64_t, SIZE> cdist_expand_size = array_to_small_vector(cdist.sizes());
  cdist_expand_size.insert(cdist_expand_size.end(), 1);
  std::vector<int64_t> tensor_broadcast_size = at::infer_size(tensor1_expand_size, tensor2_expand_size);

  at::Tensor tensor1_broadcast = x1.view(tensor1_expand_size).expand(tensor_broadcast_size).contiguous();
  at::Tensor tensor2_broadcast = x2.view(tensor2_expand_size).expand(tensor_broadcast_size).contiguous();
  at::Tensor grad_broadcast = grad.view(grad_expand_size).expand(tensor_broadcast_size).contiguous();
  at::Tensor cdist_broadcast = cdist.view(cdist_expand_size).expand(tensor_broadcast_size).contiguous();
  auto outputSize = x1.sizes();
  at::Tensor result = OpPreparation::ApplyTensor(tensor1_broadcast, outputSize);

  OpCommand cmd;
  cmd.Name("CdistGrad")
      .Input(grad_broadcast)
      .Input(tensor1_broadcast)
      .Input(tensor2_broadcast)
      .Input(cdist_broadcast)
      .Attr("p", p_float)
      .Output(result)
      .Run();
  return result;
}

} // namespace native
} // namespace at_npu
