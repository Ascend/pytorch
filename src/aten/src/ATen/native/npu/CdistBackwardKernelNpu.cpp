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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

static void check_cdist_backward_input(    
    const Tensor& grad,
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    const Tensor& cdist) {
  TORCH_CHECK(x1.is_contiguous(), "_cdist_backward requires X1 to be contiguous");
  TORCH_CHECK(x2.is_contiguous(), "_cdist_backward requires X2 to be contiguous");
  TORCH_CHECK(cdist.is_contiguous(), "_cdist_backward requires dist to be contiguous");
  TORCH_CHECK(grad.is_contiguous(), "_cdist_backward requires grad to be contiguous");
  auto device1 = x1.device().type();
  TORCH_CHECK(device1 == kCPU || device1 == kCUDA || device1 == kNPU, "_cdist_backward only supports CPU, CUDA and NPU devices, X1 got: ", device1);
  auto device2 = x2.device().type();
  TORCH_CHECK(device2 == kCPU || device2 == kCUDA || device2 == kNPU, "_cdist_backward only supports CPU, CUDA and NPU devices, X2 got: ", device2);
  TORCH_CHECK(p <= std::numeric_limits<float>::max(), "npu dose not support float64" );
}

Tensor _cdist_backward_npu(
    const Tensor& grad,
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    const Tensor& cdist) {
  
  check_cdist_backward_input(grad, x1, x2, p, cdist);
  
  // Since double is not supported in NPU, the type of P needs to be converted from double to float.
  float p_float;
  if (std::isinf(p)) {
    p_float = std::numeric_limits<float>::infinity();
  }
  else {
    p_float = (float) p;
  }

  // Broadcast  
  auto dim1 = x1.dim();
  auto dim2 = x2.dim();

  SmallVector<int64_t, SIZE> tensor1_expand_size = array_to_small_vector(x1.sizes());
  tensor1_expand_size.insert(tensor1_expand_size.begin() + (dim1 - 1), 1);

  SmallVector<int64_t, SIZE> tensor2_expand_size = array_to_small_vector(x2.sizes());
  tensor2_expand_size.insert(tensor2_expand_size.begin() + (dim2 - 2), 1);

  SmallVector<int64_t, SIZE> grad_expand_size = array_to_small_vector(grad.sizes());
  grad_expand_size.insert(grad_expand_size.end(), 1);

  SmallVector<int64_t, SIZE> cdist_expand_size = array_to_small_vector(cdist.sizes());
  cdist_expand_size.insert(cdist_expand_size.end(), 1);

  std::vector<int64_t> tensor_broadcast_size = infer_size(tensor1_expand_size, tensor2_expand_size);

  Tensor tensor1_broadcast = x1.view(tensor1_expand_size).expand(tensor_broadcast_size).contiguous();
  Tensor tensor2_broadcast = x2.view(tensor2_expand_size).expand(tensor_broadcast_size).contiguous();
  Tensor grad_broadcast = grad.view(grad_expand_size).expand(tensor_broadcast_size).contiguous();
  Tensor cdist_broadcast = cdist.view(cdist_expand_size).expand(tensor_broadcast_size).contiguous();

  //Executing the NPU operator.
  auto outputSize = input_same_output_size(x1);
  Tensor result = OpPreparation::ApplyTensor(tensor1_broadcast, outputSize);
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
} // namespace at