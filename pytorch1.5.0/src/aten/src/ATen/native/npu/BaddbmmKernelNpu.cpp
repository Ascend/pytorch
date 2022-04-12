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
#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& baddbmm_nocheck(
    const Tensor& self,	
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar beta,
    Scalar alpha,
    Tensor& result) {
  auto outputSize = baddbmm_npu_output_size(tensor1, tensor2);
  Tensor BatchMatMulTensor = OpPreparation::ApplyTensor(self, outputSize);
  bool isSelfT = CalcuOpUtil::is_transpose_last_two_dims(tensor1);
  bool isMat2T = CalcuOpUtil::is_transpose_last_two_dims(tensor2);

  OpCommand cmd;
  cmd.Name("BatchMatMul")
     .Input(tensor1)
     .Input(tensor2) 
     .Output(BatchMatMulTensor)
     .Attr("adj_x1", isSelfT)
     .Attr("adj_x2", isMat2T)
     .Run();

  Tensor alphaMulTensor = at::mul(BatchMatMulTensor, alpha);
  Tensor betaMulTensor = at::mul(self, beta);
  at::add_out(result, alphaMulTensor, betaMulTensor, 1);
  return result;
}

Tensor& baddbmm_out_npu(
    Tensor& result,
    const Tensor& self,	
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar beta,
    Scalar alpha){
      
  OpPreparation::CheckOut(
      {self, tensor1, tensor2},
      result,
      self);
  baddbmm_nocheck(self, tensor1, tensor2, beta, alpha, result);
  return result;
}

Tensor baddbmm_npu(
    const Tensor& self, 
    const Tensor& tensor1, 
    const Tensor& tensor2, 
    Scalar beta,
    Scalar alpha) {
  Tensor outputTensor = self;
  auto outputSize = baddbmm_npu_output_size(tensor1, tensor2);
  Tensor result = OpPreparation::ApplyTensor(
      outputTensor,
      outputSize);
  baddbmm_out_npu(result, self, tensor1, tensor2, beta, alpha);
  return result;
}

Tensor& baddbmm_npu_(
    Tensor& self, 
    const Tensor& tensor1, 
    const Tensor& tensor2, 
    Scalar beta,
    Scalar alpha) {
  SmallVector<Tensor, N> inputs = {self, tensor1, tensor2};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);
    
  if (!NpuUtils::check_match(&self)) {
      Tensor contiguousSelf = NpuUtils::format_contiguous(self);
      Tensor result = baddbmm_out_npu(contiguousSelf, contiguousSelf, tensor1, tensor2, beta, alpha);
      NpuUtils::format_fresh_view(self, result);
  } else {
      baddbmm_out_npu(self, self, tensor1, tensor2, beta, alpha);
  }

  return self;
}
}
}
