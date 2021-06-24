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

SmallVector<NPUTensorDesc, N> baddbmm_npu_input(
    const Tensor& self,
    const Tensor& other) {
  return CalcuOpUtil::create_npu_input_tensor_desc({self, other});
}

SmallVector<NPUTensorDesc, N> baddbmm_npu_output(
    const Tensor& result) {
  return CalcuOpUtil::create_npu_output_tensor_desc({result});
}

SmallVector<NPUAttrDesc, N> baddbmm_npu_attr(
    const Tensor& self,
    const Tensor& mat2) {
  bool isSelfT = CalcuOpUtil::is_transpose_last_two_dims(self);
  bool isMat2T = CalcuOpUtil::is_transpose_last_two_dims(mat2);
  NPUAttrDesc npuAttrSelfTranspose = NPUAttrDesc("adj_x1", isSelfT);
  NPUAttrDesc npuAttrMat2Transpose = NPUAttrDesc("adj_x2", isMat2T);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrSelfTranspose, npuAttrMat2Transpose};
  return attrs;
}

Tensor& baddbmm_out_npu(
    Tensor& result,
    const Tensor& self,	
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar beta,
    Scalar alpha) {
  Tensor BatchMatMulTensor = result;
  
  auto inputs = baddbmm_npu_input(tensor1, tensor2);
  auto outputs = baddbmm_npu_output({BatchMatMulTensor});
  auto attrs = baddbmm_npu_attr(tensor1, tensor2);
  CalcuOpUtil::execute_npu_operate("BatchMatMul", inputs, outputs, attrs);

  Tensor alphaMulTensor = at::mul(BatchMatMulTensor, alpha);

  Tensor betaMulTensor = at::mul(self, beta);
  
  at::add_out(result, alphaMulTensor, betaMulTensor);

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
  Tensor result = at::empty_with_format(
      outputSize,
      outputTensor.options(),
      CalcuOpUtil::get_tensor_npu_format(outputTensor));
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