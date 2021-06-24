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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

// Flexible transpose judgement for view+transpose+Matmul, 
// i.e., tensors with dim=2 and base_size_.size=3 can also be Matmul directly!
bool is_transpose_last_two_dims_flex(const Tensor& tensor) {
  if (tensor.dim() < 2 || tensor.dim() > 3) {
    return false;
  }
  int64_t numel = 1;
  auto storageSize = tensor.storage().get_npu_desc().storage_sizes_;

  for (int i = 0; i < storageSize.size(); i++) {
    numel *= storageSize[i];
  }

  int64_t dim1 = tensor.dim() - 1;
  int64_t dim2 = tensor.dim() - 2;

  if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2) &&
      tensor.storage().size() == numel) {
    return true;
  } else {
    return false;
  }
}

SmallVector<NPUTensorDesc, N> mm_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  Tensor contiguousTensor;
  SmallVector<NPUTensorDesc, N> inputs;

  for (int i = 0; i < inputTensor.size(); i++) {
    // transpose scene is supported by matmul operator.
    if (is_transpose_last_two_dims_flex(inputTensor[i])) {
      contiguousTensor = inputTensor[i];
    } else {
      contiguousTensor = NpuUtils::format_contiguous_add_copy_optimize(inputTensor[i]);
    }

    inputs.emplace_back(NPUTensorDesc(contiguousTensor));
  }

  return inputs;
}

SmallVector<NPUTensorDesc, N> mm_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> mm_npu_attr(
    const Tensor& self,
    const Tensor& mat2) {
  bool isSelfT = is_transpose_last_two_dims_flex(self);
  bool isMat2T = is_transpose_last_two_dims_flex(mat2);

  NPUAttrDesc npuAttrSelfTranspose = NPUAttrDesc("transpose_x1", isSelfT);
  NPUAttrDesc npuAttrMat2Transpose = NPUAttrDesc("transpose_x2", isMat2T);

  SmallVector<NPUAttrDesc, N> attrs = {
      npuAttrSelfTranspose, npuAttrMat2Transpose};

  return attrs;
}

Tensor& mm_out_npu(Tensor& result, const Tensor& self, const Tensor& mat2) {
  // constructs the input and output NPUTensorDesc
  auto inputs = mm_npu_input({self, mat2});
  auto outputs = mm_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = mm_npu_attr(self, mat2);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("MatMul", inputs, outputs, attrs);

  return result;
}

Tensor mm_npu(const Tensor& self, const Tensor& mat2) {
  // calculate the output size
  
  auto outputSize = mm_npu_output_size(self, mat2);

  NPUStorageDesc self_desc = self.storage().unsafeGetStorageImpl()->npu_desc_;
  NPUStorageDesc mat2_desc = mat2.storage().unsafeGetStorageImpl()->npu_desc_;

  Tensor selfFormatCast = self;
  Tensor mat2FormatCast = mat2;
  // Matmul cannot directly deal with view+transposed tensor with NZ format, so Transdata is necessary
  if (self.sizes().size() != self_desc.base_sizes_.size()) {
    selfFormatCast = OpPreparation::CastBackToOriFormat(self);
  }
  
  if (mat2.sizes().size() != mat2_desc.base_sizes_.size()) {
    mat2FormatCast = OpPreparation::CastBackToOriFormat(mat2);
  }
  
  // construct the output tensor of the NPU
  Tensor result;
  
  // TODO(ASCEND): 检查是否指定mm输出为NCHW。待NLP模型总体策略制定后删去
  if ((self.scalar_type() == ScalarType::Half) && !c10::npu::OptionsManager::CheckSwitchMMOutputEnable()) {
    result = at::empty_with_format(
        outputSize, self.options(), ACL_FORMAT_FRACTAL_NZ);
  } else {
    result = at::empty_with_format(outputSize, self.options());
  }

  // calculate the output result of the NPU
  mm_out_npu(result, selfFormatCast, mat2FormatCast);
  return result;
}

} // namespace native
} // namespace at
