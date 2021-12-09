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
#include "ATen/native/npu/frame/StorageDescHelper.h"

namespace at {
namespace native {
using namespace at::native::npu;

/*****************************************
Function: is_transpose_last_two_dims_flex
Description: 
  Flexible transpose judgement for view+transpose+Matmul, i.e.,
  tensors with dim=2 and base_size_.size=n can also be Matmul directly! 
Return:
  True--Cases are flex transposed(flex transpose=strict transpose+view
    transpose), which can be refreshed as a input transposed tensor proceed to Matmul:
    [1] 2-2-t(strict transpose);
    [2] 2-n-view+t(view transpose).
  False--Tensor is not transposed, proceed to format_contiguous.
*****************************************/
bool is_transpose_last_two_dims_flex(const Tensor& tensor) {
  if (c10::npu::NpuRunMode::IsGraphMode()) {
    return false;
  }
  if (tensor.dim() != 2) {
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

// Pick out strict-transpose tensors from flex-transpose tensors.
bool is_transpose_last_two_dims_strict(
    const Tensor& tensor,
    bool is_transpose_flex) {
  if (c10::npu::NpuRunMode::IsGraphMode()) {
    return false;
  }
  auto base_sizes = tensor.storage().get_npu_desc().base_sizes_;
  if (is_transpose_flex && base_sizes.size() == tensor.dim() &&
      tensor.size(-1) == base_sizes[tensor.dim() - 2] &&
      tensor.size(-2) == base_sizes[tensor.dim() - 1]) {
    return true;
  }
  return false;
}

// Refresh storage desc of view-transpose tensor.
void set_transposed_npu_desc(Tensor& tensor) {
  Tensor temp_transpose_Tensor = tensor.transpose(-1, -2);
  StorageDescHelper::SetDesc(
      tensor,
      temp_transpose_Tensor.sizes(),
      temp_transpose_Tensor.strides());
}

Tensor& mm_out_npu(Tensor& result, const Tensor& self, const Tensor& mat2) {
  Tensor contiguousResult = result.is_contiguous() ? result : result.contiguous();

  NPUStorageDesc self_desc = self.storage().get_npu_desc();
  NPUStorageDesc mat2_desc = mat2.storage().get_npu_desc();
  bool isSelfT_flex = is_transpose_last_two_dims_flex(self);
  bool isMat2T_flex = is_transpose_last_two_dims_flex(mat2);
  bool isSelfT_strict = is_transpose_last_two_dims_strict(self, isSelfT_flex);
  bool isMat2T_strict = is_transpose_last_two_dims_strict(mat2, isMat2T_flex);
  Tensor contiguousSelf = self;
  Tensor contiguousMat2 = mat2;

  if (isSelfT_flex) {
    if (!isSelfT_strict) {
      // Matmul cannot directly deal with view+transposed tensor with NZ format, so Transdata is necessary
      contiguousSelf = OpPreparation::CastBackToOriFormat(self);
      // Storage desc of view-transpose tensors should be refreshed to be matched.
      set_transposed_npu_desc(contiguousSelf);
    }
  } else {
    contiguousSelf = NpuUtils::format_contiguous_add_copy_optimize(self);
  }

  if (isMat2T_flex) {
    if (!isMat2T_strict) {
      // Matmul cannot directly deal with view+transposed tensor with NZ format, so Transdata is necessary
      contiguousMat2 = OpPreparation::CastBackToOriFormat(mat2);
      // Storage desc of view-transpose tensors should be refreshed to be matched.
      set_transposed_npu_desc(contiguousMat2);
    }
  } else {
    contiguousMat2 = NpuUtils::format_contiguous_add_copy_optimize(mat2);
  }

  auto func1 = [&contiguousSelf]() {
      bool pass = false;
      return std::tie(pass, contiguousSelf);
  };
  auto func2 = [&contiguousMat2]() {
      bool pass = false;
      return std::tie(pass, contiguousMat2);
  };
  
  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("MatMul")
      .InputWithFunc(func1)
      .InputWithFunc(func2)
      .Output(contiguousResult)
      .Attr("transpose_x1", isSelfT_flex)
      .Attr("transpose_x2", isMat2T_flex)
      .Run();

  // Recover storage desc of view-transpose tensors, i.e. the inverse process of
  // set_transposed_npu_desc
  if (isSelfT_flex && (!isSelfT_strict)) {
    self.storage().unsafeGetStorageImpl()->npu_desc_ = self_desc;
  }
  if (isMat2T_flex && (!isMat2T_strict)) {
    mat2.storage().unsafeGetStorageImpl()->npu_desc_ = mat2_desc;
  }

  if (!result.is_contiguous()) {
    result.copy_(contiguousResult);
  }
  return result;
}

Tensor mm_npu(const Tensor& self, const Tensor& mat2) {
  // calculate the output size
  auto outputSize = mm_npu_output_size(self, mat2); 
  
  // construct the output tensor of the NPU
  Tensor result;
  
  // TODO(ASCEND): 检查是否指定mm输出为NCHW。待NLP模型总体策略制定后删去
  if ((self.scalar_type() == ScalarType::Half) && !c10::npu::OptionsManager::CheckSwitchMMOutputEnable()) {
    // check is 16-algined with high-performance
    auto isAligin = [&]() {
      return (!(static_cast<uint64_t>(self.size(0)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(self.size(1)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(mat2.size(0)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(mat2.size(1)) & 0x0000000F));
    };
    // There is a data trampling problem in non-aligned scenes. For the time being, only aligned scenes are supported.
    if (env::CheckMmBmmNDEnable() && FormatHelper::IsBaseFormatType(self) &&
        FormatHelper::IsBaseFormatType(mat2) && isAligin()) {
      result = at::empty_with_format(outputSize, self.options());
    } else {
      result = at::empty_with_format(outputSize, self.options(), ACL_FORMAT_FRACTAL_NZ);
    }
  } else {
    result = at::empty_with_format(outputSize, self.options());
  }

  // calculate the output result of the NPU
  mm_out_npu(result, self, mat2);
  return result;
}

} // namespace native
} // namespace at
