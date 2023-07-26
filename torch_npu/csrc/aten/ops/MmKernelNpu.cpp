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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

/*****************************************
Function: is_transpose_last_two_dims_flex
Description:
  Flexible transpose judgement for view+transpose+Matmul, i.e.,
  tensors with dim=2 and base_size_.size=n can also be Matmul directly!
Return:
  True--Cases are flex transposed(flex transpose=strict transpose+view
    transpose), which can be refreshed as a input transposed tensor proceed to
Matmul: [1] 2-2-t(strict transpose); [2] 2-n-view+t(view transpose).
  False--Tensor is not transposed, proceed to format_contiguous.
*****************************************/
bool is_transpose_last_two_dims_flex(const at::Tensor &tensor) {
  if (c10_npu::NpuRunMode::IsGraphMode()) {
    return false;
  }
  if (tensor.dim() != 2) {
    return false;
  }

  int64_t dim1 = tensor.dim() - 1;
  int64_t dim2 = tensor.dim() - 2;

  if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2)) {
    return true;
  } else {
    return false;
  }
}


// Pick out strict-transpose tensors from flex-transpose tensors.
bool is_transpose_last_two_dims_strict(const at::Tensor &tensor, bool is_transpose_flex) {
  if (c10_npu::NpuRunMode::IsGraphMode()) {
    return false;
  }
  auto base_sizes = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->get_npu_desc().base_sizes_;
  if (is_transpose_flex && base_sizes.size() == tensor.dim() && tensor.size(-1) == base_sizes[tensor.dim() - 2] &&
      tensor.size(-2) == base_sizes[tensor.dim() - 1]) {
    return true;
  }
  return false;
}

// Refresh storage desc of view-transpose tensor.
void set_transposed_npu_desc(at::Tensor &tensor) {
  at::Tensor temp_transpose_Tensor = tensor.transpose(-1, -2);
  StorageDescHelper::SetDesc(tensor, temp_transpose_Tensor.sizes(), temp_transpose_Tensor.strides());
}

void mm_insert_input_transpose(at::Tensor &tensor, bool &is_tensor_trans_flex, bool &is_tensor_trans_strict) {
  tensor = is_tensor_trans_flex ? tensor.clone() : tensor.transpose(-1, -2).clone();
  is_tensor_trans_flex = !is_tensor_trans_flex;
  is_tensor_trans_strict = !is_tensor_trans_strict;
}

void mm_set_format_contiguous(at::Tensor &tensor, bool &is_tensor_trans_flex, bool &is_tensor_trans_strict) {
  if (is_tensor_trans_flex) {
    if (!is_tensor_trans_strict) {
      // Matmul cannot directly deal with view+transposed tensor with NZ format,
      // so Transdata is necessary
      tensor = OpPreparation::CastBackToOriFormat(tensor);
      // Storage desc of view-transpose tensors should be refreshed to be
      // matched.
      set_transposed_npu_desc(tensor);
    }
  } else {
    tensor = NpuUtils::format_contiguous_add_copy_optimize(tensor);
  }
}

bool mm_check_split_k(const at::Tensor &self, const at::Tensor &mat2, bool &is_support_nd_out) {
  if (!is_support_nd_out || !(self.dtype() == at::ScalarType::Half && mat2.dtype() == at::ScalarType::Half) ||
      !(FormatHelper::GetFormat(self) == ACL_FORMAT_ND && FormatHelper::GetFormat(mat2) == ACL_FORMAT_ND)) {
    return false;
  }
  // split_k rule, maybe modified afterwards
  const static int64_t kSplitKTimes = 8;
  return self.size(1) >= kSplitKTimes * std::max(self.size(0), mat2.size(1));
}

bool mm_check_nd_to_nz_on_the_fly(const at::Tensor &self, const at::Tensor &mat2) {
  const static int64_t kInnerAxisMinBytes = 256;
  const static int64_t kInnerAxisMaxLimit = 65535;
  int64_t self_inner_axis = self.size(self.dim() - 1);
  int64_t self_outer_axis = self.size(self.dim() - 2);
  int64_t mat2_inner_axis = mat2.size(mat2.dim() - 1);
  int64_t mat2_outer_axis = mat2.size(mat2.dim() - 2);
  if (CalcuOpUtil::IsMmTranspose(self)) {
    self_inner_axis = self.size(self.dim() - 2);
    self_outer_axis = self.size(self.dim() - 1);
  }
  if (CalcuOpUtil::IsMmTranspose(mat2)) {
    mat2_inner_axis = mat2.size(mat2.dim() - 2);
    mat2_outer_axis = mat2.size(mat2.dim() - 1);
  }
  int64_t data_type = elementSize(self.scalar_type());
  if (self_outer_axis > kInnerAxisMaxLimit && self_inner_axis * data_type < kInnerAxisMinBytes &&
      bool((self_inner_axis * data_type) & 0x1F)) {
    return false;
  }
  return !(self_inner_axis > kInnerAxisMaxLimit && self_outer_axis > kInnerAxisMaxLimit ||
           mat2_inner_axis > kInnerAxisMaxLimit && mat2_outer_axis > kInnerAxisMaxLimit);
}

at::Tensor &NPUNativeFunctions::mm_out(const at::Tensor &self, const at::Tensor &mat2, at::Tensor &result) {
  if (self.numel() == 0 || mat2.numel() == 0) {
    return result.zero_();
  }

  at::Tensor contiguous_result = result.is_contiguous() ? result : result.contiguous();

  const auto &self_desc = torch_npu::NPUBridge::GetNpuStorageImplDesc(self);
  const auto &mat2_desc = torch_npu::NPUBridge::GetNpuStorageImplDesc(mat2);
  bool is_self_trans_flex = is_transpose_last_two_dims_flex(self);
  bool is_mat2_trans_flex = is_transpose_last_two_dims_flex(mat2);
  bool is_self_trans_strict = is_transpose_last_two_dims_strict(self, is_self_trans_flex);
  bool is_mat2_trans_strict = is_transpose_last_two_dims_strict(mat2, is_mat2_trans_flex);
  at::Tensor contiguous_self = self;
  at::Tensor contiguous_mat2 = mat2;

  bool is_transpose_self = CalcuOpUtil::IsTransposeInnerAxis(contiguous_self);
  bool is_transpose_mat2 = CalcuOpUtil::IsTransposeInnerAxis(contiguous_mat2);
  if (is_transpose_self && is_transpose_mat2 &&
      !CalcuOpUtil::IsTransposeBothInnerAxis(contiguous_self, contiguous_mat2)) {
    is_transpose_self = !is_transpose_self;
    is_transpose_mat2 = !is_transpose_mat2;
  }

  int64_t m_dim = self.size(-2);
  int64_t k_dim = self.size(-1);
  int64_t n_dim = mat2.size(-1);
  int64_t data_size = elementSize(self.scalar_type());
  // 512B aligned shape is soc friendly
  const int64_t kPackage512 = 512;
  // 128 unaligned inner axis performs bad
  const int64_t kInnerDimAlignment = 128;
  // k_dim less than 512 is skipped
  const int64_t kMinKDim = 2048;
  // m/n should be less than 16384 to gain perf improvement
  const int64_t kMaxInnerDim = 16384;
  bool common_rule = k_dim > kMinKDim && ((k_dim * data_size) % kPackage512 == 0);
  common_rule &= c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 && CalcuOpUtil::IsHalfFloatDtype(self);
  bool self_cache_opti = is_self_trans_flex && (m_dim % kInnerDimAlignment != 0) && m_dim < kMaxInnerDim;
  if (is_transpose_self || (self_cache_opti && common_rule)) {
    mm_insert_input_transpose(contiguous_self, is_self_trans_flex, is_self_trans_strict);
  }
  bool mat2_cache_opti = !is_mat2_trans_flex && (n_dim % kInnerDimAlignment != 0) && n_dim < kMaxInnerDim;
  if (is_transpose_mat2 || (mat2_cache_opti && common_rule)) {
    mm_insert_input_transpose(contiguous_mat2, is_mat2_trans_flex, is_mat2_trans_strict);
  }
  if (!is_transpose_self && !is_transpose_mat2) {
    CalcuOpUtil::InsertInputPad(contiguous_self, contiguous_mat2);
  }

  mm_set_format_contiguous(contiguous_self, is_self_trans_flex, is_self_trans_strict);
  mm_set_format_contiguous(contiguous_mat2, is_mat2_trans_flex, is_mat2_trans_strict);

  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("MatMul")
      .InputWithoutContiguous(contiguous_self)
      .InputWithoutContiguous(contiguous_mat2)
      .Output(contiguous_result)
      .Attr("transpose_x1", is_self_trans_flex)
      .Attr("transpose_x2", is_mat2_trans_flex)
      .Run();

  // Recover storage desc of view-transpose tensors, i.e. the inverse process of
  // set_transposed_npu_desc
  if (is_self_trans_flex && (!is_self_trans_strict)) {
    torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_ = self_desc;
  }
  if (is_mat2_trans_flex && (!is_mat2_trans_strict)) {
    torch_npu::NPUBridge::GetNpuStorageImpl(mat2)->npu_desc_ = mat2_desc;
  }

  if (!result.is_contiguous()) {
    result.copy_(contiguous_result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::mm(const at::Tensor &self, const at::Tensor &mat2) {
  // calculate the output size
  auto output_size = {self.size(0), mat2.size(1)};
  // construct the output tensor of the NPU
  at::Tensor result;
  bool need_nd_out = false;
  static bool is_support_nd_out = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1;
  bool split_k = mm_check_split_k(self, mat2, is_support_nd_out);
  // check format_out of mm is NCHW. Delate after definite NLP model.
  if ((self.scalar_type() == at::ScalarType::Half)) {
    // check is 16-algined with high-performance
    auto is_aligin = [&]() {
      return (!(static_cast<uint64_t>(self.size(0)) & 0xF)) && (!(static_cast<uint64_t>(self.size(1)) & 0xF)) &&
             (!(static_cast<uint64_t>(mat2.size(0)) & 0xF)) && (!(static_cast<uint64_t>(mat2.size(1)) & 0xF));
    };
    // There is a data trampling problem in non-aligned scenes. For the time
    // being, only aligned scenes are supported.
    static auto mm_bmm_nd = !env::CheckMmBmmNDDisable();
    if (FormatHelper::IsBaseFormatType(self) && FormatHelper::IsBaseFormatType(mat2) && mm_bmm_nd &&
        ((is_support_nd_out && mm_check_nd_to_nz_on_the_fly(self, mat2)) || (!is_support_nd_out && is_aligin()))) {
      if (split_k) {
        result = OpPreparation::ApplyTensorWithFormat(output_size, self.options().dtype(at::ScalarType::Float),
                                                      ACL_FORMAT_ND);
      } else {
        result = OpPreparation::ApplyTensorWithFormat(output_size, self.options(), ACL_FORMAT_ND);
      }
    } else {
      need_nd_out = mm_bmm_nd;
      if (split_k) {
        result = OpPreparation::ApplyTensorWithFormat(output_size, self.options().dtype(at::ScalarType::Float),
                                                      ACL_FORMAT_FRACTAL_NZ, true);
      } else {
        result = OpPreparation::ApplyTensorWithFormat(output_size, self.options(), ACL_FORMAT_FRACTAL_NZ, true);
      }
    }
  } else {
    result = OpPreparation::ApplyTensorWithFormat(output_size, self.options(), ACL_FORMAT_ND);
  }
  // calculate the output result of the NPU
  NPUNativeFunctions::mm_out(self, mat2, result);
  if (need_nd_out) {
    result = NPUNativeFunctions::npu_format_cast(result, ACL_FORMAT_ND);
  }
  result = split_k ? NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Half) : result;
  return result;
}

} // namespace native
} // namespace at_npu