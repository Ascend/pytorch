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

#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& bmm_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& mat2) {
  bool is_self_t = CalcuOpUtil::IsTransposeLastTwoDims(self);
  bool is_mat2_t = CalcuOpUtil::IsTransposeLastTwoDims(mat2);

  at::Tensor contiguous_self = is_self_t ? self : NpuUtils::format_contiguous(self);
  at::Tensor contiguous_mat2 = is_mat2_t ? mat2 : NpuUtils::format_contiguous(mat2);

  OpCommand cmd;
  cmd.Name("BatchMatMul")
      .InputWithoutContiguous(contiguous_self)
      .InputWithoutContiguous(contiguous_mat2)
      .Output(result)
      .Attr("adj_x1", is_self_t)
      .Attr("adj_x2", is_mat2_t)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::bmm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& result) {
  TORCH_CHECK(self.device() == mat2.device(), 
              "Expected all tensors to be on the same device, but found at least two devices, ", 
              (self.device().type() == at_npu::key::NativeDeviceType ? "npu" : "cpu"),
              " and ", 
              (mat2.device().type() == at_npu::key::NativeDeviceType ? "npu! " : "cpu! "));
  auto output_size = {self.size(0), self.size(1), mat2.size(2)};

  OpPreparation::CheckOut(
      {self, mat2},
      result,
      CalcuOpUtil::GetTensorNpuFormat(result),
      self.scalar_type(),
      output_size);

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    bmm_out_npu_nocheck(contiguous_result, self, mat2);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    bmm_out_npu_nocheck(result, self, mat2);
  }
  return result;
}

at::Tensor NPUNativeFunctions::bmm(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(self.device() == mat2.device(), 
              "Expected all tensors to be on the same device, but found at least two devices, ", 
              (self.device().type() == at_npu::key::NativeDeviceType ? "npu" : "cpu"),
              " and ", 
              (mat2.device().type() == at_npu::key::NativeDeviceType ? "npu! " : "cpu! "));
  auto output_size = {self.size(0), self.size(1), mat2.size(2)};

  at::Tensor result;
  bool need_nd_out = false;
  // Check if the mm output is specified as NCHW.
  // It will be deleted after the overall strategy of the NLP model is formulated.
  if ((self.scalar_type() == at::ScalarType::Half)) {
    // check is 16-algined with high-performance
    auto is_aligin = [&]() {
      return (!(static_cast<uint64_t>(self.size(1)) & 0xF)) && (!(static_cast<uint64_t>(self.size(2)) & 0xF)) &&
             (!(static_cast<uint64_t>(mat2.size(1)) & 0xF)) && (!(static_cast<uint64_t>(mat2.size(2)) & 0xF));
    };
    static auto mm_bmm_nd = !env::CheckMmBmmNDDisable();
    static bool is_support_nd_out = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1;
    // There is a data trampling problem in non-aligned scenes. For the time being, only aligned scenes are supported.
    if (FormatHelper::IsBaseFormatType(self) && FormatHelper::IsBaseFormatType(mat2) &&
        mm_bmm_nd && ((is_support_nd_out && CalcuOpUtil::IsNdToNzOnTheFly(self, mat2)) ||
        (!is_support_nd_out && is_aligin()))) {
      result = OpPreparation::ApplyTensorWithFormat(output_size, self.options(), ACL_FORMAT_ND);
    } else {
      result = OpPreparation::ApplyTensorWithFormat(output_size, self.options(), ACL_FORMAT_FRACTAL_NZ, true);
      need_nd_out = mm_bmm_nd;
    }
  } else {
    result = OpPreparation::ApplyTensorWithFormat(output_size, self.options(), ACL_FORMAT_ND);
  }

  bmm_out_npu_nocheck(result, self, mat2);
  if (need_nd_out) {
    result = NPUNativeFunctions::npu_format_cast(result, ACL_FORMAT_ND);
  }
  return result;
}

} // namespace native
} // namespace at
