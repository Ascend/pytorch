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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"

namespace at_npu
{
  namespace native
  {

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
    bool is_transpose_last_two_dims_flex(const at::Tensor &tensor)
    {
      if (tensor.dim() != 2)
      {
        return false;
      }
      int64_t numel = 1;
      auto storageSize = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->get_npu_desc().storage_sizes_;

      for (int i = 0; i < storageSize.size(); i++)
      {
        numel *= storageSize[i];
      }

      int64_t dim1 = tensor.dim() - 1;
      int64_t dim2 = tensor.dim() - 2;

      if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2) &&
          tensor.numel() == numel)
      {
        return true;
      }
      else
      {
        return false;
      }
    }

    // Pick out strict-transpose tensors from flex-transpose tensors.
    bool is_transpose_last_two_dims_strict(
        const at::Tensor &tensor,
        bool is_transpose_flex)
    {
      auto base_sizes = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->get_npu_desc().base_sizes_;
      if (is_transpose_flex && base_sizes.size() == tensor.dim() &&
          tensor.size(-1) == base_sizes[tensor.dim() - 2] &&
          tensor.size(-2) == base_sizes[tensor.dim() - 1])
      {
        return true;
      }
      return false;
    }

    // Refresh storage desc of view-transpose tensor.
    void set_transposed_npu_desc(at::Tensor &tensor)
    {
      at::Tensor temp_transpose_Tensor = tensor.transpose(-1, -2);
      StorageDescHelper::SetDesc(
          tensor,
          temp_transpose_Tensor.sizes(),
          temp_transpose_Tensor.strides());
    }

    at::Tensor &NPUNativeFunctions::mm_out(const at::Tensor &self, const at::Tensor &mat2, at::Tensor &result)
    {
      at::Tensor contiguousResult = result.is_contiguous() ? result : result.contiguous();

      auto self_desc = torch_npu::NPUBridge::GetNpuStorageImpl(self)->get_npu_desc();
      auto mat2_desc = torch_npu::NPUBridge::GetNpuStorageImpl(mat2)->get_npu_desc();
      bool isSelfT_flex = is_transpose_last_two_dims_flex(self);
      bool isMat2T_flex = is_transpose_last_two_dims_flex(mat2);
      bool isSelfT_strict = is_transpose_last_two_dims_strict(self, isSelfT_flex);
      bool isMat2T_strict = is_transpose_last_two_dims_strict(mat2, isMat2T_flex);
      at::Tensor contiguousSelf = self;
      at::Tensor contiguousMat2 = mat2;

      if (isSelfT_flex)
      {
        if (!isSelfT_strict)
        {
          // Matmul cannot directly deal with view+transposed tensor with NZ format, so Transdata is necessary
          contiguousSelf = OpPreparation::CastBackToOriFormat(self);
          // Storage desc of view-transpose tensors should be refreshed to be matched.
          set_transposed_npu_desc(contiguousSelf);
        }
      }
      else
      {
        contiguousSelf = NpuUtils::format_contiguous_add_copy_optimize(self);
      }

      if (isMat2T_flex)
      {
        if (!isMat2T_strict)
        {
          // Matmul cannot directly deal with view+transposed tensor with NZ format, so Transdata is necessary
          contiguousMat2 = OpPreparation::CastBackToOriFormat(mat2);
          // Storage desc of view-transpose tensors should be refreshed to be matched.
          set_transposed_npu_desc(contiguousMat2);
        }
      }
      else
      {
        contiguousMat2 = NpuUtils::format_contiguous_add_copy_optimize(mat2);
      }

      auto func1 = [&contiguousSelf]()
      {
        bool pass = false;
        return std::tie(pass, contiguousSelf);
      };
      auto func2 = [&contiguousMat2]()
      {
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
      if (isSelfT_flex && (!isSelfT_strict))
      {
        torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_ = self_desc;
      }
      if (isMat2T_flex && (!isMat2T_strict))
      {
        torch_npu::NPUBridge::GetNpuStorageImpl(mat2)->npu_desc_ = mat2_desc;
      }

      if (!result.is_contiguous())
      {
        result.copy_(contiguousResult);
      }
      return result;
    }

    at::Tensor NPUNativeFunctions::mm(const at::Tensor &self, const at::Tensor &mat2)
    {
      // calculate the output size
      auto outputSize = mm_npu_output_size(self, mat2);

      // construct the output tensor of the NPU
      at::Tensor result;

      if ((self.scalar_type() == at::ScalarType::Half) && !torch_npu::option::OptionsManager::CheckSwitchMMOutputEnable())
      {
        result = OpPreparation::ApplyTensorWithFormat(
            outputSize, self.options(), ACL_FORMAT_FRACTAL_NZ);
      }
      else
      {
        result = OpPreparation::ApplyTensorWithSizes(outputSize, self.options());
      }

      // calculate the output result of the NPU
      NPUNativeFunctions::mm_out(self, mat2, result);
      return result;
    }

  } // namespace native
} // namespace at_npu
