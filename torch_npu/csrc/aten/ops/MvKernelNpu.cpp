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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    at::Tensor &mv_out_npu_nocheck(const at::Tensor &self, const at::Tensor &vec, at::Tensor &result)
    {
      bool isSelfT = CalcuOpUtil::IsTransposeLastTwoDims(self);
      at::Tensor contiguousSelf;
      contiguousSelf = isSelfT ? self : NpuUtils::format_contiguous(self);
      at::Tensor vecT = at::unsqueeze(vec, 1);

      OpCommand cmd;
      cmd.Name("MatMul")
          .InputWithoutContiguous(contiguousSelf)
          .Input(vecT)
          .Attr("transpose_x1", isSelfT)
          .Attr("transpose_x2", false)
          .Output(result)
          .Run();

      npu_fast_reshape_(result);
      return result;
    }

    at::Tensor &NPUNativeFunctions::mv_out(const at::Tensor &self, const at::Tensor &vec, at::Tensor &result)
    {
      OpPreparation::CheckOut(
          {self},
          result,
          CalcuOpUtil::GetTensorNpuFormat(self),
          self.scalar_type(),
          {self.size(0)});

      result.unsqueeze_(1);
      if (!NpuUtils::check_match(&result)) {
        at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
        mv_out_npu_nocheck(self, vec, contiguousResult);
        NpuUtils::format_fresh_view(result, contiguousResult);
      } else {
        mv_out_npu_nocheck(self, vec, result);
      }
      result.squeeze_(1);
      return result;
    }

    at::Tensor NPUNativeFunctions::mv(const at::Tensor &self, const at::Tensor &vec)
    {

      at::Tensor result = OpPreparation::ApplyTensor(self, {self.size(0), 1});

      mv_out_npu_nocheck(self, vec, result);
      result.squeeze_(1);

      return result;
    }

  } // namespace native
} // namespace at_npu
