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
    at::Tensor &NPUNativeFunctions::mv_out(const at::Tensor &self, const at::Tensor &vec, at::Tensor &result)
    {
      OpPreparation::CheckOut(
          {self},
          result,
          CalcuOpUtil::GetTensorNpuFormat(self),
          self.scalar_type(),
          {self.size(0)});

      at::Tensor vec_2d = at::unsqueeze(vec, 1);
      at::Tensor mm_out = NPUNativeFunctions::mm(self, vec_2d);
      mm_out = at::squeeze(mm_out, 1);
      result.copy_(mm_out);
      return result;
    }

    at::Tensor NPUNativeFunctions::mv(const at::Tensor &self, const at::Tensor &vec)
    {
      at::Tensor vec_2d = at::unsqueeze(vec, 1);
      at::Tensor result = NPUNativeFunctions::mm(self, vec_2d);
      result = at::squeeze(result, 1);
      return result;
    }

  } // namespace native
} // namespace at_npu
