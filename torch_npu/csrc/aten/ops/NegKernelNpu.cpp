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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    at::Tensor &neg_out_npu_nocheck(at::Tensor &result, const at::Tensor &self)
    {
      OpCommand cmd;
      cmd.Name("Neg")
          .Input(self)
          .Output(result)
          .Run();

      return result;
    }

    at::Tensor &NPUNativeFunctions::neg_out(const at::Tensor &self, at::Tensor &result)
    {
      OpPreparation::CheckOut(
          {self},
          result,
          ACL_FORMAT_ND,
          self.scalar_type(),
          self.sizes());
      neg_out_npu_nocheck(result, self);

      return result;
    }

    at::Tensor NPUNativeFunctions::neg(const at::Tensor &self)
    {
      // construct the output tensor of the NPU
      at::Tensor result = at::empty_with_format(
          self.sizes(), self.options(), CalcuOpUtil::get_tensor_npu_format(self));

      // calculate the output result of the NPU
      neg_out_npu_nocheck(result, self);

      return result;
    }

    at::Tensor &NPUNativeFunctions::neg_(at::Tensor &self)
    {
      c10::SmallVector<at::Tensor, N> inputs = {self};
      c10::SmallVector<at::Tensor, N> outputs = {self};
      CalcuOpUtil::check_memory_over_laps(inputs, outputs);

      if (!NpuUtils::check_match(&self))
      {
        at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
        at::Tensor result = neg_out_npu_nocheck(contiguousSelf, contiguousSelf);
        NpuUtils::format_fresh_view(self, result);
      }
      else
      {
        neg_out_npu_nocheck(self, self);
      }

      return self;
    }

  } // namespace native
} // namespace at_npu
