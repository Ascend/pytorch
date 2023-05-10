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

#include <ATen/record_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {
    at::Tensor &NPUNativeFunctions::npu_transpose_out(
        const at::Tensor &self,
        at::IntArrayRef perm,
        bool require_contiguous,
        at::Tensor &result
        )
    {
      OpCommand cmd;
      if (require_contiguous) {
        // Any tensor-view(discontiguous) Input Tensor from users should be transformed to be contiguous here.
        cmd.Name("Transpose")
          .Input(self)
          .Input(perm)
          .Output(result)
          .Run();
      } else {
      // For permute-opt in trans-contiguous, it accepts transposed(discontiguous) Input Tensor.
      cmd.Name("Transpose")
          .InputWithoutContiguous(self)
          .Input(perm)
          .Output(result)
          .Run();
      }
      return result;
    }

    at::Tensor NPUNativeFunctions::npu_transpose(const at::Tensor &self, at::IntArrayRef perm, bool require_contiguous)
    {
      auto outputSize = transpose_npu_output_size(self, perm);
      at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
      NPUNativeFunctions::npu_transpose_out(self, perm, require_contiguous, result);

      return result;
    }


  } // namespace native
} // namespace at_npu