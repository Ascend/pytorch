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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    at::Tensor NPUNativeFunctions::max_pool2d(
        const at::Tensor &self,
        at::IntArrayRef kernel_size,
        at::IntArrayRef stride,
        at::IntArrayRef padding,
        at::IntArrayRef dilation,
        bool ceil_mode)
    {
      auto output_and_indices = at::max_pool2d_with_indices(
          self, kernel_size, stride, padding, dilation, ceil_mode);
      return std::get<0>(output_and_indices);
    }

  } // namespace native
} // namespace at_npu