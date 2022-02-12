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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpTemplate.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {
    at::Tensor NPUNativeFunctions::ones_like(const at::Tensor &self,
                                             c10::optional<c10::ScalarType> dtype_opt,
                                             c10::optional<c10::Layout> layout_opt,
                                             c10::optional<c10::Device> device_opt,
                                             c10::optional<bool> pin_memory_opt,
                                             c10::optional<c10::MemoryFormat> optional_memory_format)
    {
      auto device = device_or_default(device_opt);
      if (!device.is_npu())
      {
        auto result = at::empty_like(self,
                                     dtype_opt,
                                     layout_opt,
                                     device_opt,
                                     pin_memory_opt,
                                     optional_memory_format);

        return result.fill_(1.);
      }

      auto outputSize = input_same_output_size(self);
      // construct the output tensor of the NPU
      at::Tensor result = NPUNativeFunctions::empty_with_format(
        outputSize, dtype_opt, layout_opt, device_opt, pin_memory_opt,
        CalcuOpUtil::get_tensor_npu_format(self));
      // calculate the output result of the NPUc
      return NPUNativeFunctions::one_(result);
    }

    at::Tensor &NPUNativeFunctions::one_(at::Tensor &self)
    {
      if (!NpuUtils::check_match(&self))
      {
        at::Tensor selfContiguous = NpuUtils::format_contiguous(self);
        OpCommand cmd;
        cmd.Name("OnesLike").Input(selfContiguous).Output(selfContiguous).Run();
        NpuUtils::format_fresh_view(self, selfContiguous);
      }
      else
      {
        OpCommand cmd;
        cmd.Name("OnesLike").Input(self).Output(self).Run();
      }

      return self;
    }

  } // namespace native
} // namespace at_npu
