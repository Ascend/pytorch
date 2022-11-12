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

    at::Tensor &zeros_like_out_npu(const at::Tensor &self, at::Tensor &result)
    {
      OpCommand cmd;
      cmd.Name("ZerosLike")
          .Input(self)
          .Output(result)
          .Run();

      return result;
    }

    at::Tensor NPUNativeFunctions::zeros_like(
        const at::Tensor &self,
        c10::optional<c10::ScalarType> dtype_opt,
        c10::optional<c10::Layout> layout_opt,
        c10::optional<c10::Device> device_opt,
        c10::optional<bool> pin_memory_opt,
        c10::optional<c10::MemoryFormat> optional_memory_format)
    {
      auto device = device_or_default(device_opt);
      if (!(device.type() == at_npu::key::NativeDeviceType))
      {
        auto result = at::empty_like(self,
                                     dtype_opt,
                                     layout_opt,
                                     device_opt,
                                     pin_memory_opt,
                                     optional_memory_format);
        return result.fill_(0);
      }

      // construct the output tensor of the NPU
      c10::TensorOptions option = c10::TensorOptions().dtype(dtype_opt)
                                                      .device(device_opt)
                                                      .layout(layout_opt)
                                                      .pinned_memory(pin_memory_opt);
      at::Tensor result = OpPreparation::ApplyTensor(self, option);
      // calculate the output result of the NPU
      return result.zero_();
    }

    at::Tensor &NPUNativeFunctions::zero_(at::Tensor &self)
    {
      if (!NpuUtils::check_match(&self))
      {
        at::Tensor selfContiguous = NpuUtils::format_contiguous(self);
        at::Tensor result = zeros_like_out_npu(selfContiguous, selfContiguous);
        NpuUtils::format_fresh_view(self, result);
      }
      else
      {
        zeros_like_out_npu(self, self);
      }

      return self;
    }

  } // namespace native
} // namespace at_npu
