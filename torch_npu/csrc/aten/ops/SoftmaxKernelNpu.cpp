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

    at::Tensor NPUNativeFunctions::softmax(
        const at::Tensor &self,
        int64_t dim,
        c10::optional<at::ScalarType> dtype)
    {
      auto result = [&]()
      {
        at::NoNamesGuard guard;
        at::Tensor converted = dtype.has_value() ? NPUNativeFunctions::npu_dtype_cast(self, dtype.value()) : self;
        return at::_softmax(converted, dim, false);
      }();
      at::namedinference::propagate_names(result, self);

      return result;
    }

    at::Tensor NPUNativeFunctions::softmax(
        const at::Tensor &self,
        at::Dimname dim,
        c10::optional<at::ScalarType> dtype)
    {
      return NPUNativeFunctions::softmax(self, dimname_to_position(self, dim), dtype);
    }

    at::Tensor NPUNativeFunctions::_softmax(const at::Tensor &self, int64_t dim, bool half_to_float)
    {

      // construct the output tensor of the NPU
      at::Tensor result;
      if (half_to_float)
      {
        result = OpPreparation::ApplyTensor(self, self.options().dtype(at::ScalarType::Float));
      }
      else
      {
        result = OpPreparation::ApplyTensor(self);
      }

      // calculate the output result of the NPU
      c10::optional<at::ScalarType> dtype = result.scalar_type();
      at::ScalarType dstType;
      if (dtype.has_value())
      {
        dstType = dtype.value();
      }
      else if (result.defined())
      {
        dstType = result.scalar_type();
      }
      else
      {
        dstType = self.scalar_type();
      }
      at::Tensor converted =
          dstType == self.scalar_type() ? self : NPUNativeFunctions::npu_dtype_cast(self, dstType);

      c10::SmallVector<int64_t, N> dimList = {dim};
      OpCommand cmd;
      cmd.Name("SoftmaxV2")
          .Input(converted)
          .Output(result)
          .Attr("axes", dimList)
          .Run();

      return result;
    }

  } // namespace native
} // namespace at_npu
