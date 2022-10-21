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

    at::Tensor &mean_out_npu_no_dtype_nocheck(
        at::Tensor &result,
        const at::Tensor &self,
        at::IntArrayRef dim,
        bool keepdim)
    {

      if (self.numel() == 0 && dim.size() == 0)
      {
        // In this scenario, needs to return nan. And the nan of the NPU can only be fp32.
        result = NPUNativeFunctions::npu_dtype_cast(result, at::kFloat).fill_(0);
        result = result / 0;
        return result;
      }

      c10::SmallVector<int64_t, N> dimVec;
      if (dim.empty())
      {
        dimVec = CalcuOpUtil::get_dimlist_for_tensor(self);
      }
      else
      {
        dimVec = array_to_small_vector(dim);
      }

      OpCommand cmd;
      cmd.Name("ReduceMean")
          .Input(self)
          .Input(dimVec, at::kLong)
          .Output(result)
          .Attr("keep_dims", keepdim)
          .Run();
      return result;
    }

    at::Tensor &mean_out_npu_no_dtype(
        at::Tensor &result,
        const at::Tensor &self,
        at::IntArrayRef dim,
        bool keepdim)
    {
      auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
      int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(result);
      if (outputSize.empty())
      {
        npu_format = ACL_FORMAT_NCHW;
      }
      OpPreparation::CheckOut(
          {self},
          result,
          npu_format,
          self.scalar_type(),
          outputSize);

      mean_out_npu_no_dtype_nocheck(result, self, dim, keepdim);
      return result;
    }

    at::Tensor &NPUNativeFunctions::mean_out(
        const at::Tensor &self,
        at::IntArrayRef dim,
        bool keepdim,
        c10::optional<c10::ScalarType> dtype,
        at::Tensor &result)
    {
      c10::ScalarType dstType;
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

      // dtype same
      if (dstType == self.scalar_type())
      {
        mean_out_npu_no_dtype(result, self, dim, keepdim);
        return result;
      }

      mean_out_npu_no_dtype(result, self.toType(dstType), dim, keepdim);
      return result;
    }

    at::Tensor &NPUNativeFunctions::mean_out(
        const at::Tensor &self,
        at::DimnameList dim,
        bool keepdim,
        c10::optional<c10::ScalarType> dtype,
        at::Tensor &result)
    {
      return NPUNativeFunctions::mean_out(self, dimnames_to_positions(self, dim), keepdim, dtype, result);
    }

    at::Tensor NPUNativeFunctions::mean(
        const at::Tensor &self,
        at::IntArrayRef dim,
        bool keepdim,
        c10::optional<c10::ScalarType> dtype)
    {
      c10::ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();

      // calculate the output size
      auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);

      int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
      // scalar scene no support nz
      if (outputSize.empty())
      {
        npu_format = ACL_FORMAT_NCHW;
      }

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize, self.options().dtype(dstType), npu_format);

      // calculate the output result of the NPU
      NPUNativeFunctions::mean_out(self, dim, keepdim, dtype, result);
      return result;
    }

    at::Tensor NPUNativeFunctions::mean(
        const at::Tensor &self,
        at::DimnameList dim,
        bool keepdim,
        c10::optional<c10::ScalarType> dtype)
    {
      return NPUNativeFunctions::mean(self, dimnames_to_positions(self, dim), keepdim, dtype);
    }

    at::Tensor NPUNativeFunctions::mean(const at::Tensor &self, c10::optional<c10::ScalarType> dtype)
    {
      return NPUNativeFunctions::mean(self, c10::SmallVector<int64_t, N>{}, false, dtype);
    }

  } // namespace native
} // namespace at_npu