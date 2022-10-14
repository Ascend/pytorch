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
    at::Tensor &sum_out_npu_no_dtype(
        at::Tensor &result,
        const at::Tensor &self,
        at::IntArrayRef dim,
        bool keepdim)
    {

      c10::SmallVector<int64_t, N> dimList;
      if (dim.empty())
      {
        dimList = CalcuOpUtil::get_dimlist_for_tensor(self);
      }
      else
      {
        dimList = c10::SmallVector<int64_t, N>(dim);
      }

      OpCommand cmd;
      cmd.Name("ReduceSum")
          .Input(self)
          .Input(dimList, at::kLong)
          .Output(result)
          .Attr("keep_dims", keepdim)
          .Run();
      return result;
    }

    at::Tensor &sum_out_npu_int_dtype(
        at::Tensor &result,
        const at::Tensor &self,
        at::IntArrayRef dim,
        bool keepdim,
        c10::ScalarType dtype)
    { 
      at::Tensor selfs = self;
      if(self.scalar_type() != c10::ScalarType::Float){
        selfs = NPUNativeFunctions::npu_dtype_cast(self, c10::ScalarType::Float);
      }    
      sum_out_npu_no_dtype(result, selfs, dim, keepdim);
      if(dtype == c10::ScalarType::Long){
        result = NPUNativeFunctions::npu_dtype_cast(result, c10::ScalarType::Long);
        return result;
      }
      result = NPUNativeFunctions::npu_dtype_cast(result, c10::ScalarType::Int);
      return result;
    }

    at::Tensor &sum_out_npu_nocheck(
        at::Tensor &result,
        const at::Tensor &self,
        at::IntArrayRef dim,
        bool keepdim,
        c10::optional<c10::ScalarType> dtype)
    {
      c10::ScalarType dstType;
      if (dtype.has_value())
      {
        if(dtype.value() == c10::ScalarType::Int || dtype.value() == c10::ScalarType::Long){
          return sum_out_npu_int_dtype(result, self, dim, keepdim, dtype.value());
        }
        else
        {
          dstType = dtype.value();
        }
      }
      else if (isIntegralType(self.scalar_type(), true))
      {
        return sum_out_npu_int_dtype(result, self, dim, keepdim, self.scalar_type());
      }
      else if (result.defined())
      {
        if (isIntegralType(result.scalar_type(), true))
        {
          return sum_out_npu_int_dtype(result, self, dim, keepdim, self.scalar_type());
        }
        else
        {
          dstType = result.scalar_type();
        }
      }
      else
      {
        dstType = self.scalar_type();
      }
      // dtype same
      if (dstType == self.scalar_type())
      {
        sum_out_npu_no_dtype(result, self, dim, keepdim);
        return result;
      }

      sum_out_npu_no_dtype(result, self.toType(dstType), dim, keepdim);
      return result;
    }

    at::Tensor &NPUNativeFunctions::sum_out(
        const at::Tensor &self,
        at::IntArrayRef dim,
        bool keepdim,
        c10::optional<c10::ScalarType> dtype,
        at::Tensor &result)
    {
      auto outputSize = sum_npu_output_size(self, dim, keepdim);
      auto dstType = self.scalar_type();
      if (dtype.has_value())
      {
        dstType = dtype.value();
      }

      OpPreparation::CheckOut(
          {self},
          result,
          ACL_FORMAT_ND,
          dstType,
          outputSize);

      OpPipeWithDefinedOut pipe;
      pipe.CheckMemory({self}, {result});

      sum_out_npu_nocheck(result, self, dim, keepdim, dtype);
      return result;
    }

    at::Tensor &NPUNativeFunctions::sum_out(
        const at::Tensor &self,
        at::DimnameList dim,
        bool keepdim,
        c10::optional<c10::ScalarType> dtype,
        at::Tensor &result)
    {
      return NPUNativeFunctions::sum_out(self,
                                         dimnames_to_positions(self, dim), keepdim, dtype, result);
    }

    at::Tensor NPUNativeFunctions::sum(
        const at::Tensor &self,
        at::IntArrayRef dim,
        bool keepdim,
        c10::optional<c10::ScalarType> dtype)
    {
      c10::ScalarType dstType;
      if (dtype.has_value())
      {
        if (isIntegralType(dtype.value(), true))
        {
          dstType = c10::ScalarType::Float;
        }
        else
        {
          dstType = dtype.value();
        }
      }
      else if (isIntegralType(self.scalar_type(), true))
      {
        dstType = c10::ScalarType::Float;
      }
      else
      {
        dstType = self.scalar_type();
      }

      // calculate the output size
      auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
      auto selfSize = self.sizes();

      for (int64_t i = 0; i < selfSize.size(); i++)
      {
        if (selfSize[i] == 0)
        {
          return at::zeros(outputSize, self.options());
        }
      }

      int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
      // scalar scene no support nz
      if (outputSize.empty() || outputSize.size() < 4)
      {
        npu_format = ACL_FORMAT_ND;
      }

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize, self.options().dtype(dstType), npu_format);

      // calculate the output result of the NPU
      sum_out_npu_nocheck(result, self, dim, keepdim, dtype);
      return result;
    }

    at::Tensor NPUNativeFunctions::sum(
        const at::Tensor &self,
        at::DimnameList dim,
        bool keepdim,
        c10::optional<c10::ScalarType> dtype)
    {
      return NPUNativeFunctions::sum(self, dimnames_to_positions(self, dim), keepdim, dtype);
    }

    at::Tensor NPUNativeFunctions::sum(const at::Tensor &self, c10::optional<c10::ScalarType> dtype)
    {
      return NPUNativeFunctions::sum(self, c10::SmallVector<int64_t, N>{}, false, dtype);
    }

  } // namespace native
} // namespace at_npu
