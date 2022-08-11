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
#include "torch_npu/csrc/aten/XLANativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    c10::SmallVector<at::Tensor, N> cat_dest_tensor_list(at::TensorList tensors)
    {
      c10::SmallVector<at::Tensor, N> dstTensorList;
      // pytorch supports empty tensors, which needs to be removed from the NPU.
      for (at::Tensor tensor : tensors)
      {
        if (tensor.dim() == 1 && tensor.sizes()[0] == 0)
        {
          continue;
        }

        dstTensorList.emplace_back(tensor);
      }

      return dstTensorList;
    }

    c10::SmallVector<int64_t, SIZE> cat_npu_output_size(
        c10::SmallVector<at::Tensor, N_SIZE> &tensors,
        int64_t dimension)
    {
      bool allSkipped = true;
      int64_t nDims = 0;
      at::Tensor *notSkippedTensor;
      int numInputs = tensors.size();
      auto should_skip = [](const at::Tensor *t)
      {
        return t->nbytes() == 0 && t->dim() == 1;
      };
      for (int i = 0; i < numInputs; i++)
      {
        if (should_skip((at::Tensor *)&tensors[i]))
        {
          continue;
        }
        // found a non-empty tensor
        allSkipped = false;
        notSkippedTensor = (at::Tensor *)&tensors[i];
        nDims = notSkippedTensor->dim();
        break;
      }

      if (allSkipped)
      {
        c10::SmallVector<int64_t, SIZE> size = {0};
        return size;
      }

      // Compute size of the result in the cat dimension
      int64_t cat_dim_size = 0;
      for (int i = 0; i < numInputs; i++)
      {
        at::Tensor *tensor = (at::Tensor *)&tensors[i];
        if (should_skip(tensor))
        {
          continue;
        }
        cat_dim_size += tensor->size(dimension);
      }

      // Compute the size of the result
      c10::SmallVector<int64_t, SIZE> size;
      size.resize(nDims);
      for (int dim = 0; dim < nDims; dim++)
      {
        int64_t result_dim_size = notSkippedTensor->size(dim);
        if (dim == dimension)
        {
          result_dim_size = cat_dim_size;
        }
        size[dim] = result_dim_size;
      }

      return size;
    }

    at::Tensor &XLANativeFunctions::_cat_out(at::TensorList tensors, int64_t dim, at::Tensor &result)
    {
      if (tensors.size() == 1)
      {
        return result.copy_(tensors[0]);
      }

      c10::SmallVector<at::Tensor, N> inputTensors = cat_dest_tensor_list(tensors);
      int64_t dim_post_expr = 0;
      if (inputTensors.size() > 0)
      {
        dim_post_expr = inputTensors[0].dim();
      }
      dim = CalcuOpUtil::make_wrap_dim(dim, dim_post_expr);

      // executing the NPU operator
      int64_t input_number = 0;
      OpCommand cmd;
      cmd.Name("ConcatD");
      for (int i = 0; i < inputTensors.size(); i++)
      {
        if (inputTensors[i].numel() == 0)
        {
          continue;
        }
        string inputName = "x" + std::to_string(input_number++);
        cmd.Input(inputTensors[i], inputName);
      }

      cmd.Output(result)
          .Attr("N", input_number)
          .Attr("concat_dim", dim)
          .Run();

      return result;
    }

    at::Tensor &XLANativeFunctions::cat_out(at::TensorList tensors, int64_t dim, at::Tensor &result)
    {
      c10::SmallVector<at::Tensor, N> inputTensors = cat_dest_tensor_list(tensors);

      int64_t dim_post_expr = 0;
      if (inputTensors.size() > 0)
      {
        dim_post_expr = inputTensors[0].dim();
      }
      dim = CalcuOpUtil::make_wrap_dim(dim, dim_post_expr);
      auto outputSize = cat_npu_output_size(inputTensors, dim);
      OpPreparation::CheckOut(
          {tensors[0]},
          result,
          ACL_FORMAT_ND,
          tensors[0].scalar_type(),
          outputSize);
      return at::_cat_out(result, tensors, dim);
    }

    at::Tensor &XLANativeFunctions::cat_out(at::TensorList tensors, at::Dimname dim, at::Tensor &result)
    {
      return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
    }

    at::Tensor XLANativeFunctions::_cat(at::TensorList tensors, int64_t dim)
    {
      c10::SmallVector<at::Tensor, N> inputTensors = cat_dest_tensor_list(tensors);

      int64_t dim_post_expr = 0;
      if (inputTensors.size() > 0)
      {
        dim_post_expr = inputTensors[0].dim();
      }
      dim = CalcuOpUtil::make_wrap_dim(dim, dim_post_expr);

      // calculate the output size
      auto outputSize = cat_npu_output_size(inputTensors, dim);

      // check tensors_dim for output format setting
      bool tensors_dim_check = true;
      for (at::Tensor t : tensors)
      {
        if (t.sizes().size() != 4)
        {
          break;
        }
        int64_t C = t.size(1);
        if (C % 16 != 0)
        {
          tensors_dim_check = false;
          break;
        }
      }

      // construct the output tensor of the NPU
      if (tensors_dim_check == true)
      {
        at::Tensor result = OpPreparation::ApplyTensor(tensors[0], outputSize);
        XLANativeFunctions::_cat_out(tensors, dim, result);
        return result;
      }
      else
      {
        at::Tensor result = OpPreparation::ApplyTensorWithFormat(tensors[0], outputSize, ACL_FORMAT_ND);
        XLANativeFunctions::_cat_out(tensors, dim, result);
        return result;
      }
    }

    at::Tensor XLANativeFunctions::cat(at::TensorList tensors, int64_t dim)
    {
      return at::_cat(tensors, dim);
    }

    at::Tensor XLANativeFunctions::cat(at::TensorList tensors, at::Dimname dim)
    {
      return at::cat(tensors, dimname_to_position(tensors[0], dim));
    }

  } // namespace native
} // namespace at_npu