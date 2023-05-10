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

#include <ATen/core/IListRef.h>
#include <ATen/native/TypeProperties.h>
#include <third_party/acl/inc/op_proto/split_combination_ops.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    namespace
    {
    template <typename ge_op_type>
    at_npu::native::DynamicInputRegFunc concat_func =
        [](DyNumAndIndex num_and_index,
          std::string op_name) -> ge::OperatorPtr
          {
            auto ge_op = std::make_shared<ge_op_type>(op_name.c_str());
            ge_op->create_dynamic_input_byindex_x(
                num_and_index.front().first, num_and_index.front().second);
            return ge_op;
          };
    }

    c10::SmallVector<at::Tensor, N> cat_dest_tensor_list(const at::MaterializedITensorListRef& tensors)
    {
      auto high_type = at::native::result_type(tensors);
      c10::SmallVector<at::Tensor, N> dstTensorList;
      // pytorch supports empty tensors, which needs to be removed from the NPU.
      for (const at::Tensor& t : tensors)
      {
        at::Tensor tensor = t;
        if (tensor.dim() == 1 && tensor.sizes()[0] == 0)
        {
          continue;
        }
        if (tensor.scalar_type() != high_type)
        {
          tensor = NPUNativeFunctions::npu_dtype_cast(tensor, high_type);
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

    at::Tensor &_cat_out(const at::MaterializedITensorListRef& tensors, int64_t dim, at::Tensor &result)
    {
      if (tensors.size() == 1)
      {
        return result.copy_(tensors[0].get());
      }

      c10::SmallVector<at::Tensor, N> inputTensors = cat_dest_tensor_list(tensors);
      int64_t dim_post_expr = 0;
      if (inputTensors.size() > 0)
      {
        dim_post_expr = inputTensors[0].dim();
      } else {
        return result;
      }
      dim = CalcuOpUtil::MakeWrapDim(dim, dim_post_expr);

      // executing the NPU operator
      int64_t input_number = 0;
      OpCommand cmd;
      cmd.Name("ConcatD");

      // In graph mode, if all of input tensors are null numel,
      // these null tensors should be passed to ConcatD as inputs.
      // Otherwise, an error will be reported when infershape.
      bool tensors_empty_in_graph_mode = false;
      if (c10_npu::NpuRunMode::IsGraphMode()) {
        tensors_empty_in_graph_mode = true;
        for (int i = 0; i < inputTensors.size(); i++) {
          if (inputTensors[i].numel() != 0) {
            tensors_empty_in_graph_mode = false;
            break;
          }
        }
      }
      input_number = 0;
      for (int i = 0; i < inputTensors.size(); i++)
      {
        if (inputTensors[i].numel() != 0 || tensors_empty_in_graph_mode)
        {
          string inputName = "x" + std::to_string(input_number++);
          cmd.Input(inputTensors[i], inputName);
        }
      }

      cmd.DynamicInputReg(concat_func<ge::op::ConcatD>, {{input_number, 0}})
          .Output(result)
          .Attr("N", input_number)
          .Attr("concat_dim", dim)
          .Run();

      return result;
    }

    at::Tensor &NPUNativeFunctions::cat_out(const at::ITensorListRef& tensors, int64_t dim, at::Tensor &result)
    {
      auto materialized = tensors.materialize();
      c10::SmallVector<at::Tensor, N> inputTensors = cat_dest_tensor_list(materialized);

      int64_t dim_post_expr = 0;
      if (inputTensors.size() > 0)
      {
        dim_post_expr = inputTensors[0].dim();
      } else {
        return result;
      }
      dim = CalcuOpUtil::MakeWrapDim(dim, dim_post_expr);
      auto outputSize = cat_npu_output_size(inputTensors, dim);
      OpPreparation::CheckOut(
          {materialized[0].get()},
          result,
          ACL_FORMAT_ND,
          materialized[0].get().scalar_type(),
          outputSize);
      return _cat_out(materialized, dim, result);
    }

    at::Tensor &NPUNativeFunctions::cat_out(at::TensorList tensors, at::Dimname dim, at::Tensor &result)
    {
      return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
    }

    at::Tensor _cat(const at::MaterializedITensorListRef& tensors, int64_t dim)
    {
      c10::SmallVector<at::Tensor, N> inputTensors = cat_dest_tensor_list(tensors);

      int64_t dim_post_expr = 0;
      if (inputTensors.size() > 0)
      {
        dim_post_expr = inputTensors[0].dim();
      } else {
        at::Tensor result = OpPreparation::ApplyTensor(tensors[0]);
        return result;
      }
      dim = CalcuOpUtil::MakeWrapDim(dim, dim_post_expr);

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
        at::Tensor result = OpPreparation::ApplyTensor(inputTensors[0], outputSize);
        _cat_out(tensors, dim, result);
        return result;
      }
      else
      {
        at::Tensor result = OpPreparation::ApplyTensorWithFormat(inputTensors[0], outputSize, ACL_FORMAT_ND);
        _cat_out(tensors, dim, result);
        return result;
      }
    }

    at::Tensor NPUNativeFunctions::cat(const at::ITensorListRef& tensors, int64_t dim)
    {
      auto materialized = tensors.materialize();
      return _cat(materialized, dim);
    }

    at::Tensor NPUNativeFunctions::cat(at::TensorList tensors, at::Dimname dim)
    {
      return at::cat(tensors, dimname_to_position(tensors[0], dim));
    }

  } // namespace native
} // namespace at_npu
