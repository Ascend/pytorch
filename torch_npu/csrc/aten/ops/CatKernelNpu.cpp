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

#include <ATen/native/TypeProperties.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

c10::SmallVector<at::Tensor, N> cat_dest_tensor_list(at::TensorList tensors) {
  at::ScalarType high_type = at::native::result_type(tensors);
  c10::SmallVector<at::Tensor, N> dst_tensor_list;
  // pytorch supports empty tensors, which needs to be removed from the NPU.
  for (at::Tensor tensor : tensors) {
    if (tensor.dim() == 1 && tensor.sizes()[0] == 0) {
      continue;
    }
    if (tensor.scalar_type() != high_type) {
      tensor = NPUNativeFunctions::npu_dtype_cast(tensor, high_type);
    }
    dst_tensor_list.emplace_back(tensor);
  }
  return dst_tensor_list;
}

c10::SmallVector<int64_t, SIZE> cat_npu_output_size(c10::SmallVector<at::Tensor, N_SIZE>& tensors, int64_t dimension) {
  bool all_skipped = true;
  int64_t n_dims = 0;
  at::Tensor *not_skipped_tensor;
  int num_inputs = tensors.size();
  auto should_skip = [](const at::Tensor *t) {
    return t->nbytes() == 0 && t->dim() == 1;
  };

  for (int i = 0; i < num_inputs; i++) {
    if (should_skip((at::Tensor *)&tensors[i])) {
      continue;
    }
    // found a non-empty tensor
    all_skipped = false;
    not_skipped_tensor = (at::Tensor *)&tensors[i];
    n_dims = not_skipped_tensor->dim();
    break;
  }

  if (all_skipped) {
    c10::SmallVector<int64_t, SIZE> size = {0};
    return size;
  }

  int64_t cat_dim_size = 0;
  for (int i = 0; i < num_inputs; i++) {
    at::Tensor *tensor = (at::Tensor *)&tensors[i];
    if (should_skip(tensor)) {
      continue;
    }
    cat_dim_size += tensor->size(dimension);
  }

  c10::SmallVector<int64_t, SIZE> size;
  size.resize(n_dims);
  for (int dim = 0; dim < n_dims; dim++) {
    int64_t result_dim_size = not_skipped_tensor->size(dim);
    if (dim == dimension) {
      result_dim_size = cat_dim_size;
    }
    size[dim] = result_dim_size;
  }
  return size;
}

at::Tensor& cat_out_nocheck( at::Tensor& result, at::TensorList tensors, int64_t dim) {
  c10::SmallVector<at::Tensor, N> input_tensors = cat_dest_tensor_list(tensors);
  int64_t dim_post_expr = 0;
  if (input_tensors.size() > 0) {
    dim_post_expr = input_tensors[0].dim();
  } else {
    return result;
  }
  dim = CalcuOpUtil::MakeWrapDim(dim, dim_post_expr);
  int64_t input_number = 0;
  OpCommand cmd;
  cmd.Name("ConcatD");

  // In graph mode, if all of input tensors are null numel,
  // these null tensors should be passed to ConcatD as inputs.
  // Otherwise, an error will be reported when infershape.
  bool tensors_empty_in_graph_mode = false;
  if (c10_npu::NpuRunMode::IsGraphMode()) {
    tensors_empty_in_graph_mode = true;
    for (int i = 0; i < input_tensors.size(); i++) {
      if (input_tensors[i].numel() != 0) {
        tensors_empty_in_graph_mode = false;
        break;
      }
    }
  }
  input_number = 0;
  for (int i = 0; i < input_tensors.size(); i++) {
    if (input_tensors[i].numel() != 0 || tensors_empty_in_graph_mode) {
      string input_name = "x" + std::to_string(input_number++);
      cmd.Input(input_tensors[i], input_name);
    }
  }

  cmd.Output(result)
      .Attr("N", input_number)
      .Attr("concat_dim", dim)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::_cat_out(at::TensorList tensors, int64_t dim, at::Tensor& result) {
  if (tensors.size() == 1) {
    return result.copy_(tensors[0]);
  }

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    cat_out_nocheck(contiguous_result, tensors, dim);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    cat_out_nocheck(result, tensors, dim);
  }
  return result;
}

at::Tensor& NPUNativeFunctions::cat_out(at::TensorList tensors, int64_t dim, at::Tensor& result) {
  c10::SmallVector<at::Tensor, N> input_tensors = cat_dest_tensor_list(tensors);

  int64_t dim_post_expr = 0;
  if (input_tensors.size() > 0) {
    dim_post_expr = input_tensors[0].dim();
  } else {
    return result;
  }
  dim = CalcuOpUtil::MakeWrapDim(dim, dim_post_expr);
  auto output_size = cat_npu_output_size(input_tensors, dim);
  OpPreparation::CheckOut(
      {tensors[0]},
      result,
      ACL_FORMAT_ND,
      tensors[0].scalar_type(),
      output_size);

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    at::_cat_out(contiguous_result, tensors, dim);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    at::_cat_out(result, tensors, dim);
  }
  return result;
}

at::Tensor& NPUNativeFunctions::cat_out(at::TensorList tensors, at::Dimname dim, at::Tensor& result) {
  return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
}

at::Tensor NPUNativeFunctions::_cat(at::TensorList tensors, int64_t dim) {
  c10::SmallVector<at::Tensor, N> input_tensors = cat_dest_tensor_list(tensors);

  int64_t dim_post_expr = 0;
  if (input_tensors.size() > 0) {
    dim_post_expr = input_tensors[0].dim();
  } else {
    at::Tensor result = OpPreparation::ApplyTensor(tensors[0]);
    return result;
  }
  dim = CalcuOpUtil::MakeWrapDim(dim, dim_post_expr);
  auto output_size = cat_npu_output_size(input_tensors, dim);

  // check tensors_dim for output format setting
  bool tensors_dim_check = true;
  for (at::Tensor t : tensors) {
    if (t.sizes().size() != 4) {
      break;
    }
    int64_t C = t.size(1);
    if (C % 16 != 0) {
      tensors_dim_check = false;
      break;
    }
  }

  at::Tensor result = OpPreparation::ApplyTensor(input_tensors[0], output_size);
  if (!tensors_dim_check) {
    result = OpPreparation::ApplyTensorWithFormat(input_tensors[0], output_size, ACL_FORMAT_ND);
  }
  NPUNativeFunctions::_cat_out(tensors, dim, result);
  return result;
}

at::Tensor NPUNativeFunctions::cat(at::TensorList tensors, int64_t dim) {
  return at::_cat(tensors, dim);
}

at::Tensor NPUNativeFunctions::cat(at::TensorList tensors, at::Dimname dim) {
  return at::cat(tensors, dimname_to_position(tensors[0], dim));
}

} // namespace native
} // namespace at_npu
