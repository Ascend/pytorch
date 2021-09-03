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

#include "c10/npu/OptionsManager.h"
#include "ATen/native/npu/utils/OpAdapter.h"
#include <torch/script.h>

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<Tensor, N> cat_dest_tensor_list(TensorList tensors) {
  SmallVector<Tensor, N> dstTensorList;
  // pytorch supports empty tensors, which needs to be removed from the NPU.
  for (Tensor tensor : tensors) {
    if (tensor.dim() == 1 && tensor.sizes()[0] == 0) {
      continue;
    }

    dstTensorList.emplace_back(tensor);
  }

  return dstTensorList;
}

SmallVector<int64_t, SIZE> cat_npu_output_size(
    SmallVector<Tensor, N_SIZE>& tensors,
    int64_t dimension) {
  bool allSkipped = true;
  int64_t nDims = 0;
  Tensor* notSkippedTensor;
  int numInputs = tensors.size();
  auto should_skip = [](const Tensor* t) {
    return t->nbytes() == 0 && t->dim() == 1;
  };
  for (int i = 0; i < numInputs; i++) {
    if (should_skip((Tensor*)&tensors[i])) {
      continue;
    }
    // found a non-empty tensor
    allSkipped = false;
    notSkippedTensor = (Tensor*)&tensors[i];
    nDims = notSkippedTensor->dim();
    break;
  }

  if (allSkipped) {
    SmallVector<int64_t, SIZE> size = {0};
    return size;
  }

  // Compute size of the result in the cat dimension
  int64_t cat_dim_size = 0;
  for (int i = 0; i < numInputs; i++) {
    Tensor* tensor = (Tensor*)&tensors[i];
    if (should_skip(tensor)) {
      continue;
    }
    cat_dim_size += tensor->size(dimension);
  }

  // Compute the size of the result
  SmallVector<int64_t, SIZE> size;
  size.resize(nDims);
  for (int dim = 0; dim < nDims; dim++) {
    int64_t result_dim_size = notSkippedTensor->size(dim);
    if (dim == dimension) {
      result_dim_size = cat_dim_size;
    }
    size[dim] = result_dim_size;
  }

  return size;
}

Tensor& _cat_out_npu(TensorList tensors, int64_t dim, Tensor& result) {
  // TODO(ascend): Remove after FE support one tensor input
  if (tensors.size() == 1) {
    return result.copy_(tensors[0]);
  }

  SmallVector<Tensor, N> inputTensors = cat_dest_tensor_list(tensors);
  dim = CalcuOpUtil::make_wrap_dim(dim, inputTensors[0].dim());

  // executing the NPU operator
  int64_t input_number = 0;
  OpCommand cmd;
  cmd.Name("ConcatD");
  input_number = 0;
  for (int i = 0; i < inputTensors.size(); i++) {
    if (inputTensors[i].numel() == 0) {
      continue;
    }
    string inputName = "x" + to_string(input_number++);
    cmd.Input(inputTensors[i], inputName);
  }
  cmd.Output(result)
    .Attr("N", input_number)
    .Attr("concat_dim", dim)
    .Run();

  return result;
}

Tensor& cat_out_npu(TensorList tensors, int64_t dim, Tensor& result) {
  return at::_cat_out(result, tensors, dim);
}

Tensor& cat_dimname_out_npu(TensorList tensors, Dimname dim, Tensor& result) {
  return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
}

Tensor _cat_npu(TensorList tensors, int64_t dim) {
  SmallVector<Tensor, N> inputTensors = cat_dest_tensor_list(tensors);

  dim = CalcuOpUtil::make_wrap_dim(dim, inputTensors[0].dim());

  // calculate the output size
  auto outputSize = cat_npu_output_size(inputTensors, dim);

  // check tensors_dim for output format setting
  bool tensors_dim_check = true;
  for (Tensor t : tensors) {
    if (t.sizes().size() != 4) {
      break;
    }
    int64_t C = t.size(1);
    if (C % 16 != 0) {
      tensors_dim_check = false;
      break;
    }
  }

  // construct the output tensor of the NPU
  if (tensors_dim_check == true) {
    Tensor result = at::empty_with_format(
        outputSize,
        inputTensors[0].options(),
        CalcuOpUtil::get_tensor_npu_format(inputTensors[0]));

    _cat_out_npu(tensors, dim, result);
    return result;
  } else {
    Tensor result = at::empty_with_format(
        outputSize, inputTensors[0].options(), ACL_FORMAT_ND);

    _cat_out_npu(tensors, dim, result);
    return result;
  }
}

Tensor cat_npu(TensorList tensors, int64_t dim) {
  return at::_cat(tensors, dim);
}

Tensor cat_dimname_npu(TensorList tensors, Dimname dim) {
  return at::cat(tensors, dimname_to_position(tensors[0], dim));
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("cat", TORCH_FN(cat_npu));
  m.impl("cat.out", TORCH_FN(cat_out_npu));
  m.impl("cat.names", TORCH_FN(cat_dimname_npu));
  m.impl("cat.names_out", TORCH_FN(cat_dimname_out_npu));
  m.impl("_cat", TORCH_FN(_cat_npu));
  m.impl("_cat.out", TORCH_FN(_cat_out_npu));
}

} // namespace native
} // namespace at