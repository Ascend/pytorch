// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/contiguous/ReshapeOpt.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

bool can_use_memecpy_for_NZ_format(const ContiguousTensorDesc &tensor_desc) {
  int64_t tensor_shape_size = tensor_desc.sizes_.size();
  int64_t base_shape_size = tensor_desc.base_sizes_.size();
  // No padding&&offset!=0 at the same time. e.g. x(3, 15, 16)[1:]
  if (((tensor_desc.sizes_[tensor_shape_size - 1] % 16 != 0) ||
       (tensor_desc.sizes_[tensor_shape_size - 2] % 16 != 0)) &&
      tensor_desc.offset_ != 0) {
    return false;
  }
  // Make sure that sizes of last 2 dims don't change
  if (tensor_desc.sizes_[tensor_shape_size - 1] !=
          tensor_desc.base_sizes_[base_shape_size - 1] ||
      tensor_desc.sizes_[tensor_shape_size - 2] !=
          tensor_desc.base_sizes_[base_shape_size - 2]) {
    return false;
  }
  return true;
}

bool can_use_memcpy_for_other_format(const ContiguousTensorDesc &tensor_desc) {
  // torch.flatten(x) case should be removed
  if (tensor_desc.sizes_.size() < 2) {
    return false;
  }
  switch (tensor_desc.npu_format_) {
  case ACL_FORMAT_FRACTAL_NZ:
    return can_use_memecpy_for_NZ_format(tensor_desc);
  // (Ascend): 5HD format can also be optimized likes NZ format
  default:
    // For other format, make sure that copy the whole memory.
    // Moreover, storage size expanding caused by padding could be avoided
    if (!(tensor_desc.base_sizes_ == tensor_desc.sizes_)) {
      return false;
    }
    // Make sure no pandding happens
    if (c10::multiply_integers(tensor_desc.sizes_) !=
        c10::multiply_integers(tensor_desc.storage_sizes_)) {
      return false;
    }
    return true;
  }
}

bool check_reshape_match(const ContiguousTensorDesc &self_desc,
                         const ContiguousTensorDesc &src_desc) {
  // For all format, both src and self are taken into consideration
  if (check_reshape_match(src_desc) && check_reshape_match(self_desc)) {
    // tensor numels eqs for self and src tensor. i.e. make sure that storage
    // keep same.
    if (!(self_desc.sizes_ == src_desc.sizes_)) {
      return false;
    }

    IF_GRAPH_MODE_THEN_RUN(
      // In single op mode, this opt will be used for reshape/slice/select
      // scenes. In graph mode, reshape opt is only used for reshape scenes,
      // npu-reshape is used to calculae and get contiguous tensor.
      if (c10::multiply_integers(src_desc.base_sizes_) != c10::multiply_integers(src_desc.sizes_)) {
        return false;
      }
    );

    return true;
  }
  return false;
}

bool check_reshape_match(const ContiguousTensorDesc &tensor_desc) {
  // (case 1) Reshape tensor should be contiguous
  if (!tensor_desc.is_contiguous_) {
    return false;
  }
  // (case2) for other format, sizes at key dims should remain unchanged
  if (!FormatHelper::IsBaseFormatType(tensor_desc.npu_format_)) {
    return can_use_memcpy_for_other_format(tensor_desc);
  }
  return true;
}

bool CanUseMemcpyForOtherFormat(const at::Tensor &tensor) {
  ContiguousTensorDesc tensor_desc = TransContiguous::GetTensorDescInfo(tensor);
  return can_use_memcpy_for_other_format(tensor_desc);
}

} // namespace native
} // namespace at_npu

