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

#include <ATen/native/npu/contiguous/ReshapeOpt.h>
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
namespace npu {

bool can_use_memecpy_for_NZ_format(const Tensor& tensor) {
  auto base_size = tensor.storage().get_npu_desc().base_sizes_;
  // No padding&&offset!=0 at same time. e.g. x(3, 15, 16)[1:] with format=29
  if (((tensor.size(-1) % 16 != 0) || (tensor.size(-2) % 16 != 0)) &&
      tensor.storage_offset() != 0) {
    return false;
  }

  // Make sure that sizes of last 2 dims don't change
  if (tensor.size(-1) != base_size[base_size.size() - 1] ||
      tensor.size(-2) != base_size[base_size.size() - 2]) {
    return false;
  }
  return true;
}

bool can_use_memcpy_for_other_format(const Tensor& src) {
  // torch.flatten(x) case should be removed
  if (src.sizes().size() < 2) {
    return false;
  }
  auto srcNpuDesc = src.storage().get_npu_desc();
  switch(srcNpuDesc.npu_format_) {
    case ACL_FORMAT_FRACTAL_NZ:
      return can_use_memecpy_for_NZ_format(src);
    // TODO: 5HD format can also be optimized likes NZ format
    default:
      // For other format, make sure that copy the whole memory.
      // Moreover, storage size expanding caused by padding could be avoided
      if (!(srcNpuDesc.base_sizes_ == array_to_small_vector(src.sizes()))) {
        return false;
      }
      // Make sure no pandding happens
      if (src.numel() != prod_intlist(srcNpuDesc.storage_sizes_)) {
        return false;
      }
      return true;
  }
}

bool check_reshape_match(const Tensor& src, Tensor& self) {
  // For all format, both src and self are taken into consideration
  if (check_reshape_match(src) && check_reshape_match(self)) {
    // tensor numels eqs for self and src tensor. i.e. make sure that storage keep same.
    if (!self.sizes().equals(src.sizes())) {
      return false;
    }

    IF_GRAPH_MODE_THEN_RUN(
      // In single op mode, this opt will be used for reshape/slice/select scenes.
      // In graph mode, reshape opt is only used for reshape scenes,
      // npu-reshape is used to calculae and get contiguous tensor.
      const auto& base_sizes = src.storage().get_npu_desc().base_sizes_;
      if (prod_intlist(base_sizes) != src.numel()) {
        return false;
      }
    );

    return true;
  }
  return false;
}

bool check_reshape_match(const Tensor& tensor) {
  // (case 1) Reshape tensor should be contiguous
  if (!tensor.is_contiguous()) {
    return false;
  }
  // (case2) for other format, sizes at key dims should remain unchanged
  if (!FormatHelper::IsBaseFormatType(tensor)) {
    return can_use_memcpy_for_other_format(tensor);
  }
  return true;
}

} // namespace npu
} // namespace native
} // namespace at
