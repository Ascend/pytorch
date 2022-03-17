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

#include<ATen/NamedTensorUtils.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
void index_copy_npu_par_check(
    const int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Tensor& result) {
  int64_t newDim = at::maybe_wrap_dim(dim, result.dim());
  TORCH_CHECK_INDEX(index.dim() < 2, "index_copy_(): Index should have dimension 1 or 0 (got ", index.dim(), ")");

  int64_t numIndices = index.numel();
  TORCH_CHECK_INDEX(!(source.dim() == 0 && numIndices != 1),
      "index_copy_(): When source is scalar, index should have one element (got ", numIndices, ")");
  TORCH_CHECK_INDEX(!((source.dim() != result.dim()) && (source.dim() != 0 && result.dim() != 0)),
      "index_copy_(): When source and destination are not scalars, \
their dimensionality must match. Source dimensionality (",
      source.dim(), "), destination dimensionality (", result.dim(), ")");
  
  TORCH_CHECK_INDEX(index.scalar_type() == at::ScalarType::Long, "index_copy_(): Expected LongTensor for index");

  // Check that source and destination slices have the same size
  auto selfSlicedSizes = result.sizes().vec();
  if (selfSlicedSizes.size() > 0) {
    selfSlicedSizes.erase(selfSlicedSizes.begin() + newDim);
  }
  auto sourceSlicedSizes = source.sizes().vec();
  if (sourceSlicedSizes.size() > 0) {
    sourceSlicedSizes.erase(sourceSlicedSizes.begin() + newDim);
  }
  if (selfSlicedSizes.size() != sourceSlicedSizes.size() ||
      !std::equal(selfSlicedSizes.begin(), selfSlicedSizes.end(),
                  sourceSlicedSizes.begin())) {
    std::stringstream ss;
    ss << "index_copy_(): Source/destination tensor must have same slice shapes. ";
    ss << "Destination slice shape: " << selfSlicedSizes << " at dimension " << newDim;
    ss << " and source slice shape: " << sourceSlicedSizes << " at dimension 0.";
    TORCH_CHECK(false, ss.str());
  }
  TORCH_CHECK_INDEX(source.dim() == 0 || numIndices == source.size(newDim),
      "index_copy_(): Number of indices (", numIndices,
      ") should be equal to source.size(newDim) (", source.size(newDim), ")");
}

at::Tensor& index_copy_npu_impl(
    const int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    at::Tensor& result) {
  index_copy_npu_par_check(dim, index, source, result);
  int64_t numIndices = index.numel();
  int64_t i;
  if (result.dim() > 1) {
    at::Tensor des;
    at::Tensor src;
    for (i = 0; i < numIndices; i++) {
      des = at::native::select(result, dim, index[i].item<int64_t>());
      src = at::native::select(source, dim, i);
      NPUNativeFunctions::copy_(des, src, false);
    }
  } else {
    for (i = 0; i < numIndices; i++) {
      result[i] = source[index[i].item<int64_t>()];
    }
  }
  return result;
}

at::Tensor index_copy_npu(
    const at::Tensor& self,
    const int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  at::Tensor result(self.clone());
  return index_copy_npu_impl(dim, index, source, result);
}

at::Tensor index_copy_npu(
    const at::Tensor& self,
    const at::Dimname dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  at::Tensor result(self.clone());
  return index_copy_npu_impl(dimname_to_position(self, dim), index, source, result);
}

at::Tensor& NPUNativeFunctions::_index_copy_(
    at::Tensor& self,
    const int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  at::Tensor contiguousSelf(self);
  if (!NpuUtils::check_match(&self)) {
    contiguousSelf = NpuUtils::format_contiguous(self);
  }
  at::Tensor result = index_copy_npu_impl(dim, index, source, contiguousSelf);
  NpuUtils::format_fresh_view(self, result);

  return self;
}

at::Tensor& index_copy_npu_(
    at::Tensor& self,
    const at::Dimname dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  at::Tensor contiguousSelf(self);
  if (!NpuUtils::check_match(&self)) {
    contiguousSelf = NpuUtils::format_contiguous(self);
  }
  at::Tensor result = index_copy_npu_impl(dimname_to_position(self, dim), index, source, contiguousSelf);
  NpuUtils::format_fresh_view(self, result);

  return self;
}
} // namespace native
} // namespace at_npu
