// Copyright (c) 2022 Huawei Technologies Co., Ltd
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

#ifndef __NATIVE_NPU_UTILS_ADVANCED_INDEX__
#define __NATIVE_NPU_UTILS_ADVANCED_INDEX__

#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/IndexingUtils.h>

namespace at {
namespace native {
namespace npu {

struct AdvancedIndex {
  AdvancedIndex(const Tensor& src, TensorList indices);
  Tensor src;
  std::vector<Tensor> indices;
  DimVector indexed_sizes;
  DimVector indexed_strides;
  int64_t dims_before;
  int64_t dims_after;
};

class AdvanceIndex {
public:
  static bool all_strides_match(TensorList tensors);
  static std::string shapes_as_str(TensorList tensors);
  static Tensor restride_src(
      const Tensor& src,
      int64_t dims_before,
      int64_t dims_indexed,
      IntArrayRef replacement_shape);
  // Add dimensions of size 1 to an index tensor so that it can be broadcast to the result
  // shape and iterated over element-wise like the result tensor and the restrided src.
  static Tensor reshape_indexer(const Tensor& index, int64_t dims_before, int64_t dims_after);
  static AdvancedIndex make_info(Tensor self, TensorList orig);
}; // AdvanceIndex

} // namespace npu
} // namespace native
} // namespace at

#endif
