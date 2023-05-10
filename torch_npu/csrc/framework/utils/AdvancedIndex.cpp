// Copyright (c) 2022 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at_npu
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "AdvancedIndex.h"
#include "NpuUtils.h"
#include "OpPreparation.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace at_npu {
namespace native {

AdvancedIndex::AdvancedIndex(const at::Tensor& src, at::TensorList indices_list) {
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  at::IntArrayRef replacement_shape;
  for (size_t dim = 0; dim < indices_list.size(); dim++) {
    if (!indices_list[dim].defined()) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = indices_list[dim].sizes();
      indexed_sizes.push_back(src.size(dim));
      indexed_strides.push_back(src.stride(dim));
    }
  }

  // Check if the indexed subspace contains a dim of size 0, but the replacement
  // shape does not. This implies that an index is out of bounds, because there
  // is no number that's a valid index for an empty tensor. Normally, out of
  // bounds is handled in the indexing kernel, but this case fails earlier in
  // restride_src with an unhelpful error message.
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
    TORCH_CHECK_INDEX(false, "index is out of bounds for dimension with size 0");
  }

  this->dims_before = dims_before;
  this->dims_after = dims_after;
  this->src = AdvanceIndex::restride_src(src, dims_before, dims_indexed, replacement_shape);

  for (auto& index : indices_list) {
    if (index.defined()) {
      indices.push_back(AdvanceIndex::reshape_indexer(index, dims_before, dims_after));
    }
  }
}

bool AdvanceIndex::all_strides_match(at::TensorList tensors) {
  TORCH_CHECK(tensors.size() >= 1);
  auto strides = tensors[0].strides();
  for (auto& tensor : tensors.slice(1)) {
    if (!strides.equals(tensor.strides())) {
      return false;
    }
  }
  return true;
}

at::Tensor AdvanceIndex::reshape_indexer(const at::Tensor& index, int64_t dims_before, int64_t dims_after) {
  auto orig_shape = index.sizes();
  auto shape = at::DimVector();
  shape.append(dims_before, 1);
  shape.append(orig_shape.begin(), orig_shape.end());
  shape.append(dims_after, 1);
  if (index.dtype() == at::kLong) {
    return index.reshape(shape);
  } else {
    return index.reshape(shape).to(at::kLong);
  }
}

at::Tensor AdvanceIndex::restride_src(const at::Tensor& src, int64_t dims_before, int64_t dims_indexed,
    at::IntArrayRef replacement_shape) {
  auto shape = at::DimVector(src.sizes());
  auto strides = at::DimVector(src.strides());
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  strides.erase(strides.begin() + dims_before, strides.begin() + end);
  shape.insert(shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());
  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
  return src.as_strided(shape, strides);
}

std::string AdvanceIndex::shapes_as_str(at::TensorList tensors) {
  std::ostringstream os;
  bool first = true;
  for (auto& tensor : tensors) {
    if (tensor.defined()) {
      if (!first) {
        os << ", ";
      }
      os << tensor.sizes();
      first = false;
    }
  }
  return os.str();
}

std::vector<at::Tensor> npu_expand_outplace(at::TensorList to_expand) {
  // expands a list of Tensors; ignores undefined (null) tensors
  bool first = true;
  std::vector<int64_t> sizes;
  for (size_t i = 0; i < to_expand.size(); ++i) {
    if (!to_expand[i].defined()) {
      continue;
    } else if (first) {
      sizes = to_expand[i].sizes().vec();
      first = false;
    } else {
      sizes = at::infer_size(sizes, to_expand[i].sizes());
    }
  }

  std::vector<at::Tensor> result(to_expand.size());
  for (size_t i = 0; i < to_expand.size(); ++i) {
    if (!to_expand[i].defined()) {
      continue;
    } else if (to_expand[i].sizes().equals(sizes)) {
      result[i] = to_expand[i];
    } else {
      if (to_expand[i].dtype() == at::kLong) {
        result[i] = to_expand[i].to(at::kInt).expand(sizes, true);
      } else {
        result[i] = to_expand[i].expand(sizes, true);
      }
    }
  }
  return result;
}

AdvancedIndex AdvanceIndex::make_info(at::Tensor self, const torch::List<c10::optional<at::Tensor>>& orig) {
  at::native::checkIndexTensorTypes(orig);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
  auto indices = at::native::expandTensors(self, orig);
  // next broadcast all index tensors together
  try {
    indices = npu_expand_outplace(indices);
  } catch (std::exception& e) {
    TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together"
        " with shapes ", shapes_as_str(indices));
  }
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  if (!at::native::hasContiguousSubspace(indices)) {
    std::tie(self, indices) = at::native::transposeToFront(self, indices);
  }
  // Ensure indices are on the same device as self
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined() && indices[i].device() != self.device()) {
      indices[i] = indices[i].to(self.device());
    }
  }
  return AdvancedIndex(self, indices);
}

at::Tensor npu_nonzero_transpose(const at::Tensor& self) {
  c10::SmallVector<int64_t, SHAPE_SIZE> output_size = {self.dim(), self.numel()};
  at::Tensor result = OpPreparation::ApplyTensor(
      output_size, self.options().dtype(at::kLong), self);
  c10::SmallVector<int64_t, N> output_sync_idx = {0};
  OpCommand cmd;
  cmd.Sync(output_sync_idx)
      .Name("NonZero")
      .Input(self)
      .Output(result)
      .Attr("transpose", true)
      .Run();
  return result;
}

std::vector<at::Tensor> AdvanceIndex::npu_expand_tensors(
    const at::Tensor& self,
    const torch::List<c10::optional<at::Tensor>>& indices) {
  // If indices come in as ByteTensor or BoolTensor (masks), expand them into the equivalent indexing by LongTensors
  std::vector<at::Tensor> result;
  for (c10::optional<at::Tensor> index_opt : indices) {
    if (!index_opt.has_value()) {
      result.emplace_back();
    } else {
      at::Tensor index = std::move(*index_opt);
      if (index.defined() && index.device() != self.device()) {
        index = index.to(self.device());
      }
      if (index.scalar_type() == at::kByte || index.scalar_type() == at::kBool) {
        if (index.scalar_type() == at::kByte) {
          TORCH_WARN("indexing with dtype torch.uint8 is now deprecated," \
              " please use a dtype torch.bool instead.");
        }
        // The sizes of the ByteTensor mask or bool tensor must match the sizes of the corresponding dimensions in self
        for (int64_t j = 0; j < index.dim(); j++) {
          int64_t srcIdx = result.size() + j;
          if (index.size(j) != self.size(srcIdx)) {
            TORCH_CHECK_INDEX(false, "The shape of the mask ", index.sizes(), " at index ", j,
                " does not match the shape of the indexed tensor ", self.sizes(), " at index ", srcIdx);
          }
        }
        // Replace with nonzeros
        auto nonzero = npu_nonzero_transpose(index);
        for (int64_t j = 0; j < index.dim(); j++) {
          result.emplace_back(nonzero.select(0, j));
        }
      } else {
        result.emplace_back(std::move(index));
      }
    }
  }
  return result;
}

} // namespace native
} // namespace at_npu
