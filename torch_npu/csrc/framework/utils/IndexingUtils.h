#pragma once
#include <ATen/ExpandUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/core/List.h>

#include <limits>

namespace at_npu { namespace native {

TORCH_API bool canUse32BitIndexMath(const at::Tensor &t, int64_t max_elem=std::numeric_limits<int32_t>::max());

[[noreturn]] static void invalid_mask(const at::Tensor & self, int64_t idx, const at::Tensor & mask, int64_t maskIdx) {
  TORCH_CHECK_INDEX(false, "The shape of the mask ", mask.sizes(), " at index ", maskIdx,
      " does not match the shape of the indexed tensor ", self.sizes(), " at index ", idx);
}


static std::vector<at::Tensor> expandTensors(const at::Tensor & self, const torch::List<c10::optional<at::Tensor>>& indices) {
  // If indices come in as ByteTensor or BoolTensor (masks), expand them into the equivalent indexing by LongTensors
  std::vector<at::Tensor> result;
  for (c10::optional<at::Tensor> index_opt : indices) {
    if (!index_opt.has_value()) {
      result.emplace_back();
    } else {
      at::Tensor index = std::move(*index_opt);
      if (index.scalar_type() == at::kByte || index.scalar_type() == at::kBool) {
        if (index.scalar_type() == at::kByte) {
          TORCH_WARN("indexing with dtype torch.uint8 is now deprecated," \
          " please use a dtype torch.bool instead.");
        }
        // The sizes of the ByteTensor mask or bool tensor must match the sizes of the
        // corresponding dimensions in self
        for (int64_t j = 0; j < index.dim(); j++) {
          int64_t srcIdx = result.size() + j;
          if (index.size(j) != self.size(srcIdx)) {
            invalid_mask(self, srcIdx, index, j);
          }
        }
        // Replace with nonzeros
        auto nonzero = index.nonzero();
        for (int64_t j = 0; j < index.dim(); j++) {
          result.emplace_back(nonzero.select(1, j));
        }
      } else {
        result.emplace_back(std::move(index));
      }
    }
  }
  return result;
}


static void checkIndexTensorTypes(const torch::List<c10::optional<at::Tensor>>& indices) {
  for (c10::optional<at::Tensor> tensor : indices) {
    if (tensor.has_value() && tensor->defined()) {
      auto scalarType = tensor->scalar_type();
      if (scalarType != at::kLong && scalarType != at::kByte && scalarType != at::kBool) {
          TORCH_CHECK_INDEX(false, "tensors used as indices must be long, byte or bool tensors");
      }
    }
  }
}

inline torch::List<c10::optional<at::Tensor>> toListOfOptionalTensors(at::ArrayRef<at::Tensor> list) {
  torch::List<c10::optional<at::Tensor>> result;
  result.reserve(list.size());
  for (const at::Tensor& a : list) {
    result.push_back(a);
  }
  return result;
}

static bool hasContiguousSubspace(at::TensorList tl) {
  // true if all the non-null tensors are adjacent
  auto isDefined = [](const at::Tensor & tensor){ return tensor.defined(); };
  auto isNull = [](const at::Tensor & tensor){ return !tensor.defined(); };
  auto start = std::find_if(tl.begin(), tl.end(), isDefined);
  auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
  auto it = std::find_if(start, stop.base(), isNull);
  return it == stop.base();
}


// Transposes the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor
// and the reordered indices. For example:
// transposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
// tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
static std::tuple<at::Tensor, std::vector<at::Tensor>> transposeToFront(
    at::Tensor self, 
    at::TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<at::Tensor> transposedIndices;
  dims.reserve(self.dim());
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (!indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back();
    }
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices));
}

inline std::tuple<at::Tensor, std::vector<at::Tensor>, std::vector<int64_t>> transposeToFrontAndInvPerm(
    at::Tensor self, 
    at::TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<int64_t> invPerm;
  std::vector<at::Tensor> transposedIndices;
  dims.reserve(self.dim());
  invPerm.resize(self.dim());
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (!indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back();
    }
  }
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    invPerm[dims[i]] = i;
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices), std::move(invPerm));
}

struct AdvancedIndex {
  AdvancedIndex(const at::Tensor& src, at::TensorList indices);

  at::Tensor src;
  std::vector<at::Tensor> indices;
  at::DimVector indexed_sizes;
  at::DimVector indexed_strides;
  int64_t dims_before;
  int64_t dims_after;
};


}}
