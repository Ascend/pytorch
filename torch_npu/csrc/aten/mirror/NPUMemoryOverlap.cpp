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


#include "NPUMemoryOverlap.h"
#include <c10/core/Layout.h>

namespace at_npu { namespace native {

MemOverlap has_internal_overlap(const at::Tensor& tensor) {
  return has_internal_overlap(tensor.unsafeGetTensorImpl());
}

MemOverlap has_internal_overlap(at::TensorImpl* t) {
  AT_ASSERT(t->layout() == at::kStrided);

  if (t->is_contiguous()) {
    return MemOverlap::NO;
  }

  if (t->storage().data() == nullptr) {
    return MemOverlap::IS_NULL;
  }

  auto strides = t->strides();
  auto sizes = t->sizes();
  for (size_t i = 0; i < strides.size(); ++i) {
    if (strides[i] == 0 && sizes[i] > 1) {
      return MemOverlap::YES;
    }
  }

  return MemOverlap::TOO_HARD;
}

void assert_no_internal_overlap(const at::Tensor& t) {
  assert_no_internal_overlap(t.unsafeGetTensorImpl());
}

void assert_no_internal_overlap(at::TensorImpl* t) {
  TORCH_CHECK(has_internal_overlap(t) != MemOverlap::YES,
      "unsupported operation: more than one element of the written-to tensor "
      "refers to a single memory location. Please clone() the tensor before "
      "performing the operation.");
}

MemOverlapStatus get_overlap_status(const at::Tensor& a, const at::Tensor& b) {
  return get_overlap_status(a.unsafeGetTensorImpl(), b.unsafeGetTensorImpl());
}

MemOverlapStatus get_overlap_status(at::TensorImpl* a, at::TensorImpl* b) {
    if (a == b) {
        return MemOverlapStatus::FULL;
    }
    if (a->storage().data() == nullptr || b->storage().data() == nullptr) {
      return MemOverlapStatus::IS_NULL;
    }
    if (a->numel() == 0 || b->numel() == 0) {
        return MemOverlapStatus::NO;
    }
    if (!a->is_contiguous() || !b->is_contiguous()) {
        return MemOverlapStatus::TOO_HARD;
    }
    if (a->storage().data() == b->storage().data()) {
      const auto a_begin = static_cast<char*>(a->data());
      const auto a_end = a_begin + a->numel() * static_cast<int64_t>(a->itemsize());
      const auto b_begin = static_cast<char*>(b->data());
      const auto b_end = b_begin + b->numel() * static_cast<int64_t>(b->itemsize());
      if (a_begin == b_begin && a_end == b_end) {
        return MemOverlapStatus::FULL;
      }
      if (a_begin < b_end && b_begin < a_end) {
        return MemOverlapStatus::PARTIAL;
      }
    }
    return MemOverlapStatus::NO;
}

void assert_no_partial_overlap(const at::Tensor& a, const at::Tensor& b) {
  assert_no_partial_overlap(a.unsafeGetTensorImpl(), b.unsafeGetTensorImpl());
}

void assert_no_partial_overlap(at::TensorImpl* a, at::TensorImpl* b) {
  TORCH_CHECK(get_overlap_status(a, b) != MemOverlapStatus::PARTIAL,
      "unsupported operation: some elements of the input tensor and "
      "the written-to tensor refer to a single memory location. "
      "Please clone() the tensor before performing the operation.");
}

}}