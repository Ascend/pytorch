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

#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"

namespace at_npu {
namespace native {

class SelectContiguousOpt : public ContiguousOpt {
public:
  bool Optimizer(at::Tensor &self, const at::Tensor &src,
                 const ContiguousTensorDesc &src_desc) override {
    // select(dim, start), length[dim] == 1
    c10::SmallVector<int64_t, MAX_DIM> start;
    c10::SmallVector<int64_t, MAX_DIM> length;

    if (can_use_select(src_desc, start, length)) {
       RECORD_FUNCTION("contiguous_d_StridedSlice",
                      std::vector<c10::IValue>({src}));
      select_to_contiguous(self, src, start, length, src_desc);
      return true;
    }
    return false;
  }

  bool CanOptimizer(const ContiguousTensorDesc &src_desc) override {
    c10::SmallVector<int64_t, MAX_DIM> start;
    c10::SmallVector<int64_t, MAX_DIM> length;
    return can_use_select(src_desc, start, length);
  }

private:
  bool can_use_select(const ContiguousTensorDesc &src_desc,
                      c10::SmallVector<int64_t, MAX_DIM> &start,
                      c10::SmallVector<int64_t, MAX_DIM> &length) {
    // base info and src info
    const auto &base_size = src_desc.base_sizes_;
    const auto &base_stride = src_desc.base_strides_;
    const auto &select_size = src_desc.sizes_;
    const auto &select_stride = src_desc.strides_;

    // len(base_size) - len(select_size) == 1  && len(base_stride) -
    // len(select_stride) == 1
    if ((base_size.size() - select_size.size() != 1) ||
        (base_stride.size() - select_stride.size() != 1)) {
      return false;
    }

    // recover src tensor info: shape and stride
    c10::SmallVector<int64_t, MAX_DIM> temp_size;
    c10::SmallVector<int64_t, MAX_DIM> temp_stride;
    for (int64_t i = 0; i <= select_size.size(); i++) {
      if (base_size[i] != select_size[i] ||
          base_stride[i] != select_stride[i]) {
        temp_size.emplace_back(base_size[i]);
        temp_stride.emplace_back(base_stride[i]);
        for (int64_t j = i + 1; j <= select_size.size(); j++) {
          temp_size.emplace_back(select_size[j - 1]);
          temp_stride.emplace_back(select_stride[j - 1]);
          i = j + 1;
        }
      } else {
        temp_size.emplace_back(select_size[i]);
        temp_stride.emplace_back(select_stride[i]);
      }
    }

    for (int64_t i = 0; i <= select_size.size(); i++) {
      if (base_size[i] == temp_size[i] && base_stride[i] == temp_stride[i]) {
        continue;
      } else {
        return false;
      }
    }

    // Collect the select infos for SliceD: dim, start, length
    // confirm the selected dim
    int64_t dim = base_size.size() - 1;
    for (int64_t i = 0; i < select_size.size(); i++) {
      if (base_size[i] != select_size[i] ||
          base_stride[i] != select_stride[i]) {
        dim = i;
        break;
      }
    }

    // Obtain start index and select length
    int64_t int_index = src_desc.offset_ / base_stride[dim];
    for (int64_t i = 0; i < base_size.size(); i++) {
      if (i == dim) {
        start.emplace_back(int_index);
        length.emplace_back(1);
      } else {
        start.emplace_back(0);
        length.emplace_back(base_size[i]);
      }
    }
    return true;
  }

  void select_to_contiguous(at::Tensor &self, const at::Tensor &src,
                            c10::SmallVector<int64_t, MAX_DIM> &start,
                            c10::SmallVector<int64_t, MAX_DIM> &length,
                            const ContiguousTensorDesc &src_desc) {
    const auto &base_size = src_desc.base_sizes_;
    // Recover base tensor(necessary) a = b.select(1, 1)
    at::Tensor temp_src = at::empty(base_size, src.options());
    temp_src.set_(src.storage(), temp_src.storage_offset(), temp_src.sizes(),
                  temp_src.strides());

    // construct StridedSlice param
    auto axis_size = start.size();
    c10::SmallVector<int64_t, MAX_DIM> strides(axis_size, 1);
    c10::SmallVector<int64_t, MAX_DIM> end;
    int64_t shrink_mask = 0;
    for (auto i = 0; i < axis_size; ++i) {
      end.emplace_back(start[i] + length[i]);
      if (length[i] == 1 && temp_src.size(i) != 1) {
        shrink_mask += std::pow(2, i);
      }
    }

    // call StridedSlice op to contiguous
    NPUNativeFunctions::npu_indexing_out(temp_src, start, end, strides, 0, 0, 0,
                                         0, shrink_mask, self);
    return;
  }
}; // class SelectContiguousOpt

REGISTER_COPY_OPT(select, SelectContiguousOpt)

} // namespace native
} // namespace at_npu