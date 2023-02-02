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

class IndexingContiguousOpt : public ContiguousOpt {
public:
  bool Optimizer(at::Tensor &self, const at::Tensor &src,
                 const ContiguousTensorDesc &src_desc) override {
    c10::SmallVector<int64_t, MAX_DIM> start;
    c10::SmallVector<int64_t, MAX_DIM> end;
    c10::SmallVector<int64_t, MAX_DIM> step;

    if (can_use_indexing(src_desc, start, end, step)) {
      RECORD_FUNCTION("contiguous_d_StridedSlice", std::vector<c10::IValue>({src}));
      indexing_to_contiguous(self, src, start, end, step, src_desc);
      return true;
    }
    return false;
  }

private:
  bool can_use_indexing(const ContiguousTensorDesc &src_desc,
                        c10::SmallVector<int64_t, MAX_DIM> &start,
                        c10::SmallVector<int64_t, MAX_DIM> &end,
                        c10::SmallVector<int64_t, MAX_DIM> &step) {
    if (at::prod_intlist(src_desc.sizes_) >=
        at::prod_intlist(src_desc.base_sizes_)) {
      return false;
    }

    if (src_desc.sizes_.size() != src_desc.base_sizes_.size()) {
      return false;
    }
    if (src_desc.strides_.size() != src_desc.base_strides_.size()) {
      return false;
    }

    const auto &base_size = src_desc.base_sizes_;
    const auto &base_stride = src_desc.base_strides_;
    const auto &indexing_size = src_desc.sizes_;
    const auto &indexing_stride = src_desc.strides_;

    for (int64_t i = 0; i < indexing_size.size(); i++) {
      if ((indexing_stride[i] < base_stride[i]) || ((indexing_stride[i] % base_stride[i]) != 0)) {
        return false;
      }
    } 

    // indexing信息获取部分
    // Get step info(for indexing step at index aixs should > 1)
    for (int64_t i = 0; i < indexing_size.size(); i++) {
      TORCH_CHECK(base_stride[i] != 0, "stride should not be 0");
      step.emplace_back(indexing_stride[i] / base_stride[i]);
    }

    // Get start index based on offset and base stride
    int64_t src_offset = src_desc.offset_;
    for (int64_t i = 0; i < indexing_size.size(); i++) {
      TORCH_CHECK(base_stride[i] != 0, "stride should not be 0");
      start.emplace_back(src_offset / base_stride[i]);
      src_offset = src_offset % base_stride[i];
    }

    // infer end index
    for (int64_t i = 0; i < indexing_size.size(); i++) {
      int64_t calculate_end = start[i] + indexing_size[i] * step[i];
      if (calculate_end - step[i] > src_desc.base_sizes_[i]) {
        // Op StrideSlice(Slice) don't support span-axis indexing(slice).
        return false;
      }
      end.emplace_back(calculate_end);
    }

    // indexing场景判断: (1) step乘积>1(=1为slice);
    //                  (2) 当前规避最后一轴indexing,
    //                  因为stridedsliceD算子不支持; (3)
    //                  除去step!=1的轴，其他轴size，stride均与base_size,
    //                  base_stride相等(排除非关键轴reshape场景); (4)
    //                  对step!=1的轴，限制stride[i]=step[i]*size[i+1]*stride[i+1];(排除关键轴的reshape场景);
    //                  (5) 对step!=1的轴,
    //                  size(i)不可以为1:主要排除潜在的unsqueeze(0)+select(1,x)等走入indexing分支
    // case 1 & 2
    if (at::prod_intlist(step) == 1 || step[step.size() - 1] != 1) {
      return false;
    }
    // case 3
    for (int64_t i = 0; i < step.size(); i++) {
      if (step[i] == 1 && indexing_size[i] != base_size[i]) {
        return false;
      }
    }
    // case 4 and 5: step!=1的轴的校验
    for (int64_t i = 0; i < step.size() - 1; i++) {
      // 对于非最后一轴的indexing，对应的stride[i]=step[i]*size[i+1]*stride[i+1],（此时最后一轴stride限制为1）
      // 不满足上述条件，需要予以剔除，主要干扰：组合类reshape操作。
      if (step[i] != 1) {
        if (indexing_size[i] == 1) {
          return false;
        }
        if (step[i + 1] == 1 &&
            (indexing_stride[i] !=
             indexing_size[i + 1] * indexing_stride[i + 1] * step[i])) {
          return false;
        }
      }
    }
    return true;
  }

  void indexing_to_contiguous(at::Tensor &self, const at::Tensor &src,
                              c10::SmallVector<int64_t, MAX_DIM> &start,
                              c10::SmallVector<int64_t, MAX_DIM> &end,
                              c10::SmallVector<int64_t, MAX_DIM> &step,
                              const ContiguousTensorDesc &src_desc) {
    const auto &base_size = src_desc.base_sizes_;
    // recover contiguous base tensor
    at::Tensor temp_src = at::empty(src_desc.base_sizes_, src.options());
    temp_src.set_(src.storage(), temp_src.storage_offset(), temp_src.sizes(),
                  temp_src.strides());

    // call StridedSlice op
    NPUNativeFunctions::npu_indexing_out(temp_src, start, end, step, 0, 0, 0, 0,
                                         0, self);

    return;
  }
}; // class IndexingContiguousOpt

REGISTER_COPY_OPT(indexing, IndexingContiguousOpt)

} // namespace native
} // namespace at_npu
