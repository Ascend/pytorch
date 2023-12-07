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

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace at_npu {
namespace native {

class BroadcastContiguousOpt : public ContiguousOpt {
public:
  bool Optimizer(at::Tensor &self, const at::Tensor &src,
                 const ContiguousTensorDesc &src_desc) override {
    if (self.dim() != src.dim()) {
      return false;
    }

    if (can_use_broadcast(src_desc)) {
      RECORD_FUNCTION("contiguous_d_BroadcastTo", std::vector<c10::IValue>({src}));
      bool can_contiguous = broadcast_to_contiguous(self, src, src_desc);
      return can_contiguous;
    }
    return false;
  }

private:
  bool can_use_broadcast(const ContiguousTensorDesc &src_desc) {
    // Reshape is used to process dimension addition cases for expand/expand_as.
    // Here, dimension expansion cases of expand/expand_as are processed.
    const auto &base_sizes = src_desc.base_sizes_;
    const auto &base_strides = src_desc.base_strides_;
    const auto &view_sizes = src_desc.sizes_;
    const auto &view_strides = src_desc.strides_;

    // The new ones will be appended at the front.
    // Any dimension of size 1 can be expanded to an arbitrary value.
    auto base_dim = static_cast<int64_t>(base_sizes.size());
    auto view_dim = static_cast<int64_t>(view_sizes.size());
    auto expand_dims = view_dim - base_dim;
    if (expand_dims < 0) {
      return false;
    }

    bool has_zero_in_stride = false;
    for (int64_t i = 0; i < base_dim; i++) {
      if (view_strides[i + expand_dims] == 0) {
        has_zero_in_stride = true;
        if (base_sizes[i] != 1 || view_sizes[i + expand_dims] == 1) {
          return false;
        }
      } else {
        if (view_sizes[i + expand_dims] != base_sizes[i] ||
            view_strides[i + expand_dims] != base_strides[i]) {
          return false;
        }
      }
    }

    for (auto i = 0; i < expand_dims; i++) {
      if (view_sizes[i] != 1 && view_strides[i] != 0) {
        return false;
      }
      has_zero_in_stride = true;
    }
    return has_zero_in_stride;
  }

  bool broadcast_to_contiguous(at::Tensor &self, const at::Tensor &src,
                               const ContiguousTensorDesc &src_desc) {
    std::vector<int64_t> src_size(src.dim());
    for (const auto i : c10::irange(src_desc.sizes_.size())) {
      if (src_desc.strides_[i] == 0) {
        src_size[i] = 1;
      } else {
        src_size[i] = src_desc.sizes_[i];
      }
    }

    // create contiguous tensor for npu BroadcastToD
    at::Tensor temp_src = at::empty({0}, src.options());
    temp_src.set_(src);
    temp_src.unsafeGetTensorImpl()->set_sizes_and_strides(src_size,
                                                          src.strides());

    if (temp_src.is_contiguous()) {
      // NPU op BroadcastTo not supports dtype of bool yet.
      if (self.dtype() == at::kBool) {
        auto temp_dst = custom_ops::npu_broadcast(temp_src, self.sizes());
        // The current logic is only used in single op mode.
        c10_npu::queue::LaunchAsyncCopyTask(self.data_ptr(),
                                            self.nbytes(),
                                            temp_dst.data_ptr(),
                                            self.nbytes(),
                                            ACL_MEMCPY_DEVICE_TO_DEVICE);
        return true;
      }
      custom_ops::npu_broadcast_out(temp_src, self.sizes(), self);
      return true;
    }
    return false;
  }
}; // class BroadcastContiguousOpt

REGISTER_COPY_OPT(broadcast, BroadcastContiguousOpt)

} // namespace native
} // namespace at_npu