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

#include <c10/npu/NPUStream.h>
#include <ATen/native/npu/contiguous/ContiguousOpt.h>

namespace at {
namespace native {
namespace npu {

class BroadcastContiguousOpt : public ContiguousOpt {
public:
  bool Optimizer(const Tensor& src, Tensor& self) override {
    if (self.dim() != src.dim()) {
        return false;
    }

    if (can_use_broadcast(src)) {
      RECORD_FUNCTION("npuBroadcast", std::vector<c10::IValue>({src}));
      bool can_contiguous = broadcast_to_contiguous(src, self);
      return can_contiguous;
    }
    return false;
  }

private:
  bool can_use_broadcast(const Tensor& src) {
    bool can_use = false;
    for (int64_t i = 0; i < src.dim(); i++) {
      if (src.stride(i) == 0) {
        can_use = true;
        break;
      }
    }
    return can_use;
  }

  bool broadcast_to_contiguous(const Tensor& src, Tensor& self) {
    std::vector<int64_t> src_size(src.dim());
    for (int64_t i = 0; i < src.dim(); i++) {
      if (src.stride(i) == 0) {
          src_size[i] = 1;
      } else {
          src_size[i] = src.size(i);
      }
    }

    // create contiguous tensor for npu BroadcastToD
    Tensor temp_src = at::empty({0}, src.options());
    temp_src.set_(src);
    temp_src.unsafeGetTensorImpl()->set_sizes_and_strides(
        src_size, src.strides());

    c10::npu::NPUStream copy_stream = c10::npu::getCurrentNPUStream();
    if (temp_src.is_contiguous()) {
      auto temp_dst = at::npu_broadcast(temp_src, self.sizes());
      aclrtMemcpyAsync(
          self.data_ptr(),
          self.nbytes(),
          temp_dst.data_ptr(),
          self.nbytes(),
          ACL_MEMCPY_DEVICE_TO_DEVICE,
          copy_stream);
      return true;
    }
    return false;
  }

}; // class BroadcastContiguousOpt

REGISTER_COPY_OPT(broadcast, BroadcastContiguousOpt)

} // npu
} // native
} // at