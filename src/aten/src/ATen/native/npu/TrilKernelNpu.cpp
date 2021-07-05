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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace  at::native::npu;

Tensor& tril_out_npu(Tensor& result, const Tensor& self, int64_t diagonal){
  OpCommand cmd;
  cmd.Name("Tril")
      .Input(self)
      .Output(result)
      .Attr("diagonal", diagonal)
      .Run();
  return result;
}

Tensor tril_npu(const Tensor& self, int64_t diagonal){
  auto selfCopy = self.npu_format_cast(ACL_FORMAT_NCHW);
  auto is_last_two_dims = [&selfCopy](){
      auto selfStorage = selfCopy.storage().get_npu_desc().storage_sizes_;
      if (selfStorage.size() <= 1){
          return false;
      }
      return true;
  };
  
  TORCH_CHECK(is_last_two_dims(), "tril require tensor should be last two dims");
  Tensor result = OpPreparation::ApplyTensor(selfCopy);
  tril_out_npu(result, selfCopy, diagonal);
  return result;
}

Tensor& tril_npu_(Tensor& self, int64_t diagonal){
  OpPreparation::CheckMemory({self}, {self});  
  self.npu_format_cast_(ACL_FORMAT_NCHW);
  if(!NpuUtils::check_match(&self)){
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    tril_out_npu(contiguousSelf, contiguousSelf, diagonal);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    tril_out_npu(self, self, diagonal);
  }
  return self;
}

} // native
} // at
