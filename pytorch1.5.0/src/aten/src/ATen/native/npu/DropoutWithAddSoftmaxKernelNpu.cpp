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
#include "c10/npu/NPUCachingAllocator.h"
#include "c10/npu/SecondaryStreamGuard.h"
#include <ATen/npu/NPUGenerator.h>

namespace at {
namespace native {
using namespace at::native::npu;

Tensor dropout_genmask(const Tensor& self, Scalar prob){
  uint32_t length = (self.numel() + 128 - 1) / 128 * 128;
  Tensor mask = OpPreparation::ApplyTensorWithFormat(
      {length},
      self.options().dtype(at::kByte),
      ACL_FORMAT_ND);
  IntArrayRef selfShape = self.sizes();

  int64_t seed = 2;
  int64_t seed2 = 0;
  OpCommand cmd;
  cmd.Name("DropOutGenMaskV3")
      .Input(selfShape)
      .Input(prob, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(mask)
      .Attr("seed", seed)
      .Attr("seed2", seed2)
      .Run();
  return mask;
}

std::tuple<Tensor, Tensor, Tensor> dropout_with_add_softmax_npu(
    const Tensor& self,
    const Tensor& x1,
    Scalar alpha,
    double p,
    int64_t dim){
  Tensor result_softmax = OpPreparation::ApplyTensor(x1);
  Tensor result_dropout = OpPreparation::ApplyTensor(self);
  SmallVector<int64_t, N> dimList = {dim};
  double retain = 1. - p;
  Scalar prob = Scalar(retain);
  Tensor mask;
  auto original_stream = c10::npu::getCurrentNPUStream();
  {
    c10::npu::SecondaryStreamGuard guard(c10::npu::getCurrentSecondaryStream());
    mask = dropout_genmask(x1, prob);
  }
  c10::npu::NPUCachingAllocator::recordStream(mask.storage().data_ptr(), original_stream);

  OpCommand cmd;
  cmd.Name("AxpyWithSoftmaxAndDropOutDoMask")
     .Input(x1)
     .Input(self)
     .Input(mask)
     .Output(result_softmax)
     .Output(result_dropout)
     .Attr("alpha", alpha)
     .Attr("input_keep_prob", prob)
     .Attr("axis", dimList)
     .Run();
   return std::tie(mask, result_softmax, result_dropout);
}

}
}
