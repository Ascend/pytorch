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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace at_npu {
namespace native {
c10::IntArrayRef input_same_output_size(const at::Tensor &input)
{
    return input.sizes();
}

c10::SmallVector<int64_t, SIZE> replication_pad2d_npu_output_size(const at::Tensor &self, c10::IntArrayRef padding)
{
    TORCH_CHECK(self.dim() == 3 || self.dim() == 4, "tensor self's dimension must be 3 or 4", OPS_ERROR(ErrCode::PARAM));
    int64_t N = self.dim() == 3 ? 1 : self.size(-4);
    int64_t C = self.size(-3);
    int64_t H = self.size(-2);
    int64_t W = self.size(-1);
    int64_t padding_l = 0;
    int64_t padding_r = 0;
    int64_t padding_t = 0;
    int64_t padding_b = 0;
    if (!padding.empty() && padding.size() == 1) {
        padding_l = padding[0];
        padding_r = padding[0];
        padding_t = padding[0];
        padding_b = padding[0];
    } else if (!padding.empty() && 4 == padding.size()) {
        padding_l = padding[0];
        padding_r = padding[1];
        padding_t = padding[2];
        padding_b = padding[3];
    }
    int64_t Ho = H + padding_t + padding_b;
    int64_t Wo = W + padding_l + padding_r;

    c10::SmallVector<int64_t, SIZE> outputSize = {N, C, Ho, Wo};
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape)
{
    c10::SmallVector<int64_t, SIZE> shape_small_vec;
    for (uint64_t i = 0; i < shape.size(); i++) {
        shape_small_vec.emplace_back(shape[i]);
    }
    return shape_small_vec;
}
}  // namespace native
}  // namespace at_npu