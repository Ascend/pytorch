// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
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

#include <torch/extension.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include "torch_npu/csrc/core/npu/NPUFormat.h"

#include "torch_npu/csrc/framework/utils/OpPreparation.h"
// test   in  .setup with relative path
#include <tmp.h>

using namespace at;

Tensor tanh_add(Tensor x, Tensor y)
{
    return x.tanh() + y.tanh();
}

Tensor npu_add(const Tensor &self_, const Tensor &other_)
{
    TORCH_INTERNAL_ASSERT(torch_npu::utils::is_npu(self_));
    TORCH_INTERNAL_ASSERT(torch_npu::utils::is_npu(other_));
    return at::add(self_, other_, 1);
}

bool check_storage_sizes(const Tensor &tensor, const c10::IntArrayRef &sizes)
{
    auto tensor_sizes = at_npu::native::get_npu_storage_sizes(tensor);
    if (tensor_sizes.size() == sizes.size()) {
        return std::equal(tensor_sizes.begin(), tensor_sizes.end(), sizes.begin());
    }
    return false;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tanh_add", &tanh_add, "tanh(x) + tanh(y)");
    m.def("npu_add", &npu_add, "x + y");
    m.def("check_storage_sizes", &check_storage_sizes, "check_storage_sizes");
}
