// Copyright (c) 2022 Huawei Technologies Co., Ltd
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

#include <torch/csrc/utils/tensor_new.h>

#include "torch_npu/csrc/core/npu/NPUFunctions.h"

namespace torch_npu {
namespace utils {

// Initializes the Python tensor type objects: torch.npu.FloatTensor,
// torch.npu.DoubleTensor, etc. and binds them in their containing modules.
void _initialize_python_bindings();

PyMethodDef* npu_extension_functions();

}
}
