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

#include "torch_npu/csrc/core/npu/SecondaryStreamGuard.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"

namespace c10_npu {

SecondaryStreamGuard::~SecondaryStreamGuard()
{
    c10_npu::NPUEvent npu_event;
    npu_event.record(guard_.current_stream());
    npu_event.block(guard_.original_stream());
}

}  // namespace c10_npu
