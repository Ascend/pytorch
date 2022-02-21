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

#pragma once

#include "torch_npu/csrc/framework/graph/execute/GraphExecutor.h"
#include "torch_npu/csrc/core/npu/NPURunMode.h"

namespace at_npu {
namespace native {
class GraphModeGuard {
public:
  GraphModeGuard() = delete;
  GraphModeGuard(const GraphModeGuard& other) = delete;
  GraphModeGuard(GraphModeGuard&& other) = delete;
  GraphModeGuard& operator=(const GraphModeGuard& other) = delete;
  GraphModeGuard& operator=(GraphModeGuard&& other) = delete;

  explicit GraphModeGuard(c10_npu::ModeKind mode) : mode_(mode) {
    ori_mode_ = c10_npu::NpuRunMode::IsGraphMode()
        ? c10_npu::ModeKind::GRAPH_MODE
        : c10_npu::ModeKind::SINGLE_OP_MODE;
    if ((ori_mode_ == c10_npu::ModeKind::GRAPH_MODE) &&
        (mode_ == c10_npu::ModeKind::SINGLE_OP_MODE)) {
      GraphExecutor::GetInstance().ConstructAndExecuteGraph();
    }
    c10_npu::NpuRunMode::SetNpuRunMode(mode_);
  }

  ~GraphModeGuard() {
    c10_npu::NpuRunMode::SetNpuRunMode(ori_mode_);
  }

private:
  c10_npu::ModeKind ori_mode_;
  c10_npu::ModeKind mode_;
};
} // namespace native
} // namespace at_npu