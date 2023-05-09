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