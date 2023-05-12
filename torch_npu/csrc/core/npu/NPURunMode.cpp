#include "NPURunMode.h"

namespace c10_npu {
ModeKind NpuRunMode::cur_mode_ = ModeKind::DEFAULT_MODE;
bool NpuRunMode::is_graph_mode = false;

void NpuRunMode::SetNpuRunMode(const ModeKind &mode) {
  cur_mode_ = mode;
  is_graph_mode = (mode == ModeKind::SINGLE_OP_MODE) ? false : true;
  return;
}

ModeKind NpuRunMode::CurRunMode() {
  return cur_mode_;
}
} // namespace c10_npu