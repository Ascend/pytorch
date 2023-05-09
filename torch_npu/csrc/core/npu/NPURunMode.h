#pragma once
#include <c10/macros/Export.h>

#include <string>

namespace c10_npu {
enum class ModeKind : uint8_t {
  DEFAULT_MODE = 0,
  SINGLE_OP_MODE = DEFAULT_MODE,
  GRAPH_MODE,
  REPLAY_MODE,
};

class  NpuRunMode{
public:
  static void SetNpuRunMode(const ModeKind& mode);
  static ModeKind CurRunMode();
  static inline bool IsGraphMode() {return is_graph_mode;};

private:
  static ModeKind cur_mode_;
  static bool is_graph_mode;
};
} // namespace c10_npu