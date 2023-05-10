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