// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include "NPURunMode.h"

namespace c10_npu {
ModeKind NpuRunMode::cur_mode_ = ModeKind::DEFAULT_MODE;

void NpuRunMode::SetNpuRunMode(const ModeKind &mode) {
  cur_mode_ = mode;
  return;
}

ModeKind NpuRunMode::CurRunMode() {
  return cur_mode_;
}

bool NpuRunMode::IsGraphMode() {
  return cur_mode_ == ModeKind::GRAPH_MODE;
}
} // namespace c10