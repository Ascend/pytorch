// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/npu/NpuSleep.h"

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/utils/LazyInit.h"
#include "third_party/op-plugin/op_plugin/utils/op_api_common.h"

namespace c10_npu {

void npu_sleep(int64_t cycles) {
  TORCH_CHECK(cycles > 0, "torch.npu._sleep(): expected positive cycles, got ", cycles);

  int64_t cycles_val = cycles;
  at::IntArrayRef cycles_arr(cycles_val);
  EXEC_NPU_CMD(aclnnSleep, cycles_arr);
}

} // namespace c10_npu