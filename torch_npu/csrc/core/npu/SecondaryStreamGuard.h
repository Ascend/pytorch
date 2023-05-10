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

#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/utils/NpuDataDumpMgr.h"

namespace c10_npu {
struct SecondaryStreamGuard {
  explicit SecondaryStreamGuard() = delete;
  // During datadump using one stream to run 
  // datadump used tdtchannel, queue does not support multiple streams
  explicit SecondaryStreamGuard(c10::Stream stream)
      : guard_(at_npu::native::NpuDataDumpMgr::GetInstance().IsDatadumpEnable()
                   ? c10_npu::getCurrentNPUStream()
                   : stream){};

  ~SecondaryStreamGuard() {
    c10_npu::NPUEvent npu_event;
    npu_event.record(guard_.current_stream());
    npu_event.block(guard_.original_stream());
  }

private:
  c10_npu::NPUStreamGuard guard_;
};
}  // namespace c10_npu
