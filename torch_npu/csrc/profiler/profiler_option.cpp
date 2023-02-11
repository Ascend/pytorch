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

#include <c10/util/Exception.h>

#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/profiler/cann_profiling.h"

namespace at_npu {
namespace native {
namespace env {

REGISTER_OPTION_HOOK(deliverswitch, [](const std::string &val) {
  if (val == "enable") {
    torch_npu::profiler::NpuProfilingDispatch::Instance().start();
  } else {
    torch_npu::profiler::NpuProfilingDispatch::Instance().stop();
  }
})

REGISTER_OPTION_HOOK(profilerResultPath, [](const std::string &val) {
  torch_npu::profiler::NpuProfiling::Instance().Init(val);
})

REGISTER_OPTION_HOOK(profiling, [](const std::string &val) {
  if (val.compare("stop") == 0) {
    torch_npu::profiler::NpuProfiling::NpuProfiling::Instance().Stop();
  } else if (val.compare("finalize") == 0) {
    torch_npu::profiler::NpuProfiling::NpuProfiling::Instance().Finalize();
  } else {
    TORCH_CHECK(false, "profiling input: (", val, " ) error!")
  }
})

}
}
}
