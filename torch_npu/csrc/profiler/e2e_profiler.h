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

#ifndef __TORCH_NPU_TOOLS_E2EPROFILER__
#define __TORCH_NPU_TOOLS_E2EPROFILER__

#include <third_party/acl/inc/acl/acl.h>
#include "torch_npu/csrc/core/npu/NPUException.h"
#include <chrono>
#include <sstream>
#include <thread>
#include <functional>
#include <ATen/record_function.h>


namespace torch_npu {
namespace profiler {

struct FileLineFunc {
    std::string filename;
    size_t line;
    std::string funcname;
};

void InitMsPorf(const std::string dump_path, uint64_t npu_event,
    uint64_t aicore_metrics);

void PushStartTime(at::RecordFunction& fn);
void PopEndTime(const at::RecordFunction& fn);

void InitE2eProfiler(const std::string dump_path,  uint64_t npu_event, uint64_t aicore_metrics, bool call_stack);

void FinalizeE2eProfiler();

std::vector<FileLineFunc> prepareCallstack(const std::vector<torch::jit::StackEntry> &cs);

std::vector<std::string> callstack2Str(const std::vector<FileLineFunc> &cs);
} // namespace profiler
} // namespace torch_npu

#endif // __TORCH_NPU_TOOLS_E2EPROFILER__