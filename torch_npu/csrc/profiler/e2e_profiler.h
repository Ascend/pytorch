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
#include "torch_npu/csrc/profiler/profiler_legacy.h"
#include <list>
#include <chrono>
#include <sstream>
#include <thread>
#include <functional>
#include <ATen/record_function.h>

std::atomic<bool>& get_global_enable_profiling();

namespace torch_npu {
namespace profiler {
#define OP_NAME_LEN 128
#define STAMP_QUEUE_LEN 128     // mark stamp ring queue, must be 2^n
#define ONCE_REPORT_NUM 4       // mark stamp once report num, must be 2^(n-m)
#define GROUP_CAPACITY ONCE_REPORT_NUM
#define GROUP_NUM 40            // range stamp group num

struct Stamp {
    unsigned short magicNum;
    unsigned short dataTag;
    int processId;
    int threadId;
    int category;
    int eventType;
    int payloadType;
    int groupId;
    int nodeId;
    unsigned long long startTime;
    unsigned long long endTime;
    int messageType;
    char message[OP_NAME_LEN];
    char resv[76];
};

struct StampRingQueue {
    int idleNodeInd;
    struct Stamp *nodes;
};

struct StampGroup {
    int idleNodeInd;
    int fillEndNodeCnt;
    struct Stamp nodes[GROUP_CAPACITY];
};

struct StampGroupMng {
    int curUsedGrpInd;
    std::list<int> idleGrpInd;
    struct StampGroup *groups;
};

void InitRangeStamp();

struct Stamp *GetRangeStamp();

void PutRangeStamp(struct Stamp *stamp);

void FlushRangeStamp();

void UninitRangeStamp();

void InitMarkStamp();

void PutMarkStamp(const std::string &opName);

void FlushMarkStamp();

void UninitMarkStamp();

void InitMsPorf(const std::string dump_path, uint64_t npu_event,
    uint64_t aicore_metrics);

void PushStartTime(at::RecordFunction& fn);
void PopEndTime(const at::RecordFunction& fn);

void InitE2eProfiler(const std::string dump_path,  uint64_t npu_event, uint64_t aicore_metrics, bool call_stack);

void FinalizeE2eProfiler();

void MarkQueueStamp(uint32_t category, const std::string &op_name);

void MarkQueueStamp(uint32_t category, void *data, size_t offset);

std::vector<FileLineFunc> prepareCallstack(const std::vector<torch::jit::StackEntry> &cs);

std::vector<std::string> callstack2Str(const std::vector<FileLineFunc> &cs);
} // namespace profiler
} // namespace torch_npu

#endif // __TORCH_NPU_TOOLS_E2EPROFILER__