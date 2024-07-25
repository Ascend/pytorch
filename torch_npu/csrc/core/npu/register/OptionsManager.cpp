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

#include <string>
#include <unistd.h>
#include <iomanip>
#include <fstream>
#include <Python.h>

#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/Exceptions.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/npu/memory_snapshot.h"

namespace c10_npu {
namespace option {

using namespace std;

bool OptionsManager::IsResumeModeEnable()
{
    const static bool isResumeModeEnable = []() -> bool {
        int32_t enable = OptionsManager::GetBoolTypeOption("RESUME_MODE_ENABLE", 0);
        return enable != 0;
    }();
    return isResumeModeEnable;
}

ReuseMode OptionsManager::GetMultiStreamMemoryReuse()
{
    const static ReuseMode reuseMode = []() -> ReuseMode {
        char *env_val = std::getenv("MULTI_STREAM_MEMORY_REUSE");
        int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : 1;
        ReuseMode mode = ERASE_RECORD_STREAM;
        switch (envFlag) {
            case 0:
                mode = CLOSE;
                break;
            case 1:
                mode = ERASE_RECORD_STREAM;
                break;
            case 2:
                mode = AVOID_RECORD_STREAM;
                break;
            default:
                TORCH_CHECK(false, "MULTI_STREAM_MEMORY_REUSE only allow 0, 1, 2", PTA_ERROR(ErrCode::VALUE));
        }
        return mode;
    }();
    return reuseMode;
}

bool OptionsManager::CheckInfNanModeEnable()
{
    const static bool checkInfNanModeEnable = []() -> bool {
        int32_t enable = OptionsManager::GetBoolTypeOption("INF_NAN_MODE_ENABLE", 1);
        return enable != 0;
    }();
    return checkInfNanModeEnable;
}

bool OptionsManager::CheckBlockingEnable()
{
    const static bool checkBlockingEnable = []() -> bool {
        int32_t blocking_enable = OptionsManager::GetBoolTypeOption("ASCEND_LAUNCH_BLOCKING", 0);
        return blocking_enable != 0;
    }();
    return checkBlockingEnable;
}

bool OptionsManager::CheckCombinedOptimizerEnable()
{
    const static bool checkCombinedOptimizerEnable = []() -> bool {
        int32_t combined_optimize = OptionsManager::GetBoolTypeOption("COMBINED_ENABLE");
        return combined_optimize != 0;
    }();
    return checkCombinedOptimizerEnable;
}

bool OptionsManager::CheckAclDumpDateEnable()
{
    const static bool checkAclDumpDateEnable = []() -> bool {
        int32_t acl_dump_data = OptionsManager::GetBoolTypeOption("ACL_DUMP_DATA");
        return acl_dump_data != 0;
    }();
    if (checkAclDumpDateEnable) {
        TORCH_NPU_WARN_ONCE(
            "The environment variable ACL_DUMP_DATA has been deprecated, "
            "please use torch_npu.npu.init_dump() instead");
    }
    return checkAclDumpDateEnable;
}

int OptionsManager::GetBoolTypeOption(const char* env_str, int defaultVal)
{
    char* env_val = std::getenv(env_str);
    int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : defaultVal;
    return (envFlag != 0) ? 1 : 0;
}

uint32_t OptionsManager::GetHCCLExecTimeout()
{
    char* env_val = std::getenv("HCCL_EXEC_TIMEOUT");
    int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : 0;
    return static_cast<uint32_t>(envFlag);
}

int32_t OptionsManager::GetACLExecTimeout()
{
    char* env_val = std::getenv("ACL_STREAM_TIMEOUT");
    int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : -1;
    return static_cast<int32_t>(envFlag);
}

uint32_t OptionsManager::CheckUseHcclAsyncErrorHandleEnable()
{
    char* asyncErrorHandling_val = std::getenv("HCCL_ASYNC_ERROR_HANDLING");
    int64_t asyncErrorHandlingFlag =
        (asyncErrorHandling_val != nullptr) ? strtol(asyncErrorHandling_val, nullptr, 10) : 0;
    return static_cast<uint32_t>(asyncErrorHandlingFlag);
}

uint32_t OptionsManager::CheckUseDesyncDebugEnable()
{
    char* desyncDebug_val = std::getenv("HCCL_DESYNC_DEBUG");
    int64_t desyncDebugFlag = (desyncDebug_val != nullptr) ? strtol(desyncDebug_val, nullptr, 10) : 0;
    return static_cast<uint32_t>(desyncDebugFlag);
}

int64_t OptionsManager::GetRankId()
{
    char* rankId_val = std::getenv("RANK");
    int64_t rankId = (rankId_val != nullptr) ? strtol(rankId_val, nullptr, 10) : -1;
    return rankId;
}

char *OptionsManager::GetNslbPath()
{
    return std::getenv("NSLB_CP");
}

uint32_t OptionsManager::GetNslbCntVal()
{
    const static uint32_t nslb_val = []() -> uint32_t {
        char* nslb_num = std::getenv("NSLB_MAX_RECORD_NUM");
        int64_t nslb_val = (nslb_num != nullptr) ? strtol(nslb_num, nullptr, 10) : 1000;
        return static_cast<uint32_t>(nslb_val);
    }();
    return nslb_val;
}

bool OptionsManager::CheckGeInitDisable()
{
    const static bool Check_Ge_Init_Disable = []() -> bool {
        int32_t ge_init_disable = OptionsManager::GetBoolTypeOption("GE_INIT_DISABLE");
        return ge_init_disable != 0;
    }();
    if (Check_Ge_Init_Disable) {
        TORCH_NPU_WARN_ONCE(
            "The environment variable GE_INIT_DISABLE has been enabled, "
            "this switch is only used for single operator simulation");
    }
    return Check_Ge_Init_Disable;
}

uint32_t OptionsManager::GetCpuAffinityConf()
{
    const static uint32_t cpu_affinity_conf = []() -> uint32_t {
        char* cpu_affinity_str = std::getenv("CPU_AFFINITY_CONF");
        int64_t cpu_affinity_conf = (cpu_affinity_str != nullptr) ? strtol(cpu_affinity_str, nullptr, 10) : 0;
        return static_cast<uint32_t>(cpu_affinity_conf);
    }();
    return cpu_affinity_conf;
}

uint32_t OptionsManager::GetTaskQueueEnable()
{
    if (CheckBlockingEnable()) {
        return 0;
    }
    const static uint32_t task_queue_enable = []() -> uint32_t {
        char* env_val = std::getenv("TASK_QUEUE_ENABLE");
        int64_t task_queue_enable = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : 1;
        return static_cast<uint32_t>(task_queue_enable);
    }();
    return task_queue_enable;
}

bool OptionsManager::CheckForceUncached()
{
    const static bool force_uncached = []() -> bool {
        bool force_uncached = OptionsManager::GetBoolTypeOption("PYTORCH_NO_NPU_MEMORY_CACHING");
        return force_uncached;
    }();
    return force_uncached;
}

std::string OptionsManager::GetOomSnapshotDumpPath()
{
    char* sanpshot_dump_path = std::getenv("OOM_SNAPSHOT_PATH");
    std::string dump_path = "./";
    if (sanpshot_dump_path != nullptr) {
        dump_path = std::string(sanpshot_dump_path);
    }
    char dump_abs_path[PATH_MAX] = {'\0'};
    if (realpath(dump_path.c_str(), dump_abs_path) == nullptr) {
        TORCH_CHECK(false, "`OOM_SNAPSHOT_PATH` is invalid.", PTA_ERROR(ErrCode::NOT_FOUND));
    }
    return dump_abs_path;
}

void oom_observer(int64_t device, int64_t allocated, int64_t device_total, int64_t device_free)
{
    auto dumppath = c10_npu::option::OptionsManager::GetOomSnapshotDumpPath();
    std::stringstream filename;
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm* timeInfo = std::localtime(&currentTime);
    filename << "oom_snapshot_" << getpid() << "_" << std::put_time(timeInfo, "%Y%m%d%H%M%S") << ".pickle";
    auto savefilepath = c10::str(dumppath, "/", filename.str());

    AutoGIL g;
    auto module = THPObjectPtr(PyImport_ImportModule("torch_npu.npu.memory"));
    if (!module) {
        throw python_error();
    }
    PyObject* p_func = PyObject_GetAttrString(module, "_dump_snapshot");
    if (!module) {
        throw python_error();
    }
    PyObject* p_args = PyTuple_New(1);
    PyTuple_SetItem(p_args, 0, PyUnicode_FromString(savefilepath.c_str()));
    PyObject* p_res = PyObject_CallObject(p_func, p_args);
}

void OptionsManager::IsOomSnapshotEnable()
{
    const static bool isOomSnapshotEnable = []() -> bool {
        int32_t enable = OptionsManager::GetBoolTypeOption("OOM_SNAPSHOT_ENABLE", 0);
        return enable != 0;
    }();
    if (isOomSnapshotEnable) {
        c10_npu::NPUCachingAllocator::attachOutOfMemoryObserver(std::move(oom_observer));
#ifndef BUILD_LIBTORCH
        torch_npu::_record_memory_history("all", "all", "python", UINT64_MAX);
#endif
    }
}

} // namespace option
} // namespace c10_npu
