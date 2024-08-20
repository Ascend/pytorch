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

#pragma once

#include <c10/util/Exception.h>
#include <map>
#include <string>
#include <unordered_map>

namespace c10_npu {
namespace option {

enum ReuseMode {
    CLOSE = 0,
    ERASE_RECORD_STREAM = 1,
    AVOID_RECORD_STREAM = 2,
};

enum SilenceCheckMode {
    CHECK_CLOSE = 0,
    PRINT_WARN_LOG = 1,
    REPORT_ALARM = 2,
    PRINT_ALL_LOG = 3,
};
class OptionsManager {
public:
    static bool IsResumeModeEnable();
    static ReuseMode GetMultiStreamMemoryReuse();
    static bool CheckInfNanModeEnable();
    static bool CheckBlockingEnable();
    static bool CheckCombinedOptimizerEnable();
    static bool CheckTriCombinedOptimizerEnable();
    static bool CheckAclDumpDateEnable();
    static uint32_t GetHCCLExecTimeout();
    static uint32_t CheckUseHcclAsyncErrorHandleEnable();
    static uint32_t CheckUseDesyncDebugEnable();
    static std::string CheckDisableDynamicPath();
    static int32_t GetACLExecTimeout();
    static int64_t GetRankId();
    static char *GetNslbPath();
    static uint32_t GetNslbCntVal();
    static uint32_t GetSilenceCheckFlag();
    static std::pair<double, double> GetSilenceUpperThresh();
    static std::pair<double, double> GetSilenceSigmaThresh();
    static bool CheckGeInitDisable();
    static uint32_t GetTaskQueueEnable();
    static uint32_t GetCpuAffinityConf();
    static std::string GetOomSnapshotDumpPath();
    static void IsOomSnapshotEnable();

private:
    static int GetBoolTypeOption(const char* env_str, int defaultVal = 0);
    static std::vector<std::string> Split(const std::string& input, char delimiter);
    static std::pair<double, double> GetSilenceThresh(const std::string& env_str,
        std::pair<double, double> defaultThresh);
};

} // namespace option
} // namespace c10_npu
