// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at_npu
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __PLUGIN_NPU_PROFILING_DISPATCH__
#define __PLUGIN_NPU_PROFILING_DISPATCH__

#include <c10/npu/interface/AclInterface.h>

namespace at_npu {
namespace native {

class NpuProfilingDispatch
{
public:
    static NpuProfilingDispatch& Instance();
    void start();
    void stop();
private:
    aclprofStepInfo* profStepInfo = nullptr;
    NpuProfilingDispatch() = default;
    ~NpuProfilingDispatch() = default;
    void init();
    void destroy();
};
}
}

#endif // __PLUGIN_NPU_PROFILING_DISPATCH__