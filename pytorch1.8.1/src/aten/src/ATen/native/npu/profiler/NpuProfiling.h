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

#ifndef __NPU_PROFILING__
#define __NPU_PROFILING__

#include <c10/npu/interface/AclInterface.h>
#include <string>

enum PROFILING_STATUS {
  PROFILING_FINALIZE,
  PROFILING_INIT,
  PROFILING_START,
  PROFILING_STOP
};

namespace at {
namespace native {
namespace npu {

class NpuProfiling
{
public:
    static NpuProfiling& Instance();
    void Init(const std::string &profilerResultPath);
    void Start();
    void Stop();
    void Finalize();
private:
    aclprofConfig* profCfg = nullptr;
    PROFILING_STATUS status = PROFILING_FINALIZE;
    NpuProfiling() = default;
    ~NpuProfiling() = default;
};
}
}
}

#endif // __NPU_PROFILING__