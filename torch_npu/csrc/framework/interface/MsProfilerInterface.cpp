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

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

namespace at_npu {
namespace native {

#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libmsprofiler, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)              \
  GET_FUNCTION(libmsprofiler, funcName)


REGISTER_LIBRARY(libmsprofiler)
LOAD_FUNCTION(aclprofSetConfig)


aclError AclprofSetConfig(aclprofConfigType configType, const char* config, size_t configLength) {
    typedef aclError(*AclprofSetConfigFunc)(aclprofConfigType, const char *, size_t);
    static AclprofSetConfigFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofSetConfigFunc)GET_FUNC(aclprofSetConfig);
        if (func == nullptr) {
            return ACL_ERROR_PROF_MODULES_UNSUPPORTED;
        }
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofSetConfig", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(configType, config, configLength);
}
}
}
