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
#include "third_party/acl/inc/acl/acl_prof.h"

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
LOAD_FUNCTION(aclprofGetSupportedFeatures)
LOAD_FUNCTION(aclprofMarkEx)


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

aclError AclprofGetSupportedFeatures(size_t* featuresSize, void** featuresData)
{
    typedef aclError(*AclprofGetSupportedFeaturesFunc)(size_t*, void**);
    static AclprofGetSupportedFeaturesFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofGetSupportedFeaturesFunc)GET_FUNC(aclprofGetSupportedFeatures);
        if (func == nullptr) {
            return ACL_ERROR_PROF_MODULES_UNSUPPORTED;
        }
    }
    return func(featuresSize, featuresData);
}

aclError AclProfilingMarkEx(const char *msg, size_t msgLen, aclrtStream stream)
{
    typedef aclError (*aclprofMarkExFunc) (const char *, size_t, aclrtStream);
    static aclprofMarkExFunc func = nullptr;
    if (func == nullptr) {
        func = (aclprofMarkExFunc)GET_FUNC(aclprofMarkEx);
        if (func == nullptr) {
            return ACL_ERROR_PROF_MODULES_UNSUPPORTED;
        }
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofMarkEx", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(msg, msgLen, stream);
}

}
}
