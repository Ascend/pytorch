// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/framework/interface/HcclInterface.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

namespace at_npu {
namespace native {
namespace hccl {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libhccl, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName) \
  GET_FUNCTION(libhccl, funcName)

REGISTER_LIBRARY(libhccl)
LOAD_FUNCTION(HcclSetConfig)
LOAD_FUNCTION(HcclGetCommName)

extern HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue) {
    typedef HcclResult (*HcclSetConfigFunc)(HcclConfig config, HcclConfigValue configValue);
    static HcclSetConfigFunc func = nullptr;
    if (func == nullptr)
    {
        func = (HcclSetConfigFunc)GET_FUNC(HcclSetConfig);
    }
    if (func == nullptr) {
        TORCH_WARN("Failed to find this HcclSetConfig function, get real hccl config, need to upgrade hccl version!");
        return HcclResult::HCCL_SUCCESS;
    }
    return func(config, configValue);
}

extern HcclResult HcclGetCommNameFace(HcclComm commHandle, char* commName) {
    typedef HcclResult (*HcclGetCommNameFace)(HcclComm commHandle, char* commName);
    static HcclGetCommNameFace func = nullptr;
    if (func == nullptr) {
        func = (HcclGetCommNameFace)GET_FUNC(HcclGetCommName);
    }
    TORCH_CHECK(func, "Failed to find function HcclGetCommName,"
                " maybe you cann version is too low, please upgrade it");
    return func(commHandle, commName);
}
} // namespace hccl
} // namespace native
} // namespace at_npu