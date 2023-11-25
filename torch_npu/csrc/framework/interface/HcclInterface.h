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

#include "third_party/hccl/inc/hccl/hccl.h"

namespace at_npu {
namespace native {
namespace hccl {
/**
 * @ingroup AscendCL
 * @brief set hccl config option value
 *
 * @param config [IN]      hccl set config type
 * @param configValue [IN]   hccl set config value
 *
 * @return HcclResult
 */
extern HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue);

/**
 * @ingroup AscendCL
 * @brief get hccl comm name
 *
 * @param commHandle [IN]    query hccl commHandle
 * @param commName [OUT]     hccl come name
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
extern HcclResult HcclGetCommNameFace(HcclComm commHandle, char* commName);
} // namespace hccl
} // namespace native
} // namespace at_npu