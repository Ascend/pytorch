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

#include <third_party/hccl/inc/hccl/hccl_types.h>
#include <third_party/acl/inc/acl/acl.h>

namespace c10 {
namespace npu {
namespace hccl {

/**
 * @brief Load HcclBarrier API.
 *
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult 
 */
HcclResult hccl_barrier(HcclComm comm, aclrtStream stream);

} // namespace hccl
} // namespace npu
} // namespace c10