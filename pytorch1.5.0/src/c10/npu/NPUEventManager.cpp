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

#include "NPUEventManager.h"
namespace c10 {
namespace npu {
NPUEventManager::NPUEventManager() {};

NPUEventManager& NPUEventManager::GetInstance() {
  static NPUEventManager instance;
  return instance;
}

aclError NPUEventManager::LazyDestroy(aclrtEvent npu_event) {
  while (!npu_events_.empty())
  {
    aclrtEvent event = npu_events_.front();
    acl::aclrtEventWaitStatus waitStatus = acl::ACL_EVENT_WAIT_STATUS_RESERVED;
    aclrtEventStatus recordStatus = ACL_EVENT_STATUS_RESERVED;
    aclError err = acl::AclQueryEventStatus(event, &waitStatus, &recordStatus);
    if (err != ACL_ERROR_NONE) {
        return err;
    }
    if ((waitStatus != acl::ACL_EVENT_WAIT_STATUS_COMPLETE) &&
      (recordStatus != ACL_EVENT_STATUS_COMPLETE)) {
      break;
    }
    err = aclrtDestroyEvent(event);
    if (err != ACL_ERROR_NONE) {
        C10_NPU_SHOW_ERR_MSG();
        return err;
    }
    npu_events_.pop_front();
  }
  npu_events_.push_back(npu_event);
  return ACL_ERROR_NONE;
}

aclError NPUEventManager::ClearEvent() {
  while(!npu_events_.empty()) {
    aclrtEvent event = npu_events_.front();
    C10_NPU_CHECK(aclrtDestroyEvent(event));
    npu_events_.pop_front();
  }
  return ACL_ERROR_NONE;
}
} // namespace NPU
} // namespace c10