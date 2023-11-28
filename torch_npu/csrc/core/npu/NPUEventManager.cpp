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

#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
namespace c10_npu {

NPUEventManager::NPUEventManager() : thread_pool_(std::make_shared<c10::TaskThreadPool>(5)) {};

NPUEventManager& NPUEventManager::GetInstance() {
  static NPUEventManager instance;
  return instance;
}

void NPUEventManager::run(aclrtEvent event) {
  int err = aclrtDestroyEvent(event);
  if (err != ACL_ERROR_NONE) {
      C10_NPU_SHOW_ERR_MSG();
      return;
  }
  ASCEND_LOGI("aclrtDestroyEvent is successfully executed.");
}

aclError NPUEventManager::QueryAndDestroyEvent() {
  std::lock_guard<std::mutex> guard(event_queue_mutex_);
  while (!npu_events_.empty())
  {
    aclrtEvent event = npu_events_.front();
    acl::aclrtEventRecordedStatus recordStatus = acl::ACL_EVENT_RECORDED_STATUS_NOT_READY;
    aclError err = acl::AclQueryEventRecordedStatus(event, &recordStatus);
    if (err != ACL_ERROR_NONE) {
      C10_NPU_SHOW_ERR_MSG();
      return err;
    }
    if (recordStatus != acl::ACL_EVENT_RECORDED_STATUS_COMPLETE) {
      break;
    } else {
      acl::aclrtEventWaitStatus waitStatus = acl::ACL_EVENT_WAIT_STATUS_RESERVED;
      // if the event usage is unknown, ensure the event id not destroyed in advance.
      aclError err_wait = acl::AclQueryEventWaitStatus(event, &waitStatus);
      if (err_wait != ACL_ERROR_NONE) {
        C10_NPU_SHOW_ERR_MSG();
        return err_wait;
      }
      if (waitStatus != acl::ACL_EVENT_WAIT_STATUS_COMPLETE) {
        break;
      }
    }
    {
      thread_pool_->run(std::bind(
          &NPUEventManager::run,
          this,
          event));
    }

    npu_events_.pop_front();
  }
  return ACL_ERROR_NONE;
}

aclError NPUEventManager::LazyDestroy(aclrtEvent npu_event) {
  std::lock_guard<std::mutex> guard(event_queue_mutex_);
  npu_events_.push_back(npu_event);
  return ACL_ERROR_NONE;
}

void NPUEventManager::ClearEvent() {
  if (thread_pool_ != nullptr) {
    thread_pool_->waitWorkComplete();
  }

  while (!npu_events_.empty()) {
    aclrtEvent event = npu_events_.front();
    auto err = aclrtDestroyEvent(event);
    if (err != ACL_ERROR_NONE) {
      NPU_CHECK_WARN(err);
    } else {
      ASCEND_LOGI("aclrtDestroyEvent is successfully executed.");
    }
    npu_events_.pop_front();
  }
}

} // namespace c10_npu
