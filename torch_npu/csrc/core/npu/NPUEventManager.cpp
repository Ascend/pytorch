#include "torch_npu/csrc/core/npu/NPUEventManager.h"
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
}

aclError NPUEventManager::QueryAndDestroyEvent() {
  std::lock_guard<std::mutex> guard(event_queue_mutex_);
  while (!npu_events_.empty())
  {
    aclrtEvent event = npu_events_.front();
    acl::aclrtEventWaitStatus waitStatus = acl::ACL_EVENT_WAIT_STATUS_RESERVED;
    aclrtEventStatus recordStatus = ACL_EVENT_STATUS_RESERVED;
    aclError err = acl::AclQueryEventStatus(event, &waitStatus, &recordStatus);
    if (err != ACL_ERROR_NONE) {
      C10_NPU_SHOW_ERR_MSG();
      return err;
    }
    if ((waitStatus != acl::ACL_EVENT_WAIT_STATUS_COMPLETE) &&
      (recordStatus != ACL_EVENT_STATUS_COMPLETE)) {
      break;
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

aclError NPUEventManager::ClearEvent() {

  if (thread_pool_ != nullptr) {
    thread_pool_->waitWorkComplete();
  }

  while(!npu_events_.empty()) {
    aclrtEvent event = npu_events_.front();
    C10_NPU_CHECK(aclrtDestroyEvent(event));
    npu_events_.pop_front();
  }

  return ACL_ERROR_NONE;
}

} // namespace c10_npu