#pragma once

#include "torch_npu/csrc/core/npu/NPUException.h"
#include <third_party/acl/inc/acl/acl.h>
#include <deque>
#include <mutex>
namespace c10_npu {

class NPUEventManager {
public:
  static NPUEventManager& GetInstance();
  aclError QueryAndDestroyEvent();
  aclError LazyDestroy(aclrtEvent npu_event);
  aclError ClearEvent();
  ~NPUEventManager(){}

private:
  std::mutex event_queue_mutex_;
  NPUEventManager();
  std::deque<aclrtEvent> npu_events_;
};

} // namespace c10_npu