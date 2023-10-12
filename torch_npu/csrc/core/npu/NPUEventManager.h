#pragma once

#include <deque>
#include <mutex>
#include <c10/core/thread_pool.h>
#include <c10/util/flat_hash_map.h>
#include <third_party/acl/inc/acl/acl.h>

#include "torch_npu/csrc/core/npu/NPUException.h"

#define ACL_EVENT_DEFAULT 0x0000000Eu

namespace c10_npu {

class NPUEventManager {
public:
  static NPUEventManager& GetInstance();
  aclError QueryAndDestroyEvent();
  aclError LazyDestroy(aclrtEvent npu_event);
  void ClearEvent();
  void IncreaseUnrecordedCount(aclrtEvent event);
  void DecreaseUnrecordedCount(aclrtEvent event);
  bool IsEventRecorded(aclrtEvent event);
  ~NPUEventManager() {}

private:
  void run(aclrtEvent event);

private:
  std::mutex event_queue_mutex_;
  NPUEventManager();
  std::deque<aclrtEvent> npu_events_;
  std::shared_ptr<c10::TaskThreadPool> thread_pool_;

  std::mutex event_unrecorded_count_mutex_;
  ska::flat_hash_map<aclrtEvent, int> event_unrecorded_count_;
};

} // namespace c10_npu