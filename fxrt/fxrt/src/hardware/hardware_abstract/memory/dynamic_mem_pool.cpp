#include "hardware/hardware_abstract/memory/dynamic_mem_pool.h"

#include <numeric>
#include <ostream>
#include "common/logger.h"

namespace fxrt {
namespace device {
static thread_local AllocatorDebugInfo debugInfo_;

AllocatorDebugInfo& DynamicMemAllocatorDebugInfo::GetDebugInfo() noexcept {
  return debugInfo_;
}

// Set the debug info when memory alloc.
void DynamicMemAllocatorDebugInfo::SetDebugInfo(
    const std::string& name,
    memory::mem_pool::MemType type,
    int inputIndex,
    int outputIndex,
    uint8_t runMode) {
  debugInfo_.name_ = name;
  debugInfo_.type_ = type;
  debugInfo_.inputIndex_ = inputIndex;
  debugInfo_.outputIndex_ = outputIndex;
  debugInfo_.runMode_ = runMode;
}

static const std::map<DynamicMemBufStatus, std::string> kBufStatusString = {
    {DynamicMemBufStatus::kMemBufIdle, "idle"},
    {DynamicMemBufStatus::kMemBufUsed, "used"},
    {DynamicMemBufStatus::kMemBufEagerFree, "eager_free"},
    {DynamicMemBufStatus::kMemBufUsedByEvent, "used_by_event"}};

const std::string& DynamicMemBufStatusToString(DynamicMemBufStatus status) {
  return kBufStatusString.at(status);
}

bool EventBase::RecordEvent(int64_t taskIdOnStream, uint32_t userStreamId, const DeviceEventPtr& event) {
  if (event == nullptr) {
    RT_GLOG(ERROR) << "Event is null.";
  }
  if (events_ == nullptr) {
    events_ = std::make_shared<std::unordered_map<uint32_t, std::shared_ptr<std::list<TaskIdOnStreamEvent>>>>();
  }
  std::shared_ptr<std::list<TaskIdOnStreamEvent>> eventList = nullptr;
  auto iter = events_->find(userStreamId);
  if (iter == events_->end()) {
    eventList = std::make_shared<std::list<TaskIdOnStreamEvent>>();
    (void)events_->emplace(userStreamId, eventList);
  } else {
    eventList = iter->second;
    if (eventList == nullptr) {
      RT_GLOG(ERROR) << "Event list is null.";
    }
  }
  (void)eventList->emplace_back(taskIdOnStream, event);
  return true;
}

bool EventBase::WaitEvent(uint32_t taskIdOnStream, uint32_t userStreamId) {
  if (events_ == nullptr) {
    return false;
  }
  auto iter = events_->find(userStreamId);
  if (iter == events_->end()) {
    return false;
  }
  auto& eventList = iter->second;
  if (eventList == nullptr) {
    RT_GLOG(ERROR) << "Event list is null.";
  }
  // Pop all element in list that not bigger than taskIdOnStream.
  while (!eventList->empty() && eventList->front().first <= taskIdOnStream) {
    eventList->pop_front();
  }
  // Remove list if event list is empty.
  if (eventList->empty()) {
    events_->erase(iter);
  }
  return true;
}

bool EventBase::IsEventNotUsed() {
  return events_ == nullptr ? true : events_->empty();
}

bool EventBase::SyncAllEvents() {
  if (IsEventNotUsed()) {
    return false;
  }

  for (auto iter = events_->begin(); iter != events_->end();) {
    auto& eventList = iter->second;
    if (eventList == nullptr) {
      RT_GLOG(ERROR) << "Event list is null.";
    }
    for (auto listIter = eventList->begin(); listIter != eventList->end();) {
      auto& event = listIter->second;
      // Sync event if event is not arrived.
      if (!event->QueryEvent()) {
        event->SyncEvent();
      }
      listIter = eventList->erase(listIter);
    }
    if (eventList->empty()) {
      // list is empty, erase list in map.
      iter = events_->erase(iter);
    } else {
      RT_GLOG(ERROR) << "Event list is not empty.";
    }
  }
  return events_->empty();
}
} // namespace device
} // namespace fxrt
