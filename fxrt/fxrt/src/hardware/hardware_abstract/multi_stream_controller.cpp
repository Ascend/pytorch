#include "hardware/hardware_abstract/multi_stream_controller.h"

#include <algorithm>
#include <atomic>
#include "common/logger.h"

namespace fxrt {
namespace device {
constexpr size_t kDefaultStreamRefreshSize = 2;

namespace {
template <typename T>
struct AtomicWrapper {
  AtomicWrapper() : value_(0L) {}
  explicit AtomicWrapper(const std::atomic<T>& value) : value_(value.load()) {}
  AtomicWrapper(const AtomicWrapper& other) : value_(other.value_.load()) {}
  AtomicWrapper& operator=(const AtomicWrapper& other) {
    if (this == &other) {
      return *this;
    }
    value_.store(other.value_.load());
    return *this;
  }

  std::atomic<T> value_;
};

class LockGuard {
 public:
  explicit LockGuard(SpinLock& lock) : spin_lock_(lock) {
    spin_lock_.lock();
  }
  ~LockGuard() {
    spin_lock_.unlock();
  }

 private:
  SpinLock& spin_lock_;
};
} // namespace

class TaskIdOnStreamManager {
 public:
  TaskIdOnStreamManager() = default;

  void Resize(uint32_t stream_size) {
    if (initialized_ && stream_size <= initialize_size_) {
      RT_VLOG(VL_HARDWARE) << "Task id on stream manager has already initialized, current size : " << initialize_size_
                           << ".";
      return;
    }
    RT_VLOG(VL_HARDWARE) << "Task id on stream manager initialize : " << initialized_
                         << ", stream_size : " << stream_size << ".";
    uint32_t min_stream_size = 2;
    initialize_size_ = std::max(stream_size, min_stream_size);
    generator_.resize(initialize_size_);
    status_.resize(initialize_size_);
    for (auto& vec : status_) {
      vec.resize(initialize_size_);
    }
    initialized_ = true;
  }

  inline int64_t Query(uint32_t userStreamId, uint32_t memoryStreamId) {
    return status_[userStreamId][memoryStreamId];
  }

  inline bool Update(int64_t taskIdOnStream, uint32_t userStreamId, uint32_t memoryStreamId) {
    if (status_[userStreamId][memoryStreamId] >= taskIdOnStream) {
      return false;
    }
    status_[userStreamId][memoryStreamId] = taskIdOnStream;
    return true;
  }

  inline int64_t Launch(uint32_t streamId) {
    if (streamId >= generator_.size()) {
      RT_VLOG(VL_HARDWARE) << "Launch stream id : " << streamId << " failed, generator_ size : " << generator_.size();
      generator_.resize(streamId + 1);
      status_.resize(streamId + 1);
      for (auto& vec : status_) {
        vec.resize(streamId + 1);
      }
    }
    return ++generator_[streamId].value_;
  }

  inline int64_t Get(uint32_t streamId) {
    return generator_[streamId].value_;
  }

 private:
  bool initialized_{false};
  uint32_t initialize_size_{0};
  std::vector<AtomicWrapper<int64_t>> generator_;
  std::vector<std::vector<int64_t>> status_;
};

// Event pool recycled with ref count, pool will reuse event when cannot create more events.
class EventPool {
 public:
  explicit EventPool(std::function<DeviceEventPtr(void)> event_creator) : event_creator_(std::move(event_creator)) {}
  ~EventPool() {
    LockGuard lock(lock_);
    expired_ = true;
    events_.clear();
    cached_events_.clear();
  }

  EventPool() = delete;
  EventPool(const EventPool&) = delete;
  EventPool& operator=(const EventPool&) = delete;

  // Get event from pool, event was wrapper by shared_ptr.
  DeviceEventPtr Get() {
    RT_VLOG(VL_HARDWARE) << "Event pool get start.";
    LockGuard lock(lock_);
    DeviceEvent* event = nullptr;
    // Try to create event firstly before reached core size.
    if (size_ < core_size_) {
      auto created_event = event_creator_();
      if (created_event != nullptr && created_event->IsReady()) {
        cached_events_.push_back(created_event);
        size_++;
        event = created_event.get();
      } else {
        core_size_ = size_;
      }
    }
    // Try to reuse event.
    if (event == nullptr) {
      auto iter = events_.begin();
      while (iter != events_.end()) {
        auto event_in_list = *iter;
        if (event_in_list == nullptr) {
          RT_GLOG(ERROR) << "exception : event in list is nullptr, events_ size : " << events_.size() << ".";
        }
        if (event_in_list->QueryEvent()) {
          event = event_in_list;
          events_.erase(iter);
          break;
        }
        iter++;
      }
    }
    // Reuse failed, try to create more event.
    if (event == nullptr) {
      auto created_event = event_creator_();
      if (created_event != nullptr && created_event->IsReady()) {
        cached_events_.push_back(created_event);
        event = created_event.get();
        size_++;
      } else {
        RT_GLOG(ERROR) << "Get event failed.";
      }
    }
    RT_VLOG(VL_HARDWARE) << "Get event, events_ size : " << events_.size() << ", event : " << event << ".";

    auto event_ptr = std::shared_ptr<DeviceEvent>(event, [&](DeviceEvent* e) {
      LockGuard lock(lock_);
      if (!expired_) {
        RT_VLOG(VL_HARDWARE) << "Return event : " << e << ".";
        events_.push_back(e);
      } else {
        RT_VLOG(VL_HARDWARE) << "Return event : " << e << "failed.";
      }
    });
    return event_ptr;
  }

 private:
  SpinLock lock_;
  bool expired_{false};
  // Pool will just create event before reach core size, use half of size limits as core size.
  size_t core_size_{32768};
  size_t size_{0};
  std::function<DeviceEventPtr(void)> event_creator_;
  std::list<DeviceEvent*> events_;
  // cached_events_ hold shared ptr of event, since device res manager return a smart pointer.
  std::list<DeviceEventPtr> cached_events_;
};
using EventPoolPtr = std::shared_ptr<EventPool>;

MultiStreamController::MultiStreamController(DeviceResManager* deviceResBase) : deviceResBase_(deviceResBase) {
  if (deviceResBase_ == nullptr) {
    RT_GLOG(ERROR) << "deviceResBase_ is nullptr.";
  }
  taskIdOnStreamManager_ = std::make_shared<TaskIdOnStreamManager>();
}

void MultiStreamController::Refresh() {
  LockGuard lock(lock_);
  auto stream_size = deviceResBase_->QueryStreamSize();
  RT_VLOG(VL_HARDWARE) << "Stream manager initialize, stream_size : " << stream_size << ".";
  if (stream_size == 0) {
    // CPU has no concept of stream, stream size must be zero.
    RT_VLOG(VL_HARDWARE) << "Stream size is 0, will initialize with 2 streams.";
    stream_size = kDefaultStreamRefreshSize;
  }
  taskIdOnStreamManager_->Resize(stream_size);
  if (eventPool_ == nullptr) {
    eventPool_ = std::make_shared<EventPool>([&]() {
      // Event in pool need to do synchronization between streams, need to enable blocking.
      return deviceResBase_->CreateRuntimeEvent(true, false);
    });
  }
}

bool MultiStreamController::UpdateTaskIdOnStream(
    int64_t taskIdOnStream,
    uint32_t userStreamId,
    uint32_t memoryStreamId) {
  LockGuard lock(lock_);
  return taskIdOnStreamManager_->Update(taskIdOnStream, userStreamId, memoryStreamId);
}

int64_t MultiStreamController::QueryTaskIdOnStream(uint32_t userStreamId, uint32_t memoryStreamId) {
  LockGuard lock(lock_);
  return taskIdOnStreamManager_->Query(userStreamId, memoryStreamId);
}

int64_t MultiStreamController::LaunchTaskIdOnStream(uint32_t streamId) {
  LockGuard lock(lock_);
  return taskIdOnStreamManager_->Launch(streamId);
}

int64_t MultiStreamController::GetTaskIdOnStream(uint32_t streamId) {
  LockGuard lock(lock_);
  return taskIdOnStreamManager_->Get(streamId);
}

std::mutex& MultiStreamController::GetStreamMutex(size_t streamId) {
  LockGuard lock(lock_);
  return streamMutexes_[streamId];
}

bool MultiStreamController::RecordEvent(
    int64_t taskIdOnStream,
    uint32_t userStreamId,
    const std::vector<std::pair<uint32_t, DeviceMemPtr>>& memoryStreamAddresses,
    const DeviceEventPtr& inputEvent) {
  LockGuard lock(lock_);
  DeviceEventPtr event = nullptr;
  if (inputEvent != nullptr) {
    event = inputEvent;
  } else {
    event = deviceResBase_->CreateRuntimeEvent(false, true);
    if (event == nullptr) {
      return true;
    }
    event->RecordEvent(userStreamId);
  }

  return deviceResBase_->RecordEvent(taskIdOnStream, userStreamId, memoryStreamAddresses, event);
}

bool MultiStreamController::WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId, uint32_t memoryStreamId) {
  LockGuard lock(lock_);
  // If update task id on stream failed, means task id on stream is elder one, no need to wait event on mem manager.
  if (!taskIdOnStreamManager_->Update(taskIdOnStream, userStreamId, memoryStreamId)) {
    RT_VLOG(VL_HARDWARE) << "Skip Wait Event.";
    return false;
  }
  return deviceResBase_->WaitEvent(taskIdOnStream, userStreamId, memoryStreamId);
}

bool MultiStreamController::WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId) {
  LockGuard lock(lock_);
  return deviceResBase_->WaitEvent(taskIdOnStream, userStreamId);
}

bool MultiStreamController::DispatchRecordWaitEvent(uint32_t userStreamId, uint32_t memoryStreamId) {
  LockGuard lock(lock_);
  if (eventPool_ == nullptr) {
    RT_VLOG(VL_HARDWARE) << "Event pool is not initialized.";
    eventPool_ = std::make_shared<EventPool>([&]() {
      // Event in pool need to do synchronization between streams, need to enable blocking.
      return deviceResBase_->CreateRuntimeEvent(true, false);
    });
  }
  auto event = eventPool_->Get();
  // Note : record event on memory stream id and wait event on user stream id to make sure memory is safe.
  event->RecordEvent(memoryStreamId);
  event->WaitEvent(userStreamId);
  return true;
}

bool MultiStreamController::SyncStream(size_t streamId) {
  LockGuard lock(lock_);
  bool ret = deviceResBase_->SyncStream(streamId);
  auto taskIdOnStream = taskIdOnStreamManager_->Get(streamId);
  deviceResBase_->WaitEvent(taskIdOnStream, streamId);
  return ret;
}

bool MultiStreamController::SyncAllStreams() {
  LockGuard lock(lock_);
  bool ret = deviceResBase_->SyncAllStreams();
  deviceResBase_->SyncAllEvents();
  return ret;
}

bool MultiStreamController::SyncNotDefaultStreams() {
  LockGuard lock(lock_);
  bool ret = deviceResBase_->SyncNotDefaultStreams();
  const auto& streamIds = deviceResBase_->GetStreamIds();
  for (auto streamId : streamIds) {
    auto taskIdOnStream = taskIdOnStreamManager_->Get(streamId);
    deviceResBase_->WaitEvent(taskIdOnStream, streamId);
  }
  return ret;
}

bool MultiStreamController::WaitMultiStream(size_t wait_stream_id) {
  LockGuard lock(lock_);
  RT_VLOG(VL_HARDWARE) << "Wait multi stream on wait stream id : " << wait_stream_id << ".";
  const auto& streamIds = deviceResBase_->GetStreamIds();
  if (eventPool_ == nullptr) {
    RT_VLOG(VL_HARDWARE) << "Event pool is not initialized.";
    eventPool_ = std::make_shared<EventPool>([&]() {
      // Event in pool need to do synchronization between streams, need to enable blocking.
      return deviceResBase_->CreateRuntimeEvent(true, false);
    });
  }
  deviceResBase_->BindDeviceToCurrentThread(true);
  auto event = eventPool_->Get();
  for (auto streamId : streamIds) {
    if (streamId != wait_stream_id) {
      event->RecordEvent(streamId);
      event->WaitEvent(wait_stream_id);
    }
  }
  return true;
}
} // namespace device
} // namespace fxrt
