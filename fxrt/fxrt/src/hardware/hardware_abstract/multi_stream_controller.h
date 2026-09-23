#ifndef FXRT_SRC_HARDWARE_MULTI_STREAM_CONTROLLER_HEADER_H
#define FXRT_SRC_HARDWARE_MULTI_STREAM_CONTROLLER_HEADER_H

#include <cstdint>

#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <utility>
#include <atomic>

#include "hardware/hardware_abstract/device_event.h"
#include "hardware/hardware_abstract/device_context.h"
#include "common/visible.h"

namespace fxrt {
namespace device {
class SpinLock {
 public:
  void lock() {
    while (locked.test_and_set(std::memory_order_acquire)) {
    }
  }

  void unlock() {
    locked.clear(std::memory_order_release);
  }

 private:
  std::atomic_flag locked = ATOMIC_FLAG_INIT;
};
class TaskIdOnStreamManager;
using TaskIdOnStreamManagerPtr = std::shared_ptr<TaskIdOnStreamManager>;

class EventPool;
using EventPoolPtr = std::shared_ptr<EventPool>;

class FXRT_EXPORT MultiStreamController {
 public:
  explicit MultiStreamController(DeviceResManager* deviceResBase);

  MultiStreamController(const MultiStreamController&) = delete;
  MultiStreamController& operator=(const MultiStreamController&) = delete;
  MultiStreamController(const MultiStreamController&&) = delete;

  ~MultiStreamController() = default;

  void Refresh();

  bool UpdateTaskIdOnStream(int64_t taskIdOnStream, uint32_t userStreamId, uint32_t memoryStreamId);

  int64_t QueryTaskIdOnStream(uint32_t userStreamId, uint32_t memoryStreamId);

  int64_t LaunchTaskIdOnStream(uint32_t streamId);
  int64_t GetTaskIdOnStream(uint32_t streamId);

  std::mutex& GetStreamMutex(size_t streamId);

  // memoryStreamAddresses pair : memoryStreamId, address.
  bool RecordEvent(
      int64_t taskIdOnStream,
      uint32_t userStreamId,
      const std::vector<std::pair<uint32_t, void*>>& memoryStreamAddresses,
      const DeviceEventPtr& inputEvent = nullptr);
  bool WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId, uint32_t memoryStreamId);
  bool WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId);
  bool DispatchRecordWaitEvent(uint32_t userStreamId, uint32_t memoryStreamId);

  bool SyncStream(size_t streamId);
  bool SyncAllStreams();
  bool SyncNotDefaultStreams();

  bool WaitMultiStream(size_t wait_stream_id);

 protected:
  TaskIdOnStreamManagerPtr taskIdOnStreamManager_;
  std::unordered_map<uint32_t, std::mutex> streamMutexes_;
  EventPoolPtr eventPool_;

  DeviceResManager* deviceResBase_;
  SpinLock lock_;
};
using MultiStreamControllerPtr = std::shared_ptr<MultiStreamController>;
} // namespace device
} // namespace fxrt
#endif
