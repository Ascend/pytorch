#ifndef FXRT_SRC_HARDWARE_DEVICE_EVENT_H
#define FXRT_SRC_HARDWARE_DEVICE_EVENT_H

#include <cstdint>
#include <memory>
#include <vector>
#include "common/visible.h"

namespace fxrt {
class FXRT_EXPORT DeviceEvent {
 public:
  virtual ~DeviceEvent() = default;
  virtual bool IsReady() const = 0;
  virtual void WaitEvent() = 0;
  virtual bool WaitEvent(uint32_t streamId) = 0;
  virtual void WaitEventWithoutReset() = 0;
  virtual void WaitEventWithoutReset(uint32_t streamId) {}
  virtual void ResetEvent() {}
  virtual void ResetEvent(uint32_t streamId) {}
  virtual void RecordEvent() = 0;
  virtual void RecordEvent(uint32_t streamId) = 0;
  virtual bool NeedWait() = 0;
  virtual void SyncEvent() = 0;
  virtual bool QueryEvent() = 0;
  virtual void ElapsedTime(float* costTime, const DeviceEvent* other) = 0;
  virtual bool DestroyEvent() = 0;
  virtual void set_wait_stream(void* stream) = 0;
  virtual void set_record_stream(void* stream) = 0;
};
using DeviceEventPtr = std::shared_ptr<DeviceEvent>;
using DeviceEventPtrList = std::vector<DeviceEventPtr>;
} // namespace fxrt
#endif // FXRT_SRC_HARDWARE_DEVICE_EVENT_H
