#ifndef FXRT_SRC_HARDWARE_ASCEND_ASCEND_EVENT_H
#define FXRT_SRC_HARDWARE_ASCEND_ASCEND_EVENT_H

#include "hardware/hardware_abstract/device_event.h"
#include "acl/acl_rt.h"
#include "common/visible.h"

namespace fxrt::device::ascend {
constexpr uint32_t ACL_EVENT_DEFAULT = 0x0000000Eu;

class FXRT_EXPORT AscendEvent : public DeviceEvent {
 public:
  AscendEvent();
  explicit AscendEvent(uint32_t flag, bool useExtensionalApi = true);
  ~AscendEvent() override;

  bool IsReady() const override;
  void WaitEvent() override;
  bool WaitEvent(uint32_t streamId) override;
  void WaitEventWithoutReset() override;
  void WaitEventWithoutReset(uint32_t streamId) override;

  void ResetEvent() override;
  void ResetEvent(uint32_t streamId) override;

  void RecordEvent() override;
  void RecordEvent(uint32_t streamId) override;
  bool NeedWait() override;
  void SyncEvent() override;
  bool QueryEvent() override;
  void ElapsedTime(float* costTime, const DeviceEvent* other) override;
  bool DestroyEvent() override;
  void set_wait_stream(aclrtStream waitStream) override {
    waitStream_ = waitStream;
  }
  void set_record_stream(aclrtStream recordStream) override {
    recordStream_ = recordStream;
  }

 protected:
  aclrtEvent event_{nullptr};
  aclrtStream waitStream_{nullptr};
  aclrtStream recordStream_{nullptr};
  bool needWait_{false};
  bool eventDestroyed_{false};
  bool hasFlag_{false};
};

class FXRT_EXPORT AscendTimeEvent : public AscendEvent {
 public:
  AscendTimeEvent();
  ~AscendTimeEvent() override = default;
};
} // namespace fxrt::device::ascend
#endif // FXRT_SRC_HARDWARE_ASCEND_ASCEND_EVENT_H
