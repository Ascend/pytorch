#include "hardware/ascend/res_manager/ascend_event.h"
#include <cstdint>
#include <string>
#include "hardware/ascend/res_manager/ascend_stream_manager.h"

#include "common/common.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace fxrt::device::ascend {
AscendEvent::AscendEvent() {
  auto ret = CALL_ASCEND_API(aclrtCreateEvent, &event_);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "aclrtCreateEvent failed, ret:" << ret;
    event_ = nullptr;
  }
}

AscendEvent::AscendEvent(uint32_t flag, bool useExtensionalApi) {
  aclError ret;
  if (useExtensionalApi) {
    ret = CALL_ASCEND_API(aclrtCreateEventExWithFlag, &event_, flag);
    if (ret != ACL_SUCCESS) {
      RT_GLOG(ERROR) << "aclrtCreateEventExWithFlag failed, ret:" << ret;
      event_ = nullptr;
    }
  } else {
    ret = CALL_ASCEND_API(aclrtCreateEventWithFlag, &event_, flag);
    if (ret != ACL_SUCCESS) {
      RT_GLOG(ERROR) << "aclrtCreateEventWithFlag failed, ret:" << ret;
      event_ = nullptr;
    }
  }
  hasFlag_ = true;
  RT_VLOG(VL_HARDWARE) << "Create ascend event success, flag : " << flag << ".";
}

AscendTimeEvent::AscendTimeEvent() {
  auto ret = CALL_ASCEND_API(aclrtCreateEventWithFlag, &event_, ACL_EVENT_TIME_LINE);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "aclrtCreateEvent failed, ret:" << ret;
    event_ = nullptr;
  }
}

AscendEvent::~AscendEvent() {
  if (!eventDestroyed_) {
    auto ret = CALL_ASCEND_API(aclrtDestroyEvent, event_);
    if (ret != ACL_SUCCESS) {
      RT_GLOG(ERROR) << "aclrtDestroyEvent failed, ret:" << ret;
    }
  }

  event_ = nullptr;
  waitStream_ = nullptr;
  recordStream_ = nullptr;
}

bool AscendEvent::IsReady() const {
  return event_ != nullptr;
}

void AscendEvent::RecordEvent() {
  CHECK_IF_NULL(event_);
  CHECK_IF_NULL(recordStream_);
  auto ret = CALL_ASCEND_API(aclrtRecordEvent, event_, recordStream_);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "aclrtRecordEvent failed, ret:" << ret;
  }
  needWait_ = true;
}

void AscendEvent::RecordEvent(uint32_t streamId) {
  RT_VLOG(VL_HARDWARE) << "Ascend record event on stream id : " << streamId << ".";
  CHECK_IF_NULL(event_);
  recordStream_ = AscendStreamMng::GetInstance().GetStream(streamId);
  CHECK_IF_NULL(recordStream_);
  auto ret = CALL_ASCEND_API(aclrtRecordEvent, event_, recordStream_);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "aclrtRecordEvent failed, ret:" << ret;
  }
  needWait_ = true;
}

void AscendEvent::WaitEvent() {
  CHECK_IF_NULL(event_);
  CHECK_IF_NULL(waitStream_);
  auto ret = CALL_ASCEND_API(aclrtStreamWaitEvent, waitStream_, event_);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "aclrtStreamWaitEvent failed, ret:" << ret;
  }
  if (!hasFlag_) {
    // The event created by aclrtCreateEventExWithFlag is not support to call
    // aclrtResetEvent/aclrtQueryEvent/aclrtQueryEventWaitStatus.
    RT_VLOG(VL_HARDWARE) << "Reset Event";
    ret = CALL_ASCEND_API(aclrtResetEvent, event_, waitStream_);
    if (ret != ACL_SUCCESS) {
      RT_GLOG(ERROR) << "aclrtResetEvent failed, ret:" << ret;
    }
  }
  needWait_ = false;
}

bool AscendEvent::WaitEvent(uint32_t streamId) {
  RT_VLOG(VL_HARDWARE) << "Ascend wait event on stream id : " << streamId << ".";
  waitStream_ = AscendStreamMng::GetInstance().GetStream(streamId);
  auto ret = CALL_ASCEND_API(aclrtStreamWaitEvent, waitStream_, event_);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "aclrtStreamWaitEvent failed, ret:" << ret;
  }
  if (!hasFlag_) {
    // Reset event after wait so that event can be reused.
    ret = CALL_ASCEND_API(aclrtResetEvent, event_, waitStream_);
    if (ret != ACL_SUCCESS) {
      RT_GLOG(ERROR) << "aclrtResetEvent failed, ret:" << ret;
    }
  }
  needWait_ = false;
  return true;
}

void AscendEvent::WaitEventWithoutReset() {
  CHECK_IF_NULL(event_);
  CHECK_IF_NULL(waitStream_);
  // Query result will be reset after aclrtResetEvent is called.
  auto ret = CALL_ASCEND_API(aclrtStreamWaitEvent, waitStream_, event_);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "aclrtStreamWaitEvent failed, ret:" << ret;
  }
  needWait_ = false;
}

void AscendEvent::WaitEventWithoutReset(uint32_t streamId) {
  waitStream_ = AscendStreamMng::GetInstance().GetStream(streamId);
  WaitEventWithoutReset();
}

void AscendEvent::ResetEvent() {
  CHECK_IF_NULL(event_);
  CHECK_IF_NULL(waitStream_);

  RT_VLOG(VL_HARDWARE) << "Reset Event";
  auto ret = CALL_ASCEND_API(aclrtResetEvent, event_, waitStream_);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "aclrtResetEvent failed, ret:" << ret;
  }
}

void AscendEvent::ResetEvent(uint32_t streamId) {
  waitStream_ = AscendStreamMng::GetInstance().GetStream(streamId);
  ResetEvent();
}

void AscendEvent::SyncEvent() {
  CHECK_IF_NULL(event_);
  auto ret = CALL_ASCEND_API(aclrtSynchronizeEvent, event_);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "aclrtSynchronizeEvent failed, ret:" << ret;
  }
}

bool AscendEvent::QueryEvent() {
  CHECK_IF_NULL(event_);
  aclrtEventRecordedStatus status;
  auto ret = CALL_ASCEND_API(aclrtQueryEventStatus, event_, &status);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "aclQueryEventStatus failed, ret:" << ret;
  }
  return status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
}

void AscendEvent::ElapsedTime(float* costTime, const DeviceEvent* other) {
  CHECK_IF_NULL(event_);
  auto ascendOther = static_cast<const AscendEvent*>(other);
  CHECK_IF_NULL(ascendOther);
  CHECK_IF_NULL(ascendOther->event_);
  auto ret = CALL_ASCEND_API(aclrtEventElapsedTime, costTime, event_, ascendOther->event_);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "aclrtEventElapsedTime failed, ret:" << ret;
  }
}

bool AscendEvent::NeedWait() {
  return needWait_;
}

bool AscendEvent::DestroyEvent() {
  CHECK_IF_NULL(event_);
  auto ret = CALL_ASCEND_API(aclrtDestroyEvent, event_);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "aclrtDestroyEvent failed, ret:" << ret;
  }
  eventDestroyed_ = true;
  return true;
}
} // namespace fxrt::device::ascend
