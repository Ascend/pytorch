#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/core/npu/NPUStreamUtils.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif
#include <chrono>
#include <thread>

namespace c10_npu {

NPUEvent::NPUEvent() {
  flags_ = c10_npu::acl::IsExistCreateEventExWithFlag() ? ACL_EVENT_SYNC : ACL_EVENT_DEFAULT;
}

NPUEvent::NPUEvent(unsigned int flags) {
  flags_ = flags;
  if (flags_ == ACL_EVENT_EXTERNAL && c10_npu::acl::IsExistValueWaitAndWrite()) {
    createEvent(getCurrentNPUStream().device_index());
  }
}

NPUEvent::NPUEvent(unsigned int flags, bool use_native_external) {
  use_native_external_ = use_native_external;
  if (use_native_external_) {
    // The native external path records/waits the event via
    // aclrtRecordEventWithFlag/aclrtStreamWaitEventWithFlag, which requires the
    // event to be created by aclrtCreateEventExWithFlag. Fail fast on old CANN
    // instead of silently falling back (explicit guidance over implicit fallback).
    TORCH_CHECK(
        c10_npu::acl::IsExistCreateEventExWithFlag() && IsGteCANNVersion("9.3.0"),
        "external=True requires CANN 9.3.0+ (aclrtCreateEventExWithFlag and the "
        "external record/wait runtime support), please upgrade the CANN package.",
        PTA_ERROR(ErrCode::NOT_SUPPORT));
    // ACL_EVENT_EXTERNAL is not allowed at creation time for the Ex API:
    // the externality comes from the runtime record/wait flag. Keep SYNC and
    // TIME_LINE so the event supports in-graph EXTERNAL records and timing.
    flags_ = ACL_EVENT_SYNC | (flags & ACL_EVENT_TIME_LINE);
  } else {
    flags_ = flags;
    if (flags_ == ACL_EVENT_EXTERNAL && c10_npu::acl::IsExistValueWaitAndWrite()) {
      createEvent(getCurrentNPUStream().device_index());
    }
  }
}

bool NPUEvent::isLegacyValueWriteExternal() const {
  return !use_native_external_ && flags_ == ACL_EVENT_EXTERNAL && c10_npu::acl::IsExistValueWaitAndWrite();
}

NPUEvent::NPUEvent(c10::DeviceIndex device_index, const aclrtIpcEventHandle* handle) : device_index_(device_index) {
  NPUGuard guard(device_index_);
  LazySetDevice(device_index_);
  NPU_CHECK_ERROR(acl::AclIpcOpenEventHandle(*handle, &event_));
  c10_npu::NPUEventManager::GetInstance().AddIpcEvent(event_);
  flags_ = ACL_EVENT_IPC;
  is_created_ = true;

#ifndef BUILD_LIBTORCH
  const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
  if (C10_UNLIKELY(trigger)) {
    trigger->traceNpuEventOpenHandle(reinterpret_cast<uintptr_t>(handle), reinterpret_cast<uintptr_t>(event_));
  }
#endif
}

NPUEvent::~NPUEvent() {
  try {
    if (flags_ == ACL_EVENT_EXTERNAL && c10_npu::acl::IsExistValueWaitAndWrite()) {
      return;
    }
    if (is_created_ && (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag())) {
      NPU_CHECK_ERROR(c10_npu::queue::LaunchLazyDestroyEventTask(event_, device_index_));
      if (!c10_npu::acl::IsExistCreateEventExWithFlag() || c10_npu::option::OptionsManager::GetPerStreamQueue() ||
          flags_ == ACL_EVENT_IPC) {
        c10_npu::NPUEventManager::GetInstance().QueryAndDestroyEvent();
      }
    }
  } catch (...) {
    // stay consistent with pytorch, no throw
  }
}

NPUEvent::NPUEvent(NPUEvent&& other) {
  moveHelper(std::move(other));
}

NPUEvent& NPUEvent::operator=(NPUEvent&& other) {
  moveHelper(std::move(other));
  return *this;
}

bool NPUEvent::query() const {
  if (!is_created_) {
    return true;
  }
  // Native external event: query the real recorded status from the runtime
  // for both record flavors. An in-graph EXTERNAL record is submitted
  // directly to the runtime and bypasses the host task queue, so only a
  // default (out-of-graph) record needs the queue check.
  if (use_native_external_) {
    if (!was_recorded_) {
      return was_recorded_;
    }
    // The record task may still sit in the task queue: until it is dequeued the
    // event has not been submitted to the runtime yet.
    if (native_record_flag_ != ACL_EVENT_RECORD_EXTERNAL && c10_npu::option::OptionsManager::GetTaskQueueEnable() &&
        !c10_npu::NPUEventManager::GetInstance().IsEventRecorded(event_)) {
      return false;
    }
    acl::aclrtEventRecordedStatus currStatus = acl::ACL_EVENT_RECORDED_STATUS_NOT_READY;
    NPU_CHECK_ERROR(acl::AclQueryEventRecordedStatus(event_, &currStatus));
    return currStatus == acl::ACL_EVENT_RECORDED_STATUS_COMPLETE;
  }
  // Legacy value-write external event: event_ is a 32-byte memory block and
  // must not be passed to the runtime query APIs. Report the host-side state
  // instead: once recorded, always True (wait does not consume the state).
  if (isLegacyValueWriteExternal()) {
    return was_recorded_;
  }
  if (c10_npu::option::OptionsManager::GetTaskQueueEnable() &&
      !c10_npu::NPUEventManager::GetInstance().IsEventRecorded(event_)) {
    return false;
  }
  acl::aclrtEventRecordedStatus currStatus = acl::ACL_EVENT_RECORDED_STATUS_NOT_READY;
  NPU_CHECK_ERROR(acl::AclQueryEventRecordedStatus(event_, &currStatus));

  if (currStatus == acl::ACL_EVENT_RECORDED_STATUS_COMPLETE) {
    return true;
  }
  return false;
}

void NPUEvent::record() {
  record(getCurrentNPUStream());
}

void NPUEvent::recordOnce(const NPUStream& stream) {
  if (!was_recorded_) {
    record(stream);
  }
}

void NPUEvent::record(const NPUStream& stream) {
  c10_npu::detail::checkNotExternalStream(stream, "NPUEvent::record");
  if (!is_created_) {
    createEvent(stream.device_index());
  }

  TORCH_CHECK(
      device_index_ == stream.device_index(),
      "Event device ",
      device_index_,
      " does not match recording stream's device ",
      stream.device_index(),
      ".",
      PTA_ERROR(ErrCode::PARAM));
  NPUGuard guard(device_index_);
  if (use_native_external_) {
    // Native external path (align with cudaEventRecordExternal): the runtime
    // flag is chosen by the current stream capture status - during capture the
    // record becomes an external event node in the graph, outside capture it
    // degrades to a normal record. The first bound flag is tracked for the
    // query() queue check; unsupported flag switches are rejected by the
    // runtime itself.
    auto captureStatus = c10_npu::currentStreamCaptureStatusMayInitCtx();
    uint32_t recordFlag =
        (captureStatus != c10_npu::CaptureStatus::None) ? ACL_EVENT_RECORD_EXTERNAL : ACL_EVENT_RECORD_DEFAULT;
    // Out-of-graph (DEFAULT) records are equivalent to normal records (CANN:
    // ACL_EVENT_RECORD_DEFAULT equals aclrtRecordEvent), so reuse the task
    // queue path of a normal Event - same FIFO as the eager ops, stream
    // ordering preserved without host sync.
    if (recordFlag == ACL_EVENT_RECORD_EXTERNAL) {
      // Check the return code explicitly instead of NPU_CHECK_ERROR: its
      // warn-once branch silently swallows 107000 (e.g. the same-graph
      // record+wait mutex) and leaves the event state inconsistent.
      aclError ret = c10_npu::acl::AclrtRecordEventWithFlag(event_, stream.stream(), recordFlag);
      TORCH_CHECK(
          ret == ACL_ERROR_NONE,
          "External event: in-graph record failed with ACL error ",
          ret,
          ". The runtime rejected the call - known reasons include: the record mode was already bound "
          "by an out-of-graph (default) record and cannot be switched to in-graph (external), the same "
          "event being both EXTERNAL-recorded and EXTERNAL-waited in the same graph (not allowed), or "
          "missing device-side support (driver/firmware). See the CANN slog for the exact reason.",
          PTA_ERROR(ErrCode::ACL));
    } else if (native_record_flag_ == ACL_EVENT_RECORD_EXTERNAL) {
      // Already bound to the in-graph (EXTERNAL) record mode: the runtime rejects
      // the DEFAULT mode switch, so submit directly to surface the rejection
      // synchronously instead of poisoning the task queue.
      aclError ret = c10_npu::acl::AclrtRecordEventWithFlag(event_, stream.stream(), recordFlag);
      TORCH_CHECK(
          ret == ACL_ERROR_NONE,
          "External event: out-of-graph record failed with ACL error ",
          ret,
          ". The event is already bound to the in-graph (external) record mode and the runtime rejects "
          "the mode switch - use a new Event(external=True) for out-of-graph records. See the CANN slog "
          "for the exact reason.",
          PTA_ERROR(ErrCode::ACL));
    } else {
      c10_npu::queue::LaunchRecordEventTask(event_, stream, flags_);
    }
    // Bind the flag only after the record call succeeds, so the query() queue
    // check reflects the actually submitted mode.
    native_record_flag_ = recordFlag;
    ASCEND_LOGI("External Event: The record is executed, flag=%u", recordFlag);
  } else {
    c10_npu::queue::LaunchRecordEventTask(event_, stream, flags_);
    if (flags_ == ACL_EVENT_EXTERNAL && c10_npu::acl::IsExistValueWaitAndWrite()) {
      ASCEND_LOGI("External Event: The record is ready to be executed via value write");
      is_waited_ = false;
    }
  }
  was_recorded_ = true;
}

void NPUEvent::block(const NPUStream& stream) {
  c10_npu::detail::checkNotExternalStream(stream, "NPUEvent::block");
  if (is_created_) {
    if (use_native_external_) {
      // Native external path (align with cudaEventWaitExternal): symmetric to
      // record - the runtime flag follows the capture status and
      // ACL_EVENT_WAIT_EXTERNAL only supports timeout 0. Unsupported flag
      // switches (e.g. waiting in capture an event that was recorded via the
      // default path) are rejected by the runtime itself.
      NPUGuard wait_guard(stream.device_index());
      auto captureStatus = c10_npu::currentStreamCaptureStatusMayInitCtx();
      uint32_t waitFlag =
          (captureStatus != c10_npu::CaptureStatus::None) ? ACL_EVENT_WAIT_EXTERNAL : ACL_EVENT_WAIT_DEFAULT;
      // Out-of-graph (DEFAULT) waits are equivalent to normal waits (CANN:
      // ACL_EVENT_WAIT_DEFAULT equals aclrtStreamWaitEvent), so reuse the task
      // queue path of a normal Event for stream ordering.
      if (waitFlag == ACL_EVENT_WAIT_EXTERNAL) {
        // Explicit return check - see the in-graph record branch above for why
        // NPU_CHECK_ERROR is not used here.
        aclError ret = c10_npu::acl::AclrtStreamWaitEventWithFlag(stream.stream(), event_, 0, waitFlag);
        TORCH_CHECK(
            ret == ACL_ERROR_NONE,
            "External event: in-graph wait failed with ACL error ",
            ret,
            ". The runtime rejected the call - known reasons include: the wait mode was already bound "
            "by an out-of-graph (default) wait and cannot be switched to in-graph (external), a "
            "default-recorded event cannot be waited inside the graph, the same event being both "
            "EXTERNAL-recorded and EXTERNAL-waited in the same graph (not allowed), or missing "
            "device-side support (driver/firmware). See the CANN slog for the exact reason.",
            PTA_ERROR(ErrCode::ACL));
      } else if (native_record_flag_ == ACL_EVENT_RECORD_EXTERNAL) {
        // Already bound to the in-graph (EXTERNAL) mode: submit the out-of-graph
        // (DEFAULT) wait directly so a mode-switch rejection surfaces synchronously
        // instead of poisoning the task queue.
        aclError ret = c10_npu::acl::AclrtStreamWaitEventWithFlag(stream.stream(), event_, 0, waitFlag);
        TORCH_CHECK(
            ret == ACL_ERROR_NONE,
            "External event: out-of-graph wait failed with ACL error ",
            ret,
            ". The event is already bound to the in-graph (external) mode and the runtime rejects "
            "the mode switch. See the CANN slog for the exact reason.",
            PTA_ERROR(ErrCode::ACL));
      } else {
        // Same ordering guards as the legacy path: with per-stream queues the
        // wait task (this stream's queue) could overtake a record task still
        // sitting in another stream's queue, and an ongoing capture could
        // submit before the enqueued wait task. An in-graph EXTERNAL record is
        // never enqueued on the task queue, so there is no record task to
        // wait for in that case.
        while (c10_npu::option::OptionsManager::GetPerStreamQueue() &&
               native_record_flag_ != ACL_EVENT_RECORD_EXTERNAL &&
               !c10_npu::NPUEventManager::GetInstance().IsEventRecorded(event_)) {
          std::this_thread::sleep_for(std::chrono::microseconds(10)); // 10 us
        }
        c10_npu::queue::LaunchWaitEventTask(event_, stream, flags_);
        if (c10_npu::NPUCachingAllocator::hasCapturesUnderway(stream.device_index())) {
          c10_npu::emptyAllNPUStream();
        }
      }
      native_wait_flag_ = waitFlag;
      ASCEND_LOGI("External Event: The block is executed, flag=%u", waitFlag);
      return;
    }
    // Legacy value-write external event: the single-wait check MUST run before
    // the wait task is enqueued. With the task queue disabled the value wait
    // executes synchronously, so a second wait would hang forever on
    // ValueWait(value == 1).
    if (isLegacyValueWriteExternal()) {
      TORCH_CHECK(
          !is_waited_,
          "External event: A single record operation cannot be associated with multiple wait operations. "
          "Please call reset() before waiting again.",
          PTA_ERROR(ErrCode::INTERNAL));
    }
    // If using multiple task queues or using IPC events across devices in a
    // single process, it is necessary to ensure that the enqueued record is
    // dequeued before wait.
    while ((c10_npu::option::OptionsManager::GetPerStreamQueue() || flags_ == ACL_EVENT_IPC) &&
           !c10_npu::NPUEventManager::GetInstance().IsEventRecorded(event_)) {
      std::this_thread::sleep_for(std::chrono::microseconds(10)); // 10 us
    }
    NPUGuard guard(stream.device_index());
    c10_npu::queue::LaunchWaitEventTask(event_, stream, flags_);
    if (c10_npu::NPUCachingAllocator::hasCapturesUnderway(stream.device_index())) {
      c10_npu::emptyAllNPUStream();
    }
    if (flags_ == ACL_EVENT_EXTERNAL && c10_npu::acl::IsExistValueWaitAndWrite()) {
      ASCEND_LOGI("External Event: The block is ready to be executed via value wait");
      is_waited_ = true;
    }
  }
}

float NPUEvent::elapsed_time(const NPUEvent& other) const {
  TORCH_CHECK(
      is_created_ && other.isCreated(),
      "Both events must be recorded before calculating elapsed time.",
      PTA_ERROR(ErrCode::INTERNAL));
  // Legacy value-write external events are 32-byte memory blocks and must
  // not be passed to the runtime timing APIs.
  TORCH_CHECK(
      !isLegacyValueWriteExternal() && !other.isLegacyValueWriteExternal(),
      "elapsed_time is not supported for external events of the legacy value write mechanism. "
      "Please use a normal Event or Event(external=True) with enable_timing=True for timing.",
      PTA_ERROR(ErrCode::NOT_SUPPORT));
  TORCH_CHECK(
      (flags_ & ACL_EVENT_TIME_LINE) && (other.flags_ & ACL_EVENT_TIME_LINE),
      "Both events must be created with enable_timing=True.",
      PTA_ERROR(ErrCode::INTERNAL));
  float time_ms = 0;
  NPUStatus ret = c10_npu::emptyAllNPUStream();
  if (ret != NPU_STATUS_SUCCESS) {
    ASCEND_LOGE("Failed to empty NPU task queue, ret: %s", ret.c_str());
  }
  NPU_CHECK_ERROR(aclrtSynchronizeEvent(event_));
  ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed, event=%p", event_);
  NPU_CHECK_ERROR(aclrtSynchronizeEvent(other.event_));
  ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed, other.event=%p", other.event_);
#ifndef BUILD_LIBTORCH
  const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
  if (C10_UNLIKELY(trigger)) {
    trigger->traceNpuEventSynchronization(reinterpret_cast<uintptr_t>(event_));
    trigger->traceNpuEventSynchronization(reinterpret_cast<uintptr_t>(other.event_));
  }
#endif
  // raise error if either event is recorded but not yet completed
  NPU_CHECK_ERROR(aclrtEventElapsedTime(&time_ms, event_, other.event_));
  return time_ms;
}

uint64_t NPUEvent::recorded_time() const {
  TORCH_CHECK(is_created_, "Event must be recorded before getting recorded timestamp.", PTA_ERROR(ErrCode::INTERNAL));
  // See elapsed_time(): legacy value-write external events must not reach the
  // runtime timing APIs.
  TORCH_CHECK(
      !isLegacyValueWriteExternal(),
      "recorded_time is not supported for external events of the legacy value write mechanism. "
      "Please use a normal Event or Event(external=True) with enable_timing=True for timing.",
      PTA_ERROR(ErrCode::NOT_SUPPORT));
  TORCH_CHECK(
      (flags_ & ACL_EVENT_TIME_LINE), "Event must be created with enable_timing=True.", PTA_ERROR(ErrCode::INTERNAL));
  NPUStatus ret = c10_npu::emptyAllNPUStream();
  if (ret != NPU_STATUS_SUCCESS) {
    ASCEND_LOGE("Failed to empty NPU task queue, ret: %s", ret.c_str());
  }
  NPU_CHECK_ERROR(aclrtSynchronizeEvent(event_));
  ASCEND_LOGI("Event: aclrtSynchronizeEvent executed successfully, event=%p", event_);
#ifndef BUILD_LIBTORCH
  const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
  if (C10_UNLIKELY(trigger)) {
    trigger->traceNpuEventSynchronization(reinterpret_cast<uintptr_t>(event_));
  }
#endif
  // raise error if either event is recorded but not yet completed
  uint64_t time_stamp = 0;
  NPU_CHECK_ERROR(c10_npu::acl::AclrtEventGetTimestamp(event_, &time_stamp));
  return time_stamp;
}

void NPUEvent::synchronize() const {
  if (is_created_) {
    if (use_native_external_) {
      if (!was_recorded_) {
        return;
      }
      // An out-of-graph (default) record may still sit in the task queue: wait
      // until it is dequeued before synchronizing. In-graph (EXTERNAL) records
      // bypass the queue.
      if (c10_npu::option::OptionsManager::GetTaskQueueEnable() && native_record_flag_ == ACL_EVENT_RECORD_DEFAULT) {
        // Spin with a short sleep: a dropped record task must degrade to a
        // low-CPU wait, not a full-core busy loop.
        NPUEventManager& mgr = c10_npu::NPUEventManager::GetInstance();
        while (!mgr.IsEventRecorded(event_)) {
          std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
      }
      NPU_CHECK_ERROR(aclrtSynchronizeEvent(event_));
      ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed, event=%p", event_);
#ifndef BUILD_LIBTORCH
      const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
      if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuEventSynchronization(reinterpret_cast<uintptr_t>(event_));
      }
#endif
      return;
    }
    // Legacy value-write external events are 32-byte memory blocks: neither
    // the event-manager polling nor aclrtSynchronizeEvent can handle them
    // (would hang or crash). Reject explicitly, same family fix as
    // query()/elapsed_time()/recorded_time().
    TORCH_CHECK(
        !isLegacyValueWriteExternal(),
        "synchronize is not supported for external events of the legacy value write mechanism. "
        "Please wait on a stream or use Event(external=True) instead.",
        PTA_ERROR(ErrCode::NOT_SUPPORT));
    bool task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();
    if (task_queue_enable) {
      NPUEventManager& mgr = c10_npu::NPUEventManager::GetInstance();
      // Same low-CPU spin as the external path above.
      while (!mgr.IsEventRecorded(event_)) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }
    NPU_CHECK_ERROR(aclrtSynchronizeEvent(event_));
    ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed, event=%p", event_);
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
      trigger->traceNpuEventSynchronization(reinterpret_cast<uintptr_t>(event_));
    }
#endif
  }
}

void NPUEvent::reset(const NPUStream& stream) const {
  TORCH_CHECK(
      !use_native_external_, "reset() is not supported for Event(external=True).", PTA_ERROR(ErrCode::NOT_SUPPORT));
  if (!c10_npu::acl::IsExistValueWaitAndWrite() && is_created_) {
    TORCH_CHECK(
        flags_ == ACL_EVENT_EXTERNAL,
        "API reset() only support ACL_EVENT_EXTERNAL flag event.",
        PTA_ERROR(ErrCode::INTERNAL));
    NPUGuard guard(stream.device_index());
    NPU_CHECK_ERROR(aclrtResetEvent(event_, stream.stream()));
  }
}

// Note: AclIpcGetEventHandle must be called on the same device as the event
void NPUEvent::ipc_handle(aclrtIpcEventHandle* handle) {
  if (!is_created_) {
    // this NPUEvent object was initially constructed from flags but event_
    // is not created yet.
    createEvent(getCurrentNPUStream().device_index());
  }

  // If using Event across processes, make sure that the enqueued record is
  // dequeued before the wait of other processes.
  while (c10_npu::option::OptionsManager::GetTaskQueueEnable() != 0 &&
         !c10_npu::NPUEventManager::GetInstance().IsEventRecorded(event_)) {
    std::this_thread::sleep_for(std::chrono::microseconds(10)); // 10 us
  }

  NPUGuard guard(device_index_);
  NPU_CHECK_ERROR(acl::AclIpcGetEventHandle(event_, handle));
#ifndef BUILD_LIBTORCH
  const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
  if (C10_UNLIKELY(trigger)) {
    trigger->traceNpuEventGetHandle(reinterpret_cast<uintptr_t>(event_), reinterpret_cast<uintptr_t>(handle));
  }
#endif
}

void NPUEvent::createEvent(c10::DeviceIndex device_index) {
  device_index_ = device_index;
  NPUGuard guard(device_index_);
  LazySetDevice(device_index_);
  if (use_native_external_) {
    NPU_CHECK_ERROR(c10_npu::acl::AclrtCreateEventExWithFlag(&event_, flags_));
    ASCEND_LOGI("Event: aclrtCreateEventExWithFlag is successfully executed, event=%p", event_);
  } else {
    NPU_CHECK_ERROR(c10_npu::acl::AclrtCreateEventWithFlag(&event_, flags_));
    ASCEND_LOGI("Event: aclrtCreateEventWithFlag is successfully executed, event=%p", event_);
  }
#ifndef BUILD_LIBTORCH
  const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
  if (C10_UNLIKELY(trigger)) {
    trigger->traceNpuEventCreation(reinterpret_cast<uintptr_t>(event_));
  }
#endif
  if (flags_ == ACL_EVENT_IPC) {
    c10_npu::NPUEventManager::GetInstance().AddIpcEvent(event_);
  }
  is_created_ = true;
}

void NPUEvent::moveHelper(NPUEvent&& other) {
  std::swap(flags_, other.flags_);
  std::swap(is_created_, other.is_created_);
  std::swap(was_recorded_, other.was_recorded_);
  std::swap(device_index_, other.device_index_);
  std::swap(event_, other.event_);
  std::swap(is_waited_, other.is_waited_);
  std::swap(use_native_external_, other.use_native_external_);
  std::swap(native_record_flag_, other.native_record_flag_);
  std::swap(native_wait_flag_, other.native_wait_flag_);
}

NPUEventPtr EventPool::get(const c10::DeviceIndex device) {
  // If the device is invalid, return a default event and no pooling
  if (device < 0 || device >= (c10::DeviceIndex)pools_.size()) {
    auto deleter = [](NPUEvent* event) { delete event; };
    return NPUEventPtr(std::make_unique<NPUEvent>(flags_).release(), deleter);
  }

  auto& pool = pools_[device];

  // Create a destructor that returns the event to the appropriate device pool
  auto destructor = [&pool](NPUEvent* event) noexcept {
    if (event != nullptr) {
      std::lock_guard<std::mutex> lock(pool.mutex_);
      pool.event_pool_.emplace_back(event);
    }
  };

  {
    std::lock_guard<std::mutex> lock(pool.mutex_);
    if (!pool.event_pool_.empty()) {
      auto event = std::move(pool.event_pool_.back());
      pool.event_pool_.pop_back();
      return NPUEventPtr(event.release(), destructor);
    }
  }

  return NPUEventPtr(std::make_unique<NPUEvent>(flags_).release(), destructor);
}

void EventPool::empty_cache() {
  for (auto& pool : pools_) {
    std::lock_guard<std::mutex> lock(pool.mutex_);
    pool.event_pool_.clear();
  }
}
} // namespace c10_npu
