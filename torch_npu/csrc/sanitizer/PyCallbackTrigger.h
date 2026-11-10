#pragma once

#include <cstdint>
#include <string>

namespace c10_npu {
namespace impl {

enum class SanitizerMode { STREAM = 0, KERNEL };

// Registered callbacks must remain valid for the lifetime of the trace object.
struct NPUTraceCallbacks {
  void (*traceNpuAclStartExecution)(const std::string& acl_name);
  void (*traceNpuAclFinishExecution)(const std::string& acl_name);
  void (*traceNpuEventCreation)(uintptr_t event);
  void (*traceNpuEventDeletion)(uintptr_t event);
  void (*traceNpuEventRecord)(uintptr_t event, uintptr_t stream);
  void (*traceNpuEventWait)(uintptr_t event, uintptr_t stream);
  void (*traceNpuEventGetHandle)(uintptr_t event, uintptr_t handle);
  void (*traceNpuEventOpenHandle)(uintptr_t handle, uintptr_t event);
  void (*traceNpuMemoryAllocation)(uintptr_t ptr);
  void (*traceNpuMemoryDeallocation)(uintptr_t ptr);
  void (*traceNpuStreamCreation)(uintptr_t stream);
  void (*traceNpuDeviceSynchronization)();
  void (*traceNpuStreamSynchronization)(uintptr_t stream);
  void (*traceNpuEventSynchronization)(uintptr_t event);
  void (*traceNpuExternalEventWrite)(uintptr_t event, uintptr_t stream);
  void (*traceNpuExternalEventWait)(uintptr_t event, uintptr_t stream);
  void (*traceNpuRecordStream)(uintptr_t data_ptr, uintptr_t stream);
  void (*traceNpuEraseStream)(uintptr_t data_ptr, uintptr_t stream);
};

struct PyCallbackTrigger {
    const SanitizerMode sanitizer_mode;

    PyCallbackTrigger(const int mode, const NPUTraceCallbacks& callbacks) : sanitizer_mode(static_cast<SanitizerMode>(mode)), callbacks_(callbacks) {}
  void traceNpuAclStartExecution(std::string acl_name) const
    {
        if (sanitizer_mode == SanitizerMode::KERNEL) {
      callbacks_.traceNpuAclStartExecution(acl_name);
        }
    }
    void traceNpuAclFinishExecution(std::string acl_name) const
    {
        if (sanitizer_mode == SanitizerMode::KERNEL) {
      callbacks_.traceNpuAclFinishExecution(acl_name);
        }
    }
    void traceNpuEventCreation(uintptr_t event) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuEventCreation(event);
        }
    }
    void traceNpuEventDeletion(uintptr_t event) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuEventDeletion(event);
        }
    }
    void traceNpuEventRecord(uintptr_t event, uintptr_t stream) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuEventRecord(event, stream);
        }
    }
    void traceNpuEventWait(uintptr_t event, uintptr_t stream) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuEventWait(event, stream);
        }
    }
    void traceNpuEventGetHandle(uintptr_t event, uintptr_t handle) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuEventGetHandle(event, handle);
        }
    }
    void traceNpuEventOpenHandle(uintptr_t handle, uintptr_t event) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuEventOpenHandle(handle, event);
        }
    }
    void traceNpuMemoryAllocation(uintptr_t ptr) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuMemoryAllocation(ptr);
        }
    }
    void traceNpuMemoryDeallocation(uintptr_t ptr) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuMemoryDeallocation(ptr);
        }
    }
    void traceNpuStreamCreation(uintptr_t stream) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuStreamCreation(stream);
        }
    }
    void traceNpuDeviceSynchronization() const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuDeviceSynchronization();
        }
    }
    void traceNpuStreamSynchronization(uintptr_t stream) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuStreamSynchronization(stream);
        }
    }
    void traceNpuEventSynchronization(uintptr_t event) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuEventSynchronization(event);
    }
  }
  void traceNpuExternalEventWrite(uintptr_t event, uintptr_t stream) const {
    if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuExternalEventWrite(event, stream);
    }
  }
  void traceNpuExternalEventWait(uintptr_t event, uintptr_t stream) const {
    if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuExternalEventWait(event, stream);
    }
  }
  void traceNpuRecordStream(uintptr_t data_ptr, uintptr_t stream) const {
    if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuRecordStream(data_ptr, stream);
    }
  }
  void traceNpuEraseStream(uintptr_t data_ptr, uintptr_t stream) const {
    if (sanitizer_mode == SanitizerMode::STREAM) {
      callbacks_.traceNpuEraseStream(data_ptr, stream);
        }
    }
    static PyCallbackTrigger* instance(const int mode);

 private:
  const NPUTraceCallbacks& callbacks_;
};

PyCallbackTrigger* getPyCallbackTrigger(const int mode);

} // namespace impl
} // namespace c10_npu
