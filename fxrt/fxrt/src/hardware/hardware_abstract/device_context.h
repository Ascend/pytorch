#ifndef FXRT_SRC_HARDWARE_DEVICE_CONTEXT_H_
#define FXRT_SRC_HARDWARE_DEVICE_CONTEXT_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <mutex>
#include <functional>
#include "common/common.h"
#include "common/visible.h"
#include "hardware/hardware_abstract/stream_util.h"
#include "hardware/hardware_abstract/device_event.h"
#include "hardware/hardware_abstract/capture_graph.h"
#include "hardware/device.h"
#ifdef __APPLE__
#include "async/spinlock.h"
#endif

namespace fxrt {
namespace runtime {
enum class KernelTaskType;
} // namespace runtime
namespace device {
using BindStreamFunc = std::function<void()>;
using AllocateFunc = std::function<void*(size_t)>;
using DeleteFunc = std::function<void(void*)>;
constexpr size_t kSizeZero = 0;

struct FXRT_EXPORT DeviceContextKey {
  // device type name, such as 'Ascend' 'CPU'.
  std::string deviceName_;
  uint32_t deviceId_{0};

  // Use the result of ToString() as key to look up DeviceContext
  // in cache map which maintains created DeviceContext objects.
  std::string ToString() const {
    return deviceName_ + "_" + std::to_string(deviceId_);
  }
};

FXRT_EXPORT DeviceContextKey DeviceToDeviceContextKey(hardware::Device device);

enum class CopyType : uint8_t { H2D = 0, D2H = 1, D2D = 2, H2H = 3 };

class DeviceResManager;
class KernelExecutor;

// DeviceContext is unified interface of interaction with device.
class FXRT_EXPORT DeviceContext {
 public:
  explicit DeviceContext(const DeviceContextKey& deviceContextKey)
      : deviceContextKey_(deviceContextKey), initialized_(false) {}
  virtual ~DeviceContext() = default;

  // Initialize the device context.
  virtual void Initialize() = 0;

  // Destroy device context and release device resource.
  virtual void Destroy() = 0;

  // Get deviceContextKey_ to obtain device name and device id.
  const DeviceContextKey& GetDeviceContextKey() const {
    return deviceContextKey_;
  }

  // Get kernel executor.
  std::shared_ptr<KernelExecutor> GetKernelExecutor() const {
    return kernelExecutor_;
  }

  void SetKernelExecutor(const std::shared_ptr<KernelExecutor>& kernelExecutor) {
    kernelExecutor_ = kernelExecutor;
  }

  // Return whether this device context is initialized.
  bool initialized() const;

  DeviceContextKey deviceContextKey_;
  std::unique_ptr<DeviceResManager> deviceResManager_;

 protected:
#ifdef __APPLE__
  // There are some problems with using mutex on Mac, use spinlocks instead.
  inline static SpinLock initLock_;
#else
  inline static std::mutex initMutex_;
#endif
  bool initialized_;

 private:
  std::shared_ptr<KernelExecutor> kernelExecutor_;
};
using DeviceContextPtr = std::shared_ptr<DeviceContext>;
class MemoryManager;
class CollectiveCommunicationLib;
class OffloadedMemPool;
using DeviceMemPtr = void*;

class FXRT_EXPORT DeviceResManager {
 public:
  DeviceResManager();

  virtual ~DeviceResManager() = default;

  // Initialize the device resource manager.
  virtual void Initialize() {}

  virtual void SetAclDeterministic() {}

  // Destroy device resource manager and release device resource.
  virtual void Destroy() {}

  // Bind device to current thread to gain device control privileges
  // If forceBind is true, bind context to current thread every time;
  // Otherwise, only bind context to current thread for the first time.
  virtual bool BindDeviceToCurrentThread(bool forceBind) const {
    return true;
  }
  virtual void ResetStreamAndCtx() const {}

  // Relevant function to allocate and free device memory of raw ptr.
  virtual void* AllocateMemory(size_t size, uint32_t streamId = kDefaultStreamIndex) const = 0;
  virtual void FreeMemory(void* ptr) const = 0;
  virtual void FreePartMemorys(
      const std::vector<void*>& freeAddrs,
      const std::vector<void*>& keepAddrs,
      const std::vector<size_t>& keepAddrSizes) const = 0;
  virtual void DefragMemory() {}
  virtual bool IsEnableVmm() const {
    return false;
  }

  // Interface for multi stream event control.
  virtual bool RecordEvent(
      int64_t taskIdOnStream,
      uint32_t userStreamId,
      const std::vector<std::pair<uint32_t, DeviceMemPtr>>& memoryStreamAddresses,
      const DeviceEventPtr& inputEvent) {
    return false;
  }

  virtual bool WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId, uint32_t memoryStreamId) {
    return false;
  }

  virtual bool WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId) {
    return false;
  }

  virtual bool SyncAllEvents() {
    return false;
  }

  virtual size_t GetMaxUsedMemorySize() const {
    return 0;
  }

  // Relevant function to manage memory statistics
  virtual size_t GetTotalMemStatistics() const {
    return 0;
  }
  virtual size_t GetTotalUsedMemStatistics() const {
    return 0;
  }
  virtual size_t GetTotalIdleMemStatistics() const {
    return 0;
  }
  virtual size_t GetTotalEagerFreeMemStatistics() const {
    return 0;
  }
  virtual size_t GetUsedMemPeakStatistics() const {
    return 0;
  }
  virtual size_t GetReservedMemPeakStatistics() const {
    return 0;
  }
  virtual std::unordered_map<std::string, std::size_t> GetBlockCountsStatistics() const {
    return {};
  }
  virtual std::unordered_map<std::string, std::size_t> GetBlockUnitSizeStatistics() const {
    return {};
  }
  virtual std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  GetCommonMemBlocksInfoStatistics() const {
    return {};
  }
  virtual std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  GetPersistentMemBlocksInfoStatistics() const {
    return {};
  }
  virtual void ResetMaxMemoryReserved() {}
  virtual void ResetMaxMemoryAllocated() {}

  virtual size_t EmptyCache() {
    return -1L;
  }

  // Allocate host memory with raii and ref count
  virtual std::shared_ptr<void> AllocateHostMemory(size_t size) const {
    return std::shared_ptr<void>(::malloc(size), ::free);
  }
  virtual size_t GetAvailableMemSize() const {
    return 0;
  }

  // Allocate continuous device memory according to size list.
  // Communication operators may need continuous memory for input and output
  // to optimize the communication performance.
  virtual std::vector<void*> AllocateContinuousMemory(
      const std::vector<size_t>& sizeList,
      uint32_t streamId = kDefaultStreamIndex) const {
    RT_GLOG(ERROR) << "Unimplemented interface.";
    return {};
  }

  // Create a stream with assigning a stream id, the assigned stream id will be written to the parameter '*streamId'.
  virtual bool CreateStream(size_t* streamId) const {
    RT_GLOG(ERROR) << "Unimplemented interface: 'CreateStream'.";
    *streamId = kSizeZero;
    return false;
  }

  // Create a stream with priority.
  virtual bool CreateStreamWithPriority(size_t* streamId, int32_t priority) const {
    *streamId = kSizeZero;
    return false;
  }

  virtual size_t QueryStreamSize() const {
    return 0L;
  }
  virtual std::vector<uint32_t> GetStreamIds() const {
    return {};
  }

  // If multi-stream used in pynative mode, other streams must be sync before the graph
  // is executed. Otherwise, out-of-order occurs. Therefore this flag is added.
  // This solution is a temporary solution, this flag will be removed after multi-stream is
  // supported in graph mode.
  virtual bool singleOpMultiStreamEnable() const {
    return false;
  }
  virtual void set_single_op_multi_stream_enable(bool singleOpMultiStreamEnable) {}

  virtual void SetAllocator(const AllocateFunc& allocator) {}

  virtual void SetDeleter(const DeleteFunc& deleter) {}

  // Get the stream pointer by streamId.
  virtual void* GetStream(size_t streamId) const {
    return nullptr;
  }

  // Set currently using stream id.
  virtual void SetCurrentStreamId(size_t streamId) {
    return;
  }

  // Get currently using stream id.
  virtual size_t GetCurrentStreamId() const {
    return kSizeZero;
  }

  virtual void SetCurrentStream(void* currentStream) {}
  virtual void* GetCurrentStream() const {
    return nullptr;
  }

  virtual void BindCurrentStream() {}
  virtual void SetBindStreamFunc(const BindStreamFunc& bindStreamFunc) {}

  virtual void* GetStream() const {
    return nullptr;
  }

  virtual size_t GetCommunicationStreamID() const {
    return kDefaultStreamIndex;
  }

  virtual size_t GetCommunicationStreamIDByGroup(const std::string& group) const {
    return GetCommunicationStreamID();
  }

  // Destroy a stream bound to the input parameter "streamId".
  virtual bool DestroyStream(size_t streamId) const {
    return false;
  }

  // Query tasks' completion status of a stream.
  virtual bool QueryStream(size_t streamId) const {
    return true;
  }

  // Synchronize stream, device such as Ascend need stream to launch kernel asynchronously,
  // Using 'SyncStream' to block thread and wait for completing all tasks on specific stream.
  // Using 'SyncAllStream' to block thread and wait for completing all tasks on all streams.
  // Devices without stream could ignore the implementation of these function.
  // Since the current entry for creating streams is not unified, the implementation of the 'SyncStream' and
  // "SyncAllStreams" interfaces are implemented by subclasses.
  virtual bool SyncStream(size_t streamId) const {
    return true;
  }

  // 'syncDevice' is used for Ascend backend.
  virtual bool SyncAllStreams(bool syncDevice = true) const {
    return true;
  }

  virtual bool SyncNotDefaultStreams() const {
    return true;
  }

  // Return default stream id. Normally it's 0.
  virtual size_t DefaultStream() const {
    return 0;
  }

  // Create device event for runtime.
  virtual DeviceEventPtr CreateRuntimeEvent(bool enableBlocking, bool enableRecordWait) {
    return nullptr;
  }

  // Create aclgraph
  virtual CaptureGraphPtr CreateCaptureGraph() {
    return nullptr;
  }

  // Create device event with flag.
  virtual DeviceEventPtr CreateEventWithFlag(bool enableTiming, bool external, bool useExtensionalApi = true) {
    return nullptr;
  }

  // Destroy specified device event.
  virtual bool DestroyEvent(const DeviceEventPtr& event) {
    return true;
  }

  // Destroy all device events.
  virtual bool DestroyAllEvents() {
    return true;
  }

  virtual std::shared_ptr<MemoryManager> mem_manager() const {
    return nullptr;
  }

  virtual bool LaunchCallback(std::function<void(void)> callbackFunc, size_t streamId, bool isBlock = false) const {
    callbackFunc();
    return true;
  }

  virtual bool AsyncCopy(void* dst, const void* src, uint64_t size, CopyType kind, void* stream) const {
    return true;
  }
  virtual bool SyncCopy(void* dst, const void* src, uint64_t size, CopyType kind) const {
    return true;
  }

 protected:
  DeviceContext* deviceContext_{nullptr};

 private:
  template <class... Args>
  friend class DeviceInterface;
  void SetDeviceContext(DeviceContext* deviceContext) {
    deviceContext_ = deviceContext;
  }
  std::shared_ptr<device::OffloadedMemPool> offloadedMemPool_;
};

using CallbackFunc = std::function<void(void)>;

class FXRT_EXPORT KernelExecutor {
 public:
  virtual ~KernelExecutor() = default;

  virtual void Initialize() {}
  virtual void Destroy() {}

  void SetDeviceContext(DeviceContext* deviceContext) {
    deviceContext_ = deviceContext;
  }

 protected:
  DeviceContext* deviceContext_{nullptr};
};

template <class... Args>
class DeviceInterface : public DeviceContext {};

template <>
class DeviceInterface<> : public DeviceContext {
 public:
  explicit DeviceInterface(const DeviceContextKey& key) : DeviceContext(key) {}

 protected:
  void CheckUnset(const void* ptr, const std::string& error_msg) const {
    if (ptr != nullptr) {
      RT_GLOG(ERROR) << error_msg;
    }
  }
};

template <class T, class... Args>
class DeviceInterface<T, Args...> : public DeviceInterface<Args...> {
 public:
  explicit DeviceInterface(const DeviceContextKey& key) : DeviceInterface<Args...>(key) {
    if constexpr (std::is_base_of_v<DeviceResManager, T>) {
      DeviceInterface::CheckUnset(
          reinterpret_cast<void*>(DeviceContext::deviceResManager_.get()), "DeviceResManager has been registered!");
      DeviceContext::deviceResManager_ = std::make_unique<T>();
      DeviceContext::deviceResManager_->SetDeviceContext(this);
    } else if constexpr (std::is_base_of_v<KernelExecutor, T>) { // NOLINT(readability/braces)
      DeviceInterface::CheckUnset(
          reinterpret_cast<void*>(DeviceContext::GetKernelExecutor().get()), "KernelExecutor has been registered!");
      DeviceContext::SetKernelExecutor(std::make_shared<T>());
      DeviceContext::GetKernelExecutor()->SetDeviceContext(this);
    }
  }
};
} // namespace device
} // namespace fxrt
#endif // FXRT_SRC_HARDWARE_DEVICE_CONTEXT_H_
