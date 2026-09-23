#ifndef FXRT_SRC_HARDWARE_ASCEND_ASCEND_RES_MANAGER_H_
#define FXRT_SRC_HARDWARE_ASCEND_ASCEND_RES_MANAGER_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <utility>
#include "acl/acl_rt.h"
#include "hardware/ascend/res_manager/ascend_stream_manager.h"
#include "hardware/hardware_abstract/device_event.h"
#include "hardware/hardware_abstract/capture_graph.h"
#include "hardware/hardware_abstract/device_context.h"
#include "common/visible.h"

namespace fxrt {
namespace device {
namespace ascend {
std::string GetCurrentDir();

using DeviceMemInfo = std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>;
class FXRT_EXPORT AscendResManager : public DeviceResManager {
 public:
  AscendResManager() = default;
  ~AscendResManager() override;

  void Initialize() override;

  void Destroy() override;

  std::shared_ptr<MemoryManager> mem_manager() const override {
    return memManager_;
  }

  std::vector<void*> AllocateContinuousMemory(
      const std::vector<size_t>& sizeList,
      uint32_t streamId = kDefaultStreamIndex) const override;
  bool IsEnableVmm() const override;

  bool BindDeviceToCurrentThread(bool forceBind) const override;
  void* GetStream() const override {
    return AscendStreamMng::GetInstance().default_stream();
  }
  void* GetCopyDataStream() const;

  void* AllocateStaticMemory(size_t size, uint32_t streamId = kDefaultStreamIndex) const;
  void* AllocateMemory(size_t size, uint32_t streamId = kDefaultStreamIndex) const override;
  void FreeMemory(void* ptr) const override;
  void SetAllocator(const AllocateFunc& allocator) override;
  void SetDeleter(const DeleteFunc& deleter) override;
  void FreePartMemorys(
      const std::vector<void*>& freeAddrs,
      const std::vector<void*>& keepAddrs,
      const std::vector<size_t>& keepAddrSizes) const override;
  void DefragMemory() override;

  size_t GetMaxUsedMemorySize() const override;

  // Relevant function to manage memory statistics
  size_t GetTotalMemStatistics() const override;
  size_t GetTotalUsedMemStatistics() const override;
  size_t GetTotalIdleMemStatistics() const override;
  size_t GetTotalEagerFreeMemStatistics() const override;
  size_t GetUsedMemPeakStatistics() const override;
  size_t GetReservedMemPeakStatistics() const override;
  std::unordered_map<std::string, std::size_t> GetBlockCountsStatistics() const override;
  std::unordered_map<std::string, std::size_t> GetBlockUnitSizeStatistics() const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> GetCommonMemBlocksInfoStatistics()
      const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  GetPersistentMemBlocksInfoStatistics() const override;
  void ResetMaxMemoryReserved() override;
  void ResetMaxMemoryAllocated() override;

  size_t EmptyCache() override;

  bool CreateStream(size_t* streamId) const override;
  bool CreateStreamWithPriority(size_t* streamId, int32_t priority) const override;
  bool DestroyStream(size_t streamId) const override;
  size_t QueryStreamSize() const override;
  std::vector<uint32_t> GetStreamIds() const override;
  void* GetStream(size_t streamId) const override;
  void SetCurrentStreamId(size_t streamId) override;
  size_t GetCurrentStreamId() const override;
  void SetCurrentStream(void* currentStream) override;
  void* GetCurrentStream() const override;
  void BindCurrentStream() override;
  void SetBindStreamFunc(const BindStreamFunc& bindStreamFunc) override;

  bool QueryStream(size_t streamId) const override;
  bool SyncStream(size_t streamId = 0) const override;
  bool SyncAllStreams(bool syncDevice = true) const override;
  bool SyncNotDefaultStreams() const override;
  size_t DefaultStream() const override;

  DeviceEventPtr CreateRuntimeEvent(bool enableBlocking, bool enableRecordWait) override;
  CaptureGraphPtr CreateCaptureGraph() override;
  DeviceEventPtr CreateEventWithFlag(bool enableTiming, bool external, bool useExtensionalApi) override;
  bool DestroyEvent(const DeviceEventPtr& event) override;
  bool DestroyAllEvents() override;

  bool singleOpMultiStreamEnable() const override;
  void set_single_op_multi_stream_enable(bool singleOpMultiStreamEnable) override;
  void SetCPUMemManager();

  // Override interface for multi stream event control.
  bool RecordEvent(
      int64_t taskIdOnStream,
      uint32_t userStreamId,
      const std::vector<std::pair<uint32_t, DeviceMemPtr>>& memoryStreamAddresses,
      const DeviceEventPtr& inputEvent) override;

  bool WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId, uint32_t memoryStreamId) override;

  bool WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId) override;

  bool SyncAllEvents() override;

  bool LaunchCallback(std::function<void(void)> callbackFunc, size_t streamId, bool isBlock = false) const override;

  void ResetStreamAndCtx() const override;

  bool AsyncCopy(void* dst, const void* src, uint64_t size, CopyType kind, void* stream) const override;
  bool SyncCopy(void* dst, const void* src, uint64_t size, CopyType kind) const override;

  // Memcpy
  static bool MemcpyDeviceToDevice(void* dst, size_t dst_size, const void* src, size_t src_size, aclrtStream stream);
  static bool MemcpyDeviceToHost(void* dst, size_t dst_size, const void* src, size_t src_size, aclrtStream stream);

 private:
  std::mutex deviceEventsMutex_;
  DeviceEventPtrList deviceEvents_{};
  std::shared_ptr<MemoryManager> memManager_{nullptr};
  uint32_t deviceId_{0};
  bool enableMemoryTracker_{false};
  bool initialized_ = false;
  BindStreamFunc bindStreamFunc_{nullptr};
  AllocateFunc allocator_{nullptr};
  DeleteFunc deleter_{nullptr};
};
} // namespace ascend
} // namespace device
} // namespace fxrt
#endif // FXRT_SRC_HARDWARE_ASCEND_ASCEND_RES_MANAGER_H_
