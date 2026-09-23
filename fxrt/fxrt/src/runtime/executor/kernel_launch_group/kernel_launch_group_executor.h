#ifndef __RUNTIME_EXECUTOR_KERNEL_LAUNCH_GROUP_EXECUTOR_H__
#define __RUNTIME_EXECUTOR_KERNEL_LAUNCH_GROUP_EXECUTOR_H__

#include <memory>
#include <map>
#include <unordered_map>
#include <vector>
#include <utility>
#include <unordered_set>
#include "runtime/executor/pipeline/pipeline_executor.h"
#include "runtime/executor/pipeline/async_task_queue.h"
#include "hardware/hardware_abstract/device_context.h"
#include "runtime/executor/kernel_launch_group/memory_cache/memory_cache.h"

namespace fxrt {
namespace runtime {
class DA_API KernelLaunchGroupExecutor : public PipelineExecutor {
 public:
  KernelLaunchGroupExecutor(
      const std::shared_ptr<std::vector<OpRunner>>& opRunners,
      const std::map<hardware::DeviceType, device::DeviceContext*>& deviceContexts,
      const std::shared_ptr<std::vector<std::pair<OpRunner*, size_t>>>& opRunnerGroups,
      const std::shared_ptr<std::vector<OpRunner*>>& serialLaunchOps,
      const std::shared_ptr<std::vector<ir::TensorPtr>>& graphInputTensors,
      const std::shared_ptr<std::vector<std::pair<ir::TensorPtr, std::vector<int64_t>>>>& graphInputsWithShape,
      const std::shared_ptr<std::unordered_set<ir::Tensor*>>& graphOutputs,
      uint64_t parallelDispatchNum,
      uint64_t parallelSliceNum,
      const ir::ValuePtr& output);

  ~KernelLaunchGroupExecutor() override;

  void Initialize();

  void Run(bool isDynamic) override;

 private:
  void UpdateInputTensors();
  bool CheckInputShapeChange();
  void ResetTensorCacheMemory();
  void RunWithRecordCacheMemory();
  void RecordMemory(OpRunner* opRunner, const ir::Value* value);

  void AllocateGraphCacheMemory() const;
  void FreeGraphCacheMemory() const;
  void AllocateGraphOutputMemory();
  void SetOutputAndWsCacheMemory(OpRunner* opRunner);
  void UpdateInputTensorData(
      const std::shared_ptr<std::unordered_map<const Tensor*, KernelMemoryTraceBlockPtr>>& tensorToKernelMemBlocks,
      const std::shared_ptr<std::map<const DeviceContext*, std::vector<MemoryTraceBlockPtr>>>&
          mergeBlocksWithDeviceContext,
      const ir::Tensor* tensor) const;
  void SetInputCacheMemory(const OpRunner* opRunner) const;

  void ParallelDispatchKernels();
  void DispatchParallelLaunchKernels(size_t index);
  void DispatchSerialLaunchKernels();

  static std::vector<std::pair<size_t, void*>> streams_;
  static std::vector<DeviceEventPtr> events_;
  static std::vector<DeviceEventPtr> eventsToDefaultStream_;
  static std::vector<AsyncTaskQueuePtr> queues_;

  std::unordered_map<OpRunner*, std::vector<DeviceEventPtr>> serialLaunchKernelsToEvents_{};

  device::DeviceContext* deviceContext_;

  std::shared_ptr<std::vector<std::pair<OpRunner*, size_t>>> opRunnerGroups_;
  std::shared_ptr<std::vector<OpRunner*>> serialLaunchOps_;

  std::shared_ptr<std::vector<ir::TensorPtr>> graphInputTensors_;
  std::shared_ptr<std::vector<std::pair<ir::TensorPtr, std::vector<int64_t>>>> graphInputTensorsWithDynamicShape_;
  std::shared_ptr<std::unordered_set<ir::Tensor*>> graphOutputs_;

  uint64_t parallelDispatchNum_{0};
  uint64_t parallelSliceNum_{0};

  MemoryCache memoryCache_;

  bool initialized_;
};
} // namespace runtime
} // namespace fxrt

#endif // __RUNTIME_EXECUTOR_KERNEL_LAUNCH_GROUP_EXECUTOR_H__
