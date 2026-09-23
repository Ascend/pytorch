#ifndef __RUNTIME_BUILDER_KERNEL_LAUNCH_GROUP_BUILDER_H__
#define __RUNTIME_BUILDER_KERNEL_LAUNCH_GROUP_BUILDER_H__

#include <vector>
#include <memory>
#include <unordered_set>
#include <utility>
#include "runtime/builder/builder.h"

namespace fxrt {
namespace runtime {
class DA_API KernelLaunchGroupBuilder : public Builder {
 public:
  explicit KernelLaunchGroupBuilder(const ir::GraphPtr& graph);
  ~KernelLaunchGroupBuilder() override = default;

  std::unique_ptr<Executor> BuildExecutor() override;

 private:
  void CheckGroupLaunchRequirements() const;
  void PartitionKernelLaunchGroups();
  void RecordGraphInputs();
  void RecordGraphOutputs();

  uint64_t parallelDispatchNum_;
  uint64_t parallelSliceNum_;
  std::shared_ptr<std::vector<std::pair<OpRunner*, size_t>>> opRunnerGroups_;
  std::shared_ptr<std::vector<OpRunner*>> serialLaunchOps_;
  std::shared_ptr<std::vector<ir::TensorPtr>> graphInputTensors_;
  std::shared_ptr<std::vector<std::pair<ir::TensorPtr, std::vector<int64_t>>>> graphInputTensorsWithDynamicShape_;
  std::shared_ptr<std::unordered_set<ir::Tensor*>> graphOutputs_;
};

} // namespace runtime
} // namespace fxrt

#endif // __RUNTIME_BUILDER_KERNEL_LAUNCH_GROUP_BUILDER_H__
