#ifndef __RUNTIME_BUILDER_KERNEL_CAPTURE_BUILDER_H__
#define __RUNTIME_BUILDER_KERNEL_CAPTURE_BUILDER_H__

#include "runtime/builder/builder.h"
#include "runtime/executor/kernel_capture/utils/graph_capture_manager.h"

namespace fxrt {
namespace runtime {
class DA_API KernelCaptureBuilder : public Builder {
 public:
  KernelCaptureBuilder() = delete;
  explicit KernelCaptureBuilder(const ir::GraphPtr& graph);
  ~KernelCaptureBuilder() override = default;

  std::unique_ptr<Executor> BuildExecutor() override;

  // Get the op runners created during analysis
  std::shared_ptr<std::vector<OpRunner>> GetOpRunners() const {
    return this->opRunners_;
  }

 private:
  // Store analysis results
  std::vector<std::pair<size_t, size_t>> capture_kernel_range_positions_;
  std::vector<std::pair<ExecutorType, size_t>> executors_;
};
} // namespace runtime
} // namespace fxrt

#endif // __RUNTIME_BUILDER_KERNEL_CAPTURE_BUILDER_H__
