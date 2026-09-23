#ifndef __RUNTIME_EXECUTOR_PIPELINE_EXECUTOR_H__
#define __RUNTIME_EXECUTOR_PIPELINE_EXECUTOR_H__

#include <atomic>
#include <mutex>
#include <condition_variable>
#include "runtime/executor/executor.h"

namespace fxrt {
namespace runtime {
class DA_API PipelineExecutor : public Executor {
 public:
  PipelineExecutor() = delete;
  PipelineExecutor(
      const std::shared_ptr<std::vector<OpRunner>>& opRunners,
      const std::map<hardware::DeviceType, device::DeviceContext*>& deviceContexts,
      const ir::ValuePtr& output);
  ~PipelineExecutor() override = default;

  void Initialize();
  void Run(bool isDynamic) override;

 private:
  void TryWaitLastRunFinish();

  bool initialized_;
  // This variable is used to determine whether, in pipeline mode, the main thread needs to wait for the completion of
  // the graph's launch task when re-entering the graph. In other words, before the launch task of the same graph is
  // completed, the main thread must wait.
  std::atomic<bool> waitLastRunFinish_{false};
  std::mutex mtx_;
  std::condition_variable cv_;
};

} // namespace runtime
} // namespace fxrt

#endif // __RUNTIME_EXECUTOR_PIPELINE_EXECUTOR_H__
