#ifndef __RUNTIME_EXECUTOR_KERNEL_CAPTURE_EXECUTOR_H__
#define __RUNTIME_EXECUTOR_KERNEL_CAPTURE_EXECUTOR_H__

#include <memory>
#include <vector>
#include <string>

#include "runtime/executor/pipeline/pipeline_executor.h"
#include "runtime/executor/kernel_capture/utils/graph_capture_manager.h"
#include "hardware/hardware_abstract/device_context.h"
#include "ir/graph.h"

namespace fxrt {
namespace runtime {
using CaptureId_t = uint64_t;
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;
using BeginAllocFunc = std::function<void(MempoolId_t)>;
using EndAllocFunc = std::function<void(MempoolId_t)>;
class KernelCaptureBuilder;
class DA_API KernelCaptureExecutor : public PipelineExecutor {
 public:
  KernelCaptureExecutor() = delete;
  KernelCaptureExecutor(
      const std::shared_ptr<std::vector<OpRunner>>& opRunners,
      const std::map<hardware::DeviceType, device::DeviceContext*>& deviceContexts,
      const ir::ValuePtr& output);
  ~KernelCaptureExecutor() override = default;

  void Initialize(const ir::GraphPtr& graph);
  void Run(bool isDynamic) override;
  const ir::ValuePtr& GetOutput() const override;

 private:
  void SetOutput(const ir::ValuePtr& output);

  // Graph used for initialization
  ir::GraphPtr graph_{nullptr};

  // Flag to control capture mode
  bool enableGraphCapture_{true};

  // Flag to control fullgraph vs piecewise mode
  bool useFullGraphMode_{true};

  // void *captureStream_{nullptr};
  void* updateStream_{nullptr};
  // Current shape key for caching
  std::string currentShapeKey_;

  device::DeviceContext* deviceContext_;

  std::unordered_map<std::string, std::unique_ptr<KernelCaptureBuilder>> builderRecorder_;
  // Graph capture manager instance
  GraphCaptureManager graphCaptureManager_;

  ir::ValuePtr currentOutput_{nullptr};
};

class DA_API KernelCaptureExecutorManager {
 public:
  static KernelCaptureExecutorManager& GetInstance();

  void SetInCapture(bool in_capture) {
    in_capture_ = in_capture;
  }
  bool InCapture() const {
    return in_capture_;
  }

  void SetInReplay(bool in_replay) {
    in_replay_ = in_replay;
  }
  bool InReplay() const {
    return in_replay_;
  }
  void SetPoolId(MempoolId_t pool_id) {
    pool_id_ = pool_id;
  }
  MempoolId_t PoolId() const {
    return pool_id_;
  }
  void SetCaptureBeginFunc(BeginAllocFunc func) {
    begin_alloc_func_ = func;
  }
  BeginAllocFunc GetBeginAllocFunc() const {
    return begin_alloc_func_;
  }
  void SetCaptureEndFunc(EndAllocFunc func) {
    end_alloc_func_ = func;
  }
  EndAllocFunc GetEndAllocFunc() const {
    return end_alloc_func_;
  }
  void SetOpCaptureSkip(const std::vector<std::string>& op_capture_skip) {
    op_capture_skip_ = op_capture_skip;
  }
  const std::vector<std::string>& OpCaptureSkip() const {
    return op_capture_skip_;
  }
  void SetShapeKey(const std::string& shape_key) {
    shape_key_ = shape_key;
  }
  const std::string& ShapeKey() const {
    return shape_key_;
  }

 private:
  KernelCaptureExecutorManager() = default;
  ~KernelCaptureExecutorManager() = default;
  DISABLE_COPY_AND_ASSIGN(KernelCaptureExecutorManager);
  bool in_capture_ = false;
  bool in_replay_ = false;
  MempoolId_t pool_id_ = {-1, -1};
  BeginAllocFunc begin_alloc_func_;
  EndAllocFunc end_alloc_func_;
  std::vector<std::string> op_capture_skip_{};
  std::string shape_key_{};
};

} // namespace runtime
} // namespace fxrt

#endif // __RUNTIME_EXECUTOR_KERNEL_CAPTURE_EXECUTOR_H__
