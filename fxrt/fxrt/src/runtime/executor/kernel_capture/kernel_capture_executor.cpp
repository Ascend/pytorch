#include "runtime/executor/kernel_capture/kernel_capture_executor.h"
#include "runtime/builder/kernel_capture/kernel_capture_builder.h"
#include "runtime/executor/kernel_capture/utils/graph_capture_manager.h"
#include "ir/graph.h"
#include "runtime/builder/builder.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "hardware/hardware_abstract/device_context_manager.h"
#include "hardware/hardware_abstract/device_context.h"
#include "ops/utils/async.h"

namespace fxrt {
namespace runtime {
KernelCaptureExecutorManager& KernelCaptureExecutorManager::GetInstance() {
  static KernelCaptureExecutorManager instance;
  return instance;
}

KernelCaptureExecutor::KernelCaptureExecutor(
    const std::shared_ptr<std::vector<OpRunner>>& opRunners,
    const std::map<hardware::DeviceType, device::DeviceContext*>& deviceContexts,
    const ir::ValuePtr& output)
    : PipelineExecutor(opRunners, deviceContexts, output) {
  device::DeviceContextKey deviceContextKey = device::DeviceToDeviceContextKey(
      {hardware::DeviceType::NPU, Uint32ToInt8(fxrt::collective::CollectiveManager::Instance().local_rank_id())});
  deviceContext_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(deviceContextKey);
}

void KernelCaptureExecutor::Initialize(const ir::GraphPtr& graph) {
  PipelineExecutor::Initialize();
  graph_ = graph;
  graphCaptureManager_.Initialize(*opRunners_, deviceContext_);

  if (useFullGraphMode_) {
    size_t streamID;
    if (!deviceContext_->deviceResManager_->CreateStream(&streamID)) {
      RT_GLOG(EXCEPTION) << "Create stream failed.";
    }
    updateStream_ = deviceContext_->deviceResManager_->GetStream(streamID);
  }
  size_t captureStreamID;
  if (!deviceContext_->deviceResManager_->CreateStream(&captureStreamID)) {
    RT_GLOG(EXCEPTION) << "Create stream failed.";
  }
  // captureStream_ = deviceContext_->deviceResManager_->GetStream(captureStreamID);
  // CHECK_IF_NULL(captureStream_);
}

void KernelCaptureExecutor::Run(bool isDynamic) {
  KernelCaptureExecutorManager::GetInstance().SetInReplay(false);
  for (const auto& item : deviceContexts_) {
    auto* deviceContext = item.second;
    CHECK_IF_NULL(deviceContext);
    CHECK_IF_NULL(deviceContext->deviceResManager_);
    deviceContext->deviceResManager_->BindDeviceToCurrentThread(false);
  }

  // Generate current shape key for this execution
  currentShapeKey_ = KernelCaptureExecutorManager::GetInstance().ShapeKey();
  graphCaptureManager_.SetShapeKey(currentShapeKey_);
  void* currentStream = deviceContext_->deviceResManager_->GetCurrentStream();

  // Need to wait launch async copy task to staticize input tensor and eager mode task finish(push all task to current
  // stream first).
  WaitLaunchTaskFinish();

  // Check if we have a cached graph for this shape
  if (graphCaptureManager_.HasCapturedGraph()) {
    KernelCaptureExecutorManager::GetInstance().SetInReplay(true);
    RT_VLOG(VL_RUNTIME) << "Replaying captured graph for shape key: " << currentShapeKey_;
    auto& cachedBuilder = builderRecorder_[currentShapeKey_];
    CHECK_IF_NULL(cachedBuilder);
    graphCaptureManager_.LaunchAllKernelsWithReplayFullGraph(
        *(cachedBuilder->GetOpRunners()), currentStream, updateStream_);
    SetOutput(cachedBuilder->GetGraphOutput());
  } else {
    // Check if we're in capture period
    if (KernelCaptureExecutorManager::GetInstance().InCapture()) {
      // Case 5.1: Need to capture new graph
      RT_VLOG(VL_RUNTIME) << "Capturing new graph for shape key: " << currentShapeKey_;

      // Create a copy of the graph for capture
      auto copiedGraph = graph_->DeepCopy();
      // RT_VLOG(VL_RUNTIME) << "Copied graph for capture ";
      // Create a new kernel capture builder for the copied graph to generate new op runners
      auto captureBuilder = std::make_unique<KernelCaptureBuilder>(copiedGraph);
      captureBuilder->SetupOpRunners();
      SetOutput(captureBuilder->GetGraphOutput());

      // Get the op runners from the capture builder
      auto captureOpRunners = captureBuilder->GetOpRunners();
      builderRecorder_.emplace(currentShapeKey_, std::move(captureBuilder));

      // Initialize capture graphs
      graphCaptureManager_.CreateCaptureGraph(deviceContext_, useFullGraphMode_);

      if (useFullGraphMode_) {
        // Use fullgraph capture mode
        // Allocate memory for the capture graph
        auto begin_alloc = KernelCaptureExecutorManager::GetInstance().GetBeginAllocFunc();
        begin_alloc(KernelCaptureExecutorManager::GetInstance().PoolId());
        // Must use current stream for fxrt ops to capture custom call (op torch call) ops
        graphCaptureManager_.LaunchAllKernelsWithCaptureFullGraph(*captureOpRunners, currentStream);
        auto end_alloc = KernelCaptureExecutorManager::GetInstance().GetEndAllocFunc();
        end_alloc(KernelCaptureExecutorManager::GetInstance().PoolId());

        graphCaptureManager_.LaunchAllKernelsWithReplayFullGraph(*captureOpRunners, currentStream, updateStream_);
      } else {
        // Use piecewise capture mode
        graphCaptureManager_.LaunchAllKernelsWithCapture(*captureOpRunners);
      }
    } else {
      RT_VLOG(VL_RUNTIME) << "Pipeline run graph for shape key: " << currentShapeKey_;
      PipelineExecutor::Run(true);
      SetOutput(Executor::GetOutput());
    }
  }
}

const ir::ValuePtr& KernelCaptureExecutor::GetOutput() const {
  return currentOutput_;
}

void KernelCaptureExecutor::SetOutput(const ir::ValuePtr& output) {
  currentOutput_ = output;
}
} // namespace runtime
} // namespace fxrt
