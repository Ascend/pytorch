#include "runtime/builder/kernel_capture/kernel_capture_builder.h"
#include "runtime/executor/kernel_capture/kernel_capture_executor.h"
#include "runtime/executor/kernel_capture/utils/graph_capture_manager.h"
#include "ir/graph.h"

namespace fxrt {
namespace runtime {

KernelCaptureBuilder::KernelCaptureBuilder(const ir::GraphPtr& graph) : Builder(graph) {}

std::unique_ptr<Executor> KernelCaptureBuilder::BuildExecutor() {
  RT_VLOG(VL_RUNTIME) << "Begin build kernel capture executor.";

  // Setup OpRunners for the base graph
  SetupOpRunners();
  auto kernelCaptureExecutor = std::make_unique<KernelCaptureExecutor>(opRunners_, deviceContexts_, GetGraphOutput());

  // Initialize the executor
  kernelCaptureExecutor->Initialize(graph_);

  RT_VLOG(VL_RUNTIME) << "End build kernel capture executor.";
  return kernelCaptureExecutor;
}

} // namespace runtime
} // namespace fxrt
