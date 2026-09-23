#include "runtime/builder/pipeline/pipeline_builder.h"
#include "runtime/executor/pipeline/pipeline_executor.h"

namespace fxrt {
namespace runtime {
PipelineBuilder::PipelineBuilder(const ir::GraphPtr& graph) : Builder(graph) {}

std::unique_ptr<Executor> PipelineBuilder::BuildExecutor() {
  RT_VLOG(VL_RUNTIME) << "Begin build pipeline executor.";
  SetupOpRunners();

  auto pipelineExecutor = std::make_unique<PipelineExecutor>(opRunners_, deviceContexts_, GetGraphOutput());
  pipelineExecutor->Initialize();
  RT_VLOG(VL_RUNTIME) << "End build pipeline executor.";
  return pipelineExecutor;
}
} // namespace runtime
} // namespace fxrt
