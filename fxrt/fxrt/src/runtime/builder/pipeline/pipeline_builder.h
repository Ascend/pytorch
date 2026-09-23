#ifndef __RUNTIME_BUILDER_PIPELINE_BUILDER_H__
#define __RUNTIME_BUILDER_PIPELINE_BUILDER_H__

#include "runtime/builder/builder.h"

namespace fxrt {
namespace runtime {
class DA_API PipelineBuilder : public Builder {
 public:
  PipelineBuilder() = delete;
  explicit PipelineBuilder(const ir::GraphPtr& graph);
  ~PipelineBuilder() override = default;

  std::unique_ptr<Executor> BuildExecutor() override;
};

} // namespace runtime
} // namespace fxrt

#endif // __RUNTIME_BUILDER_PIPELINE_BUILDER_H__
