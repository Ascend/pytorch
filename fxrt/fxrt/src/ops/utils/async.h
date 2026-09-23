#ifndef __OPS_UTILS_ASYNC_H__
#define __OPS_UTILS_ASYNC_H__

#include <cstdlib>
#include <string>
#include <string_view>
#include <functional>
#include "common/visible.h"

namespace fxrt {
namespace ops {

using BindStreamFunc = std::function<void()>;
using ProcFunc = std::function<int()>;
using LaunchOpFunc = std::function<void(const std::string& /* op_name */, const ProcFunc& /* func */, bool /* sync */)>;
using WaitLaunchFinishFunc = std::function<void()>;

class DA_API OpAsync {
 public:
  static void SetLaunchOpFunc(const LaunchOpFunc& launchOpFunc);
  static const LaunchOpFunc& GetLaunchOpFunc();
  static void SetWaitLaunchFinishFunc(const WaitLaunchFinishFunc& waitLaunchFinishFunc);
  static WaitLaunchFinishFunc const& GetWaitLaunchFinishFunc();
};

static inline bool IsEnablePipeline() {
  static const char enablePipelineEnv[] = "TASK_QUEUE_ENABLE";
  const char* enablePipelineCStr = std::getenv(enablePipelineEnv);
  const bool disablePipeline = (enablePipelineCStr != nullptr) && (std::string_view(enablePipelineCStr) == "0");
  return !disablePipeline;
}
} // namespace ops

static inline void WaitLaunchTaskFinish() {
  auto& waitLaunchFinish = fxrt::ops::OpAsync::GetWaitLaunchFinishFunc();
  if (waitLaunchFinish != nullptr) {
    waitLaunchFinish();
  }
}
} // namespace fxrt

#endif // __OPS_UTILS_ASYNC_H__
