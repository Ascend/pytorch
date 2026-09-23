#include "ops/utils/async.h"

namespace fxrt {
namespace ops {

namespace {
LaunchOpFunc gLaunchOpFunc = nullptr;
WaitLaunchFinishFunc gWaitLaunchFinishFunc = nullptr;
} // namespace

void OpAsync::SetLaunchOpFunc(const LaunchOpFunc& launchOpFunc) {
  gLaunchOpFunc = launchOpFunc;
}

const LaunchOpFunc& OpAsync::GetLaunchOpFunc() {
  return gLaunchOpFunc;
}

void OpAsync::SetWaitLaunchFinishFunc(const WaitLaunchFinishFunc& waitLaunchFinishFunc) {
  gWaitLaunchFinishFunc = waitLaunchFinishFunc;
}

const WaitLaunchFinishFunc& OpAsync::GetWaitLaunchFinishFunc() {
  return gWaitLaunchFinishFunc;
}

} // namespace ops
} // namespace fxrt
