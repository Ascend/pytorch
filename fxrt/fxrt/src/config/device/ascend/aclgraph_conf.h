#ifndef __CONFIG_DEVICE_ASCEND_ACLGRAPH_CONF_H__
#define __CONFIG_DEVICE_ASCEND_ACLGRAPH_CONF_H__

#include <string>

#include "common/common.h"
#include "common/visible.h"
#include "config/device/ascend/common.h"
#include "runtime/executor/kernel_capture/kernel_capture_executor.h"

namespace fxrt {
namespace config {
namespace ascend {
using MempoolId_t = fxrt::runtime::MempoolId_t;
using KernelCaptureExecutorManager = fxrt::runtime::KernelCaptureExecutorManager;
class FXRT_EXPORT AclGraphConf {
 public:
  static AclGraphConf& Instance();

  void BeginCapture();

  void EndCapture();

  bool IsCapturing() const;

  MempoolId_t GetPoolId() const;

  void SetPoolId(MempoolId_t poolId);

  void SetOpCaptureSkip(const std::vector<std::string>& op_capture_skip = {});

 private:
  AclGraphConf() = default;
  DISABLE_COPY_AND_ASSIGN(AclGraphConf);
};

} // namespace ascend
} // namespace config
} // namespace fxrt
#endif // __CONFIG_DEVICE_ASCEND_ACLGRAPH_CONF_H__
