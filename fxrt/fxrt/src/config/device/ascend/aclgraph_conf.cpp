#include "config/device/ascend/aclgraph_conf.h"
namespace fxrt {
namespace config {
namespace ascend {

AclGraphConf& AclGraphConf::Instance() {
  static AclGraphConf instance;
  return instance;
}

void AclGraphConf::BeginCapture() {
  KernelCaptureExecutorManager::GetInstance().SetInCapture(true);
}

void AclGraphConf::EndCapture() {
  KernelCaptureExecutorManager::GetInstance().SetInCapture(false);
}

bool AclGraphConf::IsCapturing() const {
  return KernelCaptureExecutorManager::GetInstance().InCapture();
}

MempoolId_t AclGraphConf::GetPoolId() const {
  return KernelCaptureExecutorManager::GetInstance().PoolId();
}

void AclGraphConf::SetPoolId(MempoolId_t poolId) {
  KernelCaptureExecutorManager::GetInstance().SetPoolId(poolId);
}

void AclGraphConf::SetOpCaptureSkip(const std::vector<std::string>& op_capture_skip) {
  KernelCaptureExecutorManager::GetInstance().SetOpCaptureSkip(op_capture_skip);
}
} // namespace ascend
} // namespace config
} // namespace fxrt
