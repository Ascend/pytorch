#ifndef FXRT_SRC_HARDWARE_ASCEND_ASCEND_HAL_MANAGER_H_
#define FXRT_SRC_HARDWARE_ASCEND_ASCEND_HAL_MANAGER_H_

#include <map>
#include <mutex>
#include <set>
#include "acl/acl_rt.h"
#include "common/visible.h"

namespace fxrt {
namespace device {
namespace ascend {
class FXRT_EXPORT AscendHalManager {
 public:
  static AscendHalManager& GetInstance();

  ~AscendHalManager() {}
  // init

  // device
  uint32_t GetDeviceCount();
  void InitDevice(uint32_t deviceId);
  void ResetDevice(uint32_t deviceId);
  void SetDeviceSatMode(const aclrtFloatOverflowMode& overflowMode);
  void SetOpWaitTimeout(uint32_t opWaitTimeout);
  void SetOpExecuteTimeOut(uint32_t opExecuteTimeout);
  void InitializeAcl();
  bool EnableLccl();

  // context
  aclrtContext CreateContext(uint32_t deviceId);
  // reset the default context of deviceId
  void ResetContext(uint32_t deviceId);
  void SetContext(uint32_t deviceId);
  void SetContextForce(uint32_t deviceId);
  void DestroyContext(aclrtContext context);
  void DestroyAllContext();

 private:
  static AscendHalManager instance_;
  std::set<uint32_t> initializedDeviceSet_{};
  // default <deviceId, aclrtcontext> pair
  std::map<uint32_t, aclrtContext> defaultDeviceContextMap_;

  // rt_contexts by aclrtCreateContext, to destroy
  std::set<aclrtContext> rtContexts_;

  bool aclInitialized_ = false;
  std::mutex aclInitMutex_;
};

} // namespace ascend
} // namespace device
} // namespace fxrt

#endif // FXRT_SRC_HARDWARE_ASCEND_ASCEND_HAL_MANAGER_H_
