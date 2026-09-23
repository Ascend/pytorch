#ifndef FXRT_SRC_HARDWARE_ASCEND_ASCEND_DEVICE_CONTEXT_H_
#define FXRT_SRC_HARDWARE_ASCEND_ASCEND_DEVICE_CONTEXT_H_

#include <memory>
#include <string>
#include <map>
#include "common/common.h"
#include "hardware/hardware_abstract/device_context.h"
#include "hardware/hardware_abstract/memory_manager.h"
#include "hardware/ascend/res_manager/ascend_res_manager.h"

namespace fxrt {
namespace device {
namespace ascend {
class AscendResManager;

class FXRT_EXPORT AscendDeviceContext : public DeviceInterface<AscendResManager> {
 public:
  explicit AscendDeviceContext(const DeviceContextKey& deviceContextKey) : DeviceInterface(deviceContextKey) {}
  ~AscendDeviceContext() override = default;

  void Initialize() override;

  void InitializeForAclop() const;

  void Destroy() override;

 private:
  DISABLE_COPY_AND_ASSIGN(AscendDeviceContext);

  mutable bool initializedAclop_{false};
  pid_t pid_; // Indicates the process id which creates the context.
};
} // namespace ascend
} // namespace device
} // namespace fxrt

#endif // FXRT_SRC_HARDWARE_ASCEND_ASCEND_DEVICE_CONTEXT_H_
