#ifndef FXRT_SRC_HARDWARE_CPU_CPU_DEVICE_CONTEXT_H_
#define FXRT_SRC_HARDWARE_CPU_CPU_DEVICE_CONTEXT_H_

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <mutex>
#include "hardware/hardware_abstract/device_context.h"
#include "hardware/cpu/res_manager/cpu_res_manager.h"
#include "common/visible.h"

namespace fxrt {
namespace device {
namespace cpu {

class FXRT_EXPORT CPUDeviceContext : public DeviceInterface<CPUResManager> {
 public:
  explicit CPUDeviceContext(const DeviceContextKey& deviceContextKey) : DeviceInterface(deviceContextKey) {}
  ~CPUDeviceContext() override = default;

  void Initialize() override;

  void Destroy() override;

 private:
  DISABLE_COPY_AND_ASSIGN(CPUDeviceContext);
};
} // namespace cpu
} // namespace device
} // namespace fxrt

#endif // fxrt_CCSRC_RUNTIME_HARDWARE_CPU_CPU_DEVICE_CONTEXT_H_
