#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/GetAffinityCPUInfo.h"

namespace c10_npu {
// The irq binding affinity feature is implemented by modifying the
// /proc/irq/{id}/smp_affinity file. Therefore, this feature requires root
// privileges on the physical machine to take effect (the irqbalance daemon
// on the physical machine cannot be stopped within a docker container).
bool bindIrqAffinity(int device_id, const CoreIdList& irq_cores);

// When irq binding affinity is enabled, torch_npu uses the
// 'systemctl stop irqbalance' command to stop the irqbalance service,
// preventing the irqbalance service from conflicting with torch_npu's
// irq binding configuration. When the process exits normally,
// torch_npu uses the 'systemctl restart irqbalance' command to restart
// the irqbalance service. In extreme cases where the irqbalance service
// fails to restart, users need to manually restart the irqbalance
// service on the server using the 'sudo systemctl restart irqbalance' command.
void stopIrqbalance();

C10_NPU_API void restartIrqbalance();

} // namespace c10_npu
