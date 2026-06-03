#ifndef __TORCH_NPU_MSPTIINTERFACE__
#define __TORCH_NPU_MSPTIINTERFACE__

#include <third_party/mspti/mspti_activity.h>

namespace at_npu {
namespace native {

bool IsSupportMsptiFunc();

bool MsptiActivityIsEnabled(msptiActivityKind kind);

} // namespace native
} // namespace at_npu

#endif
