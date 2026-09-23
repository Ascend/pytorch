#include "ops/ascend/hccl/hccl_kernel.h"

#include <map>
#include <set>
#include <unordered_set>

#include "ops/ascend/hccl/hccl_adapter.h"
#include "ops/ascend/hccl/hcom_utils.h"

namespace fxrt {
namespace ops {

HcclKernel::HcclKernel() : hcclCount_(0), rootId_(0), comm_(nullptr) {}

} // namespace ops
} // namespace fxrt
