#ifndef OPS_ASCEND_HCCL_KERNEL_H_
#define OPS_ASCEND_HCCL_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <condition_variable>

#include "ops/operator.h"
#include "ops/ascend/hccl/hcom_utils.h"
#include "hccl/hccl_types.h"

namespace fxrt {
namespace ops {
class HcclKernel {
 public:
  HcclKernel();
  ~HcclKernel() = default;

 public:
  HcclDataType hcclDataType_;
  uint64_t hcclCount_;
  uint32_t rootId_;
  std::string group_;
  HcclComm comm_;
  std::string hcclInnerCommName_;
};

} // namespace ops
} // namespace fxrt
#endif // OPS_ASCEND_HCCL_KERNEL_H_
