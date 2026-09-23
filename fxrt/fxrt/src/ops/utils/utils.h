#ifndef __OPS_OP_BASE_OP_UTILS_H__
#define __OPS_OP_BASE_OP_UTILS_H__

#include <vector>

#include "common/common.h"
#include "ir/tensor/tensor.h"
#include "ir/tensor/format.h"

namespace fxrt {
namespace ops {
using ir::MemoryFormat;
/// Cache capacity for ATB; from env FXRT_ATB_CACHE_CAPACITY, default 64.
FXRT_EXPORT size_t GetAtbCacheCapacity();
/// Cache capacity for ACLNN; from env FXRT_ACLNN_CACHE_CAPACITY, default 10000.
FXRT_EXPORT size_t GetAclnnCacheCapacity();
FXRT_EXPORT void CalBroadCastShape(
    const std::vector<int64_t>& xShape,
    const std::vector<int64_t>& yShape,
    std::vector<int64_t>* broadcastShape);
FXRT_EXPORT bool IsBaseFormat(MemoryFormat format);
FXRT_EXPORT bool IsTensorBaseFormat(const ir::TensorPtr& tensor);
FXRT_EXPORT MemoryFormat GetBaseFormat(MemoryFormat format);
FXRT_EXPORT bool IsDefiniteTensorWhenMetaDataChanges(const ir::TensorPtr& tensor, const std::vector<int64_t>& shape);
} // namespace ops
} // namespace fxrt

#endif // __OPS_OP_BASE_OP_UTILS_H__
