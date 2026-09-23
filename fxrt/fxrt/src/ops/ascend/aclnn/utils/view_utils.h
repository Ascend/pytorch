#ifndef __OPS_ASCEND_ACLNN_UTILS_VIEW_UTILS_H__
#define __OPS_ASCEND_ACLNN_UTILS_VIEW_UTILS_H__

#include <cstdint>
#include <optional>
#include <vector>
#include "common/visible.h"
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"

namespace fxrt {
namespace ops {
DA_API std::optional<std::vector<int64_t>> CalculateViewStrides(
    const std::vector<int64_t>& curShape,
    const std::vector<int64_t>& curStrides,
    const std::vector<int64_t>& newShape);

DA_API std::vector<int64_t> CalculateStrides(const std::vector<int64_t>& shape);

DA_API int64_t DynamicDimWrap(int64_t dim, int64_t dimPostExpr, bool wrapScalar = false);

DA_API std::vector<int64_t> GetTensorStrides(const ir::TensorPtr& tensorPtr);

DA_API void CheckViewMetaDataChangeForFormat(const ir::TensorPtr& inputTensorPtr, const std::vector<int64_t>& newShape);

DA_API void UpdateTensorViewInfo(
    const ir::TensorPtr& inputTensorPtr,
    const ir::TensorPtr& outputTensorPtr,
    const std::vector<int64_t>& newShape,
    const std::vector<int64_t>& newStrides,
    size_t newStorageOffset);

DA_API inline void UpdateTensorViewInfo(
    const ir::TensorPtr& inputTensorPtr,
    const ir::TensorPtr& outputTensorPtr,
    const std::vector<int64_t>& newShape,
    const std::vector<int64_t>& newStrides) {
  UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, newShape, newStrides, inputTensorPtr->StorageOffset());
}

DA_API std::vector<std::pair<uint32_t, uint32_t>> GenerateOutputInputRefPair(const ir::Value* output);
} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_UTILS_VIEW_UTILS_H__
