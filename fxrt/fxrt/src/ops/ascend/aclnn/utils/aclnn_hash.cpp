#include <cstdint>
#include <vector>
#include "ops/ascend/aclnn/utils/hash_buf.h"

#include "ops/ascend/aclnn/utils/aclnn_hash.h"

namespace fxrt {
namespace ops {
constexpr size_t kSizeFive = 5;
DA_API void GatherHash(const ir::TensorPtr& tensor) {
  if (tensor == nullptr || tensor->Dtype().value == ir::DataType::Type::Unknown) {
    MemcpyToBuf("None", kSizeFive);
    return;
  }

  // view shape
  const auto& shape = tensor->Shape();
  if (!shape.empty()) {
    MemcpyToBuf(shape.data(), static_cast<int64_t>(tensor->Dim() * sizeof(int64_t)));
  }

  // storage shape
  const auto& storageShape = tensor->StorageShape();
  if (!storageShape.empty()) {
    MemcpyToBuf(storageShape.data(), static_cast<int64_t>(storageShape.size() * sizeof(int64_t)));
  }

  // dtype
  auto dtype = tensor->Dtype().value;
  MemcpyToBuf(&dtype, sizeof(int8_t));

  // strides
  const auto& strides = tensor->Strides();
  if (!strides.empty()) {
    MemcpyToBuf(strides.data(), static_cast<int64_t>(strides.size() * sizeof(int64_t)));
  }

  // offset
  auto offset = tensor->StorageOffset();
  MemcpyToBuf(&offset, sizeof(int64_t));
}

} // namespace ops
} // namespace fxrt
