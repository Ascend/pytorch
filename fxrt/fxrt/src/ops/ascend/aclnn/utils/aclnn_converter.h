#ifndef __OPS_ASCEND_ACLNN_UTILS_ACLNN_CONVERTER_H__
#define __OPS_ASCEND_ACLNN_UTILS_ACLNN_CONVERTER_H__

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <tuple>
#include <string>
#include <optional>
#include <type_traits>

#include "ir/value/value.h"
#include "ops/utils/op_constants.h"
#include "ops/utils/utils.h"
#include "ops/ascend/aclnn/utils/aclnn_common_meta.h"
#include "ops/ascend/aclnn/utils/opapi_lib_loader.h"

namespace fxrt {
namespace ops {
// Convert dtype
DA_API aclDataType Convert(ir::DataType::Type dtype);

// Convert value to aclScalar
DA_API aclScalar* Convert(const ir::Value* value);

// Convert ValuePtr to aclScalar
inline aclScalar* Convert(const ir::ValuePtr& value) {
  return Convert(value.get());
}

// Convert tensor
inline aclTensor* Convert(const ir::TensorPtr& tensor) {
  static const auto aclCreateTensor = GET_ACLNN_COMMON_META_FUNC(aclCreateTensor);
  CHECK_IF_NULL(aclCreateTensor);
  if (tensor == nullptr || tensor->Dtype().value == ir::DataType::Type::Unknown) {
    return nullptr;
  }

  auto aclDtype = Convert(tensor->Dtype().value);
  aclFormat format = ACL_FORMAT_ND;
  std::vector<int64_t> storageDims;
  if (!IsTensorBaseFormat(tensor)) {
    format = static_cast<aclFormat>(tensor->Format());
    if (aclDtype != ACL_STRING) {
      storageDims = tensor->StorageShape();
    }
  } else {
    switch (tensor->Dim()) {
      case kDim3:
        format = ACL_FORMAT_NCL;
        break;
      case kDim4:
        format = ACL_FORMAT_NCHW;
        break;
      case kDim5:
        format = ACL_FORMAT_NCDHW;
        break;
      default:
        format = ACL_FORMAT_ND;
    }
    if (aclDtype != ACL_STRING) {
      storageDims.emplace_back(tensor->GetStorage()->SizeBytes() / tensor->Dtype().GetSize());
    }
  }

  RT_VLOG(VL_OPS) << "Create aclTensor, viewShape=" << tensor->Shape() << ", strides=" << tensor->Strides()
                  << ", StorageOffset=" << tensor->StorageOffset() << ", storageShape=" << tensor->StorageShape()
                  << ", storageDims=" << storageDims
                  << ", format=" << ir::FormatEnumToStr(static_cast<ir::MemoryFormat>(format));

  return aclCreateTensor(
      tensor->Shape().data(),
      tensor->Dim(),
      aclDtype,
      tensor->Strides().data(),
      tensor->StorageOffset(),
      format,
      storageDims.data(),
      storageDims.size(),
      tensor->GetStorage()->Data());
}

inline aclTensor* Convert(const std::optional<ir::TensorPtr>& tensorOpt) {
  if (tensorOpt.has_value()) {
    return Convert(tensorOpt.value());
  }
  return nullptr;
}

inline aclTensorList* Convert(const std::vector<ir::TensorPtr>& tensorList) {
  if (tensorList.empty()) {
    RT_VLOG(VL_OPS) << "tensorList is empty";
  }
  static const auto aclCreateTensorList = GET_ACLNN_COMMON_META_FUNC(aclCreateTensorList);
  std::vector<aclTensor*> aclTensorList;
  for (const auto& tensor : tensorList) {
    (void)aclTensorList.emplace_back(Convert(tensor));
  }
  return aclCreateTensorList(aclTensorList.data(), aclTensorList.size());
}

// Convert scalar
template <typename T, typename = std::enable_if_t<std::is_scalar_v<T>>>
T Convert(T value) {
  return value;
}

inline const char* Convert(const std::string& str) {
  return str.c_str();
}

inline const char* Convert(const std::optional<std::string>& strOpt) {
  if (strOpt.has_value()) {
    return Convert(strOpt.value());
  }
  return nullptr;
}

inline aclIntArray* Convert(const std::vector<int64_t>& intList) {
  if (intList.empty()) {
    return nullptr;
  }
  static const auto aclCreateIntArray = GET_ACLNN_COMMON_META_FUNC(aclCreateIntArray);
  CHECK_IF_NULL(aclCreateIntArray);
  return aclCreateIntArray(intList.data(), intList.size());
}

inline aclIntArray* Convert(const std::optional<std::vector<int64_t>>& intListOpt) {
  if (intListOpt.has_value()) {
    return Convert(intListOpt.value());
  }
  return nullptr;
}

inline aclIntArray* Convert(const std::pair<std::vector<int64_t>, bool>& intListPair) {
  return Convert(intListPair.first);
}

inline aclBoolArray* Convert(const std::vector<uint8_t>& boolList) {
  static const auto aclCreateBoolArray = GET_ACLNN_COMMON_META_FUNC(aclCreateBoolArray);
  CHECK_IF_NULL(aclCreateBoolArray);
  return aclCreateBoolArray(reinterpret_cast<const bool*>(boolList.data()), boolList.size());
}

inline aclBoolArray* Convert(const std::optional<std::vector<uint8_t>>& boolListOpt) {
  if (boolListOpt.has_value()) {
    return Convert(boolListOpt.value());
  }
  return nullptr;
}

inline aclFloatArray* Convert(const std::vector<float>& floatList) {
  static const auto aclCreateFloatArray = GET_ACLNN_COMMON_META_FUNC(aclCreateFloatArray);
  CHECK_IF_NULL(aclCreateFloatArray);
  return aclCreateFloatArray(floatList.data(), floatList.size());
}

inline aclFloatArray* Convert(const std::optional<std::vector<float>>& floatListOpt) {
  if (floatListOpt.has_value()) {
    return Convert(floatListOpt.value());
  }
  return nullptr;
}

// Main entry for convert
template <typename... Args>
constexpr auto ConvertParams(const Args&... args) {
  return std::make_tuple(Convert(args)...);
}

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_UTILS_ACLNN_CONVERTER_H__
