#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <type_traits>

namespace fx_converter {

// FXRT-owned semantic tags for Const payloads. They are not GE enum values.
enum class FxrtConstType : int32_t {
  kUnknown = 0,
  kBytePtr,
  kCStringPtr,
  kBool,
  kInt32T,
  kInt64T,
  kUInt32T,
  kUInt64T,
  kSizeT,
  kGeDataType,
  kGertTensorPlacement,
  kVoidPtr,
  kKernelFuncPtr,
  kRtStreamTPtr,
  kGertStorageShapePtr,
  kGertTensorPtr,
  kGertTensorDataPtr,
  kGertTensorDataObjectPtr,
  kReservedConstType18,
  kGertOpExecuteOptionsPtr,
  kTypedContinuousVectorSizeTPtr,
  kTypedContinuousVectorInt64Ptr,
  kReservedConstType22,
  kGertContinuousVectorVectorInt32Ptr,
  kRtTaskCfgInfoTPtr,
  kAiCoreDfxExeArgPtr,
  kAiCoreComputeNodeDescPtr,
  kAiCoreArgsInfosDescPtr,
  kAclrtLaunchKernelAttrPtr,
  kAclrtLaunchKernelCfgPtr,
  kAiCoreMixCoreArgsPtr,
  kReservedConstType31,
  kReservedConstType32,
  kReservedConstType33,
  kReservedConstType34,
  kGertAllocatorsPtr,
  kGertTensorDataVectorPtr,
  kGertTilingFwkDataPtr,
  kGertCoreTypeIndexPtr,
};

constexpr const char* kCustomConstTypeAttr = "custom_const_type";

// Converts the construction-time C++ representation into the byte payload that
// the corresponding FxrtDeserializeConst* kernel restores during Init.
bool SerializeFxrtConst(FxrtConstType type, const uint8_t* data, size_t size, std::vector<uint8_t>* serialized);
const char* FxrtConstDecoderType(FxrtConstType type);
bool IsFxrtConstInlineInt64(FxrtConstType type);
bool IsFxrtConstPassedThrough(FxrtConstType type, bool is_string);

template <typename T>
constexpr FxrtConstType InferFxrtConstType(const bool is_string) {
  using ValueType = std::remove_cv_t<T>;
  if (is_string || std::is_same_v<ValueType, char>) {
    return FxrtConstType::kCStringPtr;
  }
  if constexpr (std::is_same_v<ValueType, bool>) {
    return FxrtConstType::kBool;
  } else if constexpr (std::is_same_v<ValueType, int32_t>) {
    return FxrtConstType::kInt32T;
  } else if constexpr (std::is_same_v<ValueType, int64_t>) {
    return FxrtConstType::kInt64T;
  } else if constexpr (std::is_same_v<ValueType, uint32_t>) {
    return FxrtConstType::kUInt32T;
  } else if constexpr (std::is_same_v<ValueType, uint64_t>) {
    return FxrtConstType::kUInt64T;
  } else if constexpr (std::is_pointer_v<ValueType>) {
    if constexpr (std::is_function_v<std::remove_pointer_t<ValueType>>) {
      return FxrtConstType::kKernelFuncPtr;
    }
    return FxrtConstType::kVoidPtr;
  } else {
    return FxrtConstType::kBytePtr;
  }
}

} // namespace fx_converter
