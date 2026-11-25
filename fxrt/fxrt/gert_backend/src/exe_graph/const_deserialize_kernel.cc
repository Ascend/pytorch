/* Copyright 2026 Huawei Technologies Co., Ltd */

#include "exe_graph/const_value_serialization.h"
#include "exe_graph/fxrt_kernel_export.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "acl/acl_rt.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/op_execute_context.h"
#include "exe_graph/runtime/runtime_tensor.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor_data.h"
#include "ge/ge_ir_build.h"
#include "lowering/aicore/aicore_rt_layout.h"
#include "runtime/rt_external_base.h"

namespace {

struct TensorDataPayload {
  uint64_t address;
  uint64_t size;
  int32_t placement;
  int64_t stream_id;
};

struct TensorConstPayload {
  uint64_t address;
  uint64_t size;
  int32_t placement;
  int32_t data_type;
  int32_t origin_format;
  int32_t storage_format;
  uint64_t expand_dims_size;
  uint64_t expand_dims_mask;
  uint32_t origin_dim_num;
  uint32_t storage_dim_num;
  int64_t origin_dims[gert::Shape::kMaxDimNum];
  int64_t storage_dims[gert::Shape::kMaxDimNum];
};

struct DfxExeArgPayload {
  bool need_assert;
  bool need_print;
  int64_t buffer_size;
};

struct MixCoreArgsPayload {
  uint32_t mix_type;
  uint32_t all_core_num;
  uint32_t vec_core_num;
  uint32_t ai_core_num;
};

struct LaunchKernelAttrPayload {
  int32_t id;
  uint32_t value;
};

enum AclRtLaunchKernelAttrIdCompat : int32_t {
  kAclRtLaunchKernelAttrSchemMode = 1,
  kAclRtLaunchKernelAttrDynUbufSize = 2,
  kAclRtLaunchKernelAttrEngineType = 3,
  kAclRtLaunchKernelAttrBlockDimOffset = 4,
  kAclRtLaunchKernelAttrEnableProfiling = 9,
};

union AclRtLaunchKernelAttrValueCompat {
  uint8_t schemMode;
  uint32_t dynUBufSize;
  aclrtEngineType engineType;
  uint32_t blockDimOffset;
  uint8_t enableProfiling;
  uint32_t reserved[4];
};

struct AclRtLaunchKernelAttrCompat {
  AclRtLaunchKernelAttrIdCompat id;
  AclRtLaunchKernelAttrValueCompat value;
};

static_assert(sizeof(AclRtLaunchKernelAttrCompat) == sizeof(aclrtLaunchKernelAttr), "ACL launch attr ABI changed");

template <typename T>
bool Read(const uint8_t *data, const size_t size, T *value) {
  if (data == nullptr || value == nullptr || size != sizeof(T)) {
    return false;
  }
  std::memcpy(value, data, sizeof(T));
  return true;
}

template <typename T>
void *ToInlineBits(const T &value) {
  static_assert(sizeof(T) <= sizeof(void *), "Chain inline value is larger than a pointer");
  void *bits = nullptr;
  std::memcpy(&bits, &value, sizeof(T));
  return bits;
}

bool ReadInput(gert::KernelContext *context, const fx_converter::FxrtConstType type, const uint8_t **payload,
               size_t *size) {
  (void)type;
  if (context == nullptr || payload == nullptr || size == nullptr || context->GetInputNum() != 1U ||
      context->GetOutputNum() != 1U) {
    return false;
  }
  const auto *const model_buffer = context->GetInputValue<const ge::ModelBufferData *>(0U);
  if (model_buffer == nullptr || model_buffer->length > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return false;
  }
  *payload = model_buffer->data.get();
  *size = static_cast<size_t>(model_buffer->length);
  return *payload != nullptr || *size == 0U;
}

template <typename T>
ge::graphStatus SetOwned(gert::KernelContext *context, T *storage) {
  if (storage == nullptr || context == nullptr) {
    delete storage;
    return ge::GRAPH_FAILED;
  }
  auto *const output = context->GetOutput(0U);
  if (output == nullptr) {
    delete storage;
    return ge::GRAPH_FAILED;
  }
  output->SetWithDefaultDeleter(storage);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus RestoreStorageShape(const uint8_t *payload, const size_t size, gert::KernelContext *context) {
  constexpr size_t kShapeBytes = sizeof(uint32_t) + sizeof(int64_t) * gert::Shape::kMaxDimNum;
  constexpr size_t kExpectedBytes = kShapeBytes * 2U;
  if (payload == nullptr || size != kExpectedBytes) {
    return ge::GRAPH_PARAM_INVALID;
  }
  uint32_t origin_dim_num = 0U;
  uint32_t storage_dim_num = 0U;
  std::memcpy(&origin_dim_num, payload, sizeof(origin_dim_num));
  std::memcpy(&storage_dim_num, payload + kShapeBytes, sizeof(storage_dim_num));
  if (origin_dim_num > gert::Shape::kMaxDimNum || storage_dim_num > gert::Shape::kMaxDimNum) {
    return ge::GRAPH_PARAM_INVALID;
  }
  auto *const shape = new (std::nothrow) gert::StorageShape();
  if (shape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  shape->MutableOriginShape().SetDimNum(origin_dim_num);
  shape->MutableStorageShape().SetDimNum(storage_dim_num);
  for (size_t i = 0U; i < origin_dim_num; ++i) {
    int64_t dim = 0;
    std::memcpy(&dim, payload + sizeof(uint32_t) + i * sizeof(int64_t), sizeof(dim));
    shape->MutableOriginShape().SetDim(i, dim);
  }
  for (size_t i = 0U; i < storage_dim_num; ++i) {
    int64_t dim = 0;
    std::memcpy(&dim, payload + kShapeBytes + sizeof(uint32_t) + i * sizeof(int64_t), sizeof(dim));
    shape->MutableStorageShape().SetDim(i, dim);
  }
  return SetOwned(context, shape);
}

ge::graphStatus RestoreTensorData(const uint8_t *payload, const size_t size, gert::KernelContext *context,
                                  const bool with_wrapper) {
  TensorDataPayload serialized{};
  if (!Read(payload, size, &serialized) || serialized.address > static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max()) ||
      serialized.size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const auto placement = static_cast<gert::TensorPlacement>(serialized.placement);
  if ((placement < gert::kOnDeviceHbm || placement >= gert::kTensorPlacementEnd) &&
      !(placement == gert::kTensorPlacementEnd && serialized.address == 0U && serialized.size == 0U)) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const auto address = reinterpret_cast<void *>(static_cast<uintptr_t>(serialized.address));
  const auto data_size = static_cast<size_t>(serialized.size);
  if (with_wrapper) {
    // Serialized constants borrow their storage and use the default stream id.
    // The default constructor and setters are provided by the CANN headers.
    if (serialized.stream_id != -1) {
      return ge::GRAPH_PARAM_INVALID;
    }
    auto data = std::unique_ptr<gert::GertTensorData>(new (std::nothrow) gert::GertTensorData());
    if (data == nullptr) {
      return ge::GRAPH_FAILED;
    }
    const auto ret = data->MutableTensorData().SetAddr(address, nullptr);
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    data->SetSize(data_size);
    data->SetPlacement(placement);
    return SetOwned(context, data.release());
  }
  auto *const data = new (std::nothrow) gert::TensorData(address, nullptr, data_size, placement);
  return SetOwned(context, data);
}

ge::graphStatus RestoreTensor(const uint8_t *payload, const size_t size, gert::KernelContext *context) {
  TensorConstPayload serialized{};
  if (!Read(payload, size, &serialized) || serialized.address > static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max()) ||
      serialized.size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      serialized.origin_dim_num > gert::Shape::kMaxDimNum || serialized.storage_dim_num > gert::Shape::kMaxDimNum ||
      serialized.expand_dims_size > gert::ExpandDimsType::kMaxExpandSize) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const bool no_data = serialized.address == 0U && serialized.size == 0U;
  const auto placement = static_cast<gert::TensorPlacement>(serialized.placement);
  if ((placement < gert::kOnDeviceHbm || placement >= gert::kTensorPlacementEnd) &&
      !(placement == gert::kTensorPlacementEnd && no_data)) {
    return ge::GRAPH_PARAM_INVALID;
  }
  gert::StorageShape shape;
  shape.MutableOriginShape().SetDimNum(serialized.origin_dim_num);
  shape.MutableStorageShape().SetDimNum(serialized.storage_dim_num);
  for (size_t i = 0U; i < serialized.origin_dim_num; ++i) {
    shape.MutableOriginShape().SetDim(i, serialized.origin_dims[i]);
  }
  for (size_t i = 0U; i < serialized.storage_dim_num; ++i) {
    shape.MutableStorageShape().SetDim(i, serialized.storage_dims[i]);
  }
  const uint64_t expand_bits =
      (serialized.expand_dims_size << gert::ExpandDimsType::kMaxExpandSize) |
      (serialized.expand_dims_mask & ((uint64_t{1} << 56U) - 1U));
  const gert::StorageFormat format(static_cast<ge::Format>(serialized.origin_format),
                                   static_cast<ge::Format>(serialized.storage_format),
                                   gert::ExpandDimsType(static_cast<int64_t>(expand_bits)));
  auto *const tensor = new (std::nothrow) gert::Tensor(
      shape, format, placement, static_cast<ge::DataType>(serialized.data_type),
      reinterpret_cast<void *>(static_cast<uintptr_t>(serialized.address)));
  if (tensor == nullptr) {
    return ge::GRAPH_FAILED;
  }
  tensor->SetSize(static_cast<size_t>(serialized.size));
  return SetOwned(context, tensor);
}

template <typename T>
ge::graphStatus RestoreContinuousVector(const uint8_t *payload, const size_t size, gert::KernelContext *context) {
  if (payload == nullptr || size < sizeof(uint64_t)) {
    return ge::GRAPH_PARAM_INVALID;
  }
  uint64_t wire_count = 0U;
  std::memcpy(&wire_count, payload, sizeof(wire_count));
  if (wire_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const size_t count = static_cast<size_t>(wire_count);
  if (count > (std::numeric_limits<size_t>::max() - sizeof(gert::ContinuousVector)) / sizeof(T) ||
      sizeof(uint64_t) + count * sizeof(T) != size) {
    return ge::GRAPH_PARAM_INVALID;
  }
  size_t total_size = 0U;
  auto storage = gert::ContinuousVector::Create<T>(count, total_size);
  if (storage == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *const vector = reinterpret_cast<gert::ContinuousVector *>(storage.get());
  if (vector->SetSize(count) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  const size_t value_bytes = count * sizeof(T);
  if (value_bytes != 0U) {
    std::memcpy(vector->MutableData(), payload + sizeof(wire_count), value_bytes);
  }
  auto *const output = context->GetOutput(0U);
  if (output == nullptr) {
    return ge::GRAPH_FAILED;
  }
  output->SetWithDefaultDeleter<uint8_t[]>(storage.release());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus RestoreContinuousVectorVectorInt32(const uint8_t *payload, const size_t size,
                                                    gert::KernelContext *context) {
  if (payload == nullptr || size < sizeof(uint64_t)) {
    return ge::GRAPH_PARAM_INVALID;
  }
  uint64_t wire_count = 0U;
  std::memcpy(&wire_count, payload, sizeof(wire_count));
  if (wire_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const size_t count = static_cast<size_t>(wire_count);
  if (count > (size - sizeof(wire_count)) / sizeof(uint64_t)) {
    return ge::GRAPH_PARAM_INVALID;
  }
  size_t cursor = sizeof(wire_count);
  size_t total_size = gert::ContinuousVectorVector::GetOverHeadLength(count);
  std::vector<size_t> inner_counts;
  inner_counts.reserve(count);
  for (size_t i = 0U; i < count; ++i) {
    if (cursor > size || sizeof(uint64_t) > size - cursor) {
      return ge::GRAPH_PARAM_INVALID;
    }
    uint64_t wire_inner_count = 0U;
    std::memcpy(&wire_inner_count, payload + cursor, sizeof(wire_inner_count));
    cursor += sizeof(wire_inner_count);
    if (wire_inner_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      return ge::GRAPH_PARAM_INVALID;
    }
    const size_t inner_count = static_cast<size_t>(wire_inner_count);
    size_t element_bytes = 0U;
    size_t encoded_size = 0U;
    size_t vector_storage_size = 0U;
    if (__builtin_mul_overflow(inner_count, sizeof(int32_t), &element_bytes) ||
        __builtin_add_overflow(sizeof(uint64_t), element_bytes, &encoded_size) || encoded_size > size - cursor ||
        __builtin_add_overflow(sizeof(gert::ContinuousVector), element_bytes, &vector_storage_size) ||
        __builtin_add_overflow(total_size, vector_storage_size, &total_size)) {
      return ge::GRAPH_PARAM_INVALID;
    }
    inner_counts.push_back(inner_count);
    cursor += element_bytes;
  }
  if (cursor != size) {
    return ge::GRAPH_PARAM_INVALID;
  }

  auto storage = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]);
  if (storage == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *const vector_vector = new (storage.get()) gert::ContinuousVectorVector();
  vector_vector->Init(count);
  cursor = sizeof(wire_count);
  for (const size_t inner_count : inner_counts) {
    cursor += sizeof(uint64_t);
    auto *const inner = vector_vector->Add<int32_t>(inner_count);
    if (inner == nullptr) {
      return ge::GRAPH_FAILED;
    }
    const size_t value_bytes = inner_count * sizeof(int32_t);
    if (value_bytes != 0U) {
      std::memcpy(inner->MutableData(), payload + cursor, value_bytes);
      cursor += value_bytes;
    }
  }
  auto *const output = context->GetOutput(0U);
  if (output == nullptr) {
    return ge::GRAPH_FAILED;
  }
  output->SetWithDefaultDeleter<uint8_t[]>(storage.release());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus RestoreByteArray(const uint8_t *payload, const size_t size, gert::KernelContext *context) {
  if (payload == nullptr || size == 0U) {
    return ge::GRAPH_PARAM_INVALID;
  }
  auto *storage = new (std::nothrow) uint8_t[size];
  if (storage == nullptr) {
    return ge::GRAPH_FAILED;
  }
  std::memcpy(storage, payload, size);
  auto *const output = context->GetOutput(0U);
  if (output == nullptr) {
    delete[] storage;
    return ge::GRAPH_FAILED;
  }
  output->SetWithDefaultDeleter<uint8_t[]>(storage);
  return ge::GRAPH_SUCCESS;
}

template <fx_converter::FxrtConstType Type>
ge::graphStatus DeserializeFxrtConst(gert::KernelContext *context) {
  const uint8_t *payload = nullptr;
  size_t size = 0U;
  if (!ReadInput(context, Type, &payload, &size)) {
    return ge::GRAPH_PARAM_INVALID;
  }
  auto *const output = context->GetOutput(0U);
  if (output == nullptr) {
    return ge::GRAPH_FAILED;
  }

  if constexpr (Type == fx_converter::FxrtConstType::kBytePtr) {
    output->Set(const_cast<uint8_t *>(payload), nullptr);
    return ge::GRAPH_SUCCESS;
  } else if constexpr (Type == fx_converter::FxrtConstType::kBool) {
    bool value = false;
    if (!Read(payload, size, &value)) return ge::GRAPH_PARAM_INVALID;
    output->Set(ToInlineBits(value), nullptr);
    return ge::GRAPH_SUCCESS;
  } else if constexpr (Type == fx_converter::FxrtConstType::kInt32T) {
    int32_t value = 0;
    if (!Read(payload, size, &value)) return ge::GRAPH_PARAM_INVALID;
    output->Set(ToInlineBits(value), nullptr);
    return ge::GRAPH_SUCCESS;
  } else if constexpr (Type == fx_converter::FxrtConstType::kInt64T) {
    int64_t value = 0;
    if (!Read(payload, size, &value)) return ge::GRAPH_PARAM_INVALID;
    output->Set(ToInlineBits(value), nullptr);
    return ge::GRAPH_SUCCESS;
  } else if constexpr (Type == fx_converter::FxrtConstType::kUInt32T ||
                       Type == fx_converter::FxrtConstType::kGeDataType ||
                       Type == fx_converter::FxrtConstType::kGertTensorPlacement ||
                       Type == fx_converter::FxrtConstType::kGertCoreTypeIndexPtr) {
    uint32_t value = 0U;
    if (!Read(payload, size, &value)) return ge::GRAPH_PARAM_INVALID;
    output->Set(ToInlineBits(value), nullptr);
    return ge::GRAPH_SUCCESS;
  } else if constexpr (Type == fx_converter::FxrtConstType::kUInt64T) {
    uint64_t value = 0U;
    if (!Read(payload, size, &value)) return ge::GRAPH_PARAM_INVALID;
    output->Set(ToInlineBits(value), nullptr);
    return ge::GRAPH_SUCCESS;
  } else if constexpr (Type == fx_converter::FxrtConstType::kSizeT) {
    size_t value = 0U;
    if (!Read(payload, size, &value)) return ge::GRAPH_PARAM_INVALID;
    output->Set(ToInlineBits(value), nullptr);
    return ge::GRAPH_SUCCESS;
  } else if constexpr (Type == fx_converter::FxrtConstType::kVoidPtr ||
                       Type == fx_converter::FxrtConstType::kKernelFuncPtr ||
                       Type == fx_converter::FxrtConstType::kRtStreamTPtr ||
                       Type == fx_converter::FxrtConstType::kGertAllocatorsPtr ||
                       Type == fx_converter::FxrtConstType::kGertTensorDataVectorPtr ||
                       Type == fx_converter::FxrtConstType::kGertTilingFwkDataPtr) {
    uintptr_t address = 0U;
    if (!Read(payload, size, &address)) return ge::GRAPH_PARAM_INVALID;
    output->Set(reinterpret_cast<void *>(address), nullptr);
    return ge::GRAPH_SUCCESS;
  } else if constexpr (Type == fx_converter::FxrtConstType::kGertStorageShapePtr) {
    return RestoreStorageShape(payload, size, context);
  } else if constexpr (Type == fx_converter::FxrtConstType::kGertTensorDataPtr) {
    return RestoreTensorData(payload, size, context, true);
  } else if constexpr (Type == fx_converter::FxrtConstType::kGertTensorDataObjectPtr) {
    return RestoreTensorData(payload, size, context, false);
  } else if constexpr (Type == fx_converter::FxrtConstType::kGertTensorPtr) {
    return RestoreTensor(payload, size, context);
  } else if constexpr (Type == fx_converter::FxrtConstType::kGertOpExecuteOptionsPtr) {
    if (size != sizeof(gert::OpExecuteOptions)) return ge::GRAPH_PARAM_INVALID;
    auto *const options = new (std::nothrow) gert::OpExecuteOptions{};
    if (options == nullptr) return ge::GRAPH_FAILED;
    std::memcpy(options, payload, sizeof(*options));
    return SetOwned(context, options);
  } else if constexpr (Type == fx_converter::FxrtConstType::kTypedContinuousVectorSizeTPtr) {
    return RestoreContinuousVector<size_t>(payload, size, context);
  } else if constexpr (Type == fx_converter::FxrtConstType::kTypedContinuousVectorInt64Ptr) {
    return RestoreContinuousVector<int64_t>(payload, size, context);
  } else if constexpr (Type == fx_converter::FxrtConstType::kGertContinuousVectorVectorInt32Ptr) {
    return RestoreContinuousVectorVectorInt32(payload, size, context);
  } else if constexpr (Type == fx_converter::FxrtConstType::kRtTaskCfgInfoTPtr) {
    rtTaskCfgInfo_t serialized{};
    if (!Read(payload, size, &serialized)) return ge::GRAPH_PARAM_INVALID;
    auto *const config = new (std::nothrow) rtTaskCfgInfo_t{};
    return SetOwned(context, config);
  } else if constexpr (Type == fx_converter::FxrtConstType::kAiCoreDfxExeArgPtr) {
    DfxExeArgPayload serialized{};
    if (!Read(payload, size, &serialized)) return ge::GRAPH_PARAM_INVALID;
    const fx_converter::aicore_rt::DfxExeArg value{serialized.need_assert, serialized.need_print,
                                                    serialized.buffer_size};
    return SetOwned(context, new (std::nothrow) fx_converter::aicore_rt::DfxExeArg(value));
  } else if constexpr (Type == fx_converter::FxrtConstType::kAiCoreComputeNodeDescPtr ||
                       Type == fx_converter::FxrtConstType::kAiCoreArgsInfosDescPtr) {
    return RestoreByteArray(payload, size, context);
  } else if constexpr (Type == fx_converter::FxrtConstType::kAclrtLaunchKernelAttrPtr) {
    constexpr size_t kAttrCount = 5U;
    if (payload == nullptr || size != sizeof(LaunchKernelAttrPayload) * kAttrCount) {
      return ge::GRAPH_PARAM_INVALID;
    }
    auto *const attrs = new (std::nothrow) AclRtLaunchKernelAttrCompat[kAttrCount]{};
    if (attrs == nullptr) return ge::GRAPH_FAILED;
    const auto *const serialized = reinterpret_cast<const LaunchKernelAttrPayload *>(payload);
    for (size_t i = 0U; i < kAttrCount; ++i) {
      attrs[i].id = static_cast<AclRtLaunchKernelAttrIdCompat>(serialized[i].id);
      switch (attrs[i].id) {
        case kAclRtLaunchKernelAttrSchemMode:
          if (serialized[i].value > std::numeric_limits<uint8_t>::max()) {
            delete[] attrs;
            return ge::GRAPH_PARAM_INVALID;
          }
          attrs[i].value.schemMode = static_cast<uint8_t>(serialized[i].value);
          break;
        case kAclRtLaunchKernelAttrDynUbufSize: attrs[i].value.dynUBufSize = serialized[i].value; break;
        case kAclRtLaunchKernelAttrEngineType:
          attrs[i].value.engineType = static_cast<aclrtEngineType>(serialized[i].value);
          break;
        case kAclRtLaunchKernelAttrBlockDimOffset: attrs[i].value.blockDimOffset = serialized[i].value; break;
        case kAclRtLaunchKernelAttrEnableProfiling:
          attrs[i].value.enableProfiling = serialized[i].value;
          break;
        default:
          delete[] attrs;
          return ge::GRAPH_PARAM_INVALID;
      }
    }
    auto *const output = context->GetOutput(0U);
    if (output == nullptr) {
      delete[] attrs;
      return ge::GRAPH_FAILED;
    }
    output->SetWithDefaultDeleter<AclRtLaunchKernelAttrCompat[]>(attrs);
    return ge::GRAPH_SUCCESS;
  } else if constexpr (Type == fx_converter::FxrtConstType::kAclrtLaunchKernelCfgPtr) {
    size_t num_attrs = 0U;
    if (!Read(payload, size, &num_attrs) || num_attrs != 5U) return ge::GRAPH_PARAM_INVALID;
    auto *const config = new (std::nothrow) aclrtLaunchKernelCfg{};
    if (config == nullptr) return ge::GRAPH_FAILED;
    config->numAttrs = num_attrs;
    return SetOwned(context, config);
  } else if constexpr (Type == fx_converter::FxrtConstType::kAiCoreMixCoreArgsPtr) {
    MixCoreArgsPayload serialized{};
    if (!Read(payload, size, &serialized) ||
        (serialized.mix_type != static_cast<uint32_t>(fx_converter::aicore_rt::MixType::kMixAiCore) &&
         serialized.mix_type != static_cast<uint32_t>(fx_converter::aicore_rt::MixType::kMixVectorCore))) {
      return ge::GRAPH_PARAM_INVALID;
    }
    fx_converter::aicore_rt::MixCoreArgs value{};
    value.mix_type = static_cast<fx_converter::aicore_rt::MixType>(serialized.mix_type);
    value.all_core_num = serialized.all_core_num;
    value.vec_core_num = serialized.vec_core_num;
    value.ai_core_num = serialized.ai_core_num;
    return SetOwned(context, new (std::nothrow) fx_converter::aicore_rt::MixCoreArgs(value));
  } else {
    return ge::GRAPH_FAILED;
  }
}

}  // namespace

#define FXRT_REGISTER_CONST_DECODER(Name, Type) extern "C" __attribute__((visibility("default"))) uint32_t fxrt_FxDeserializeConst##Name##_Run(gert::KernelContext *context) { return static_cast<uint32_t>(DeserializeFxrtConst<fx_converter::FxrtConstType::Type>(context)); }

FXRT_REGISTER_CONST_DECODER(BytePtr, kBytePtr);
FXRT_REGISTER_CONST_DECODER(Bool, kBool);
FXRT_REGISTER_CONST_DECODER(Int32, kInt32T);
FXRT_REGISTER_CONST_DECODER(Int64, kInt64T);
FXRT_REGISTER_CONST_DECODER(UInt32, kUInt32T);
FXRT_REGISTER_CONST_DECODER(UInt64, kUInt64T);
FXRT_REGISTER_CONST_DECODER(SizeT, kSizeT);
FXRT_REGISTER_CONST_DECODER(GeDataType, kGeDataType);
FXRT_REGISTER_CONST_DECODER(TensorPlacement, kGertTensorPlacement);
FXRT_REGISTER_CONST_DECODER(CoreTypeIndex, kGertCoreTypeIndexPtr);
FXRT_REGISTER_CONST_DECODER(VoidPtr, kVoidPtr);
FXRT_REGISTER_CONST_DECODER(KernelFuncPtr, kKernelFuncPtr);
FXRT_REGISTER_CONST_DECODER(RtStreamPtr, kRtStreamTPtr);
FXRT_REGISTER_CONST_DECODER(StorageShape, kGertStorageShapePtr);
FXRT_REGISTER_CONST_DECODER(Tensor, kGertTensorPtr);
FXRT_REGISTER_CONST_DECODER(GertTensorData, kGertTensorDataPtr);
FXRT_REGISTER_CONST_DECODER(TensorData, kGertTensorDataObjectPtr);
FXRT_REGISTER_CONST_DECODER(OpExecuteOptions, kGertOpExecuteOptionsPtr);
FXRT_REGISTER_CONST_DECODER(ContinuousVectorSizeT, kTypedContinuousVectorSizeTPtr);
FXRT_REGISTER_CONST_DECODER(ContinuousVectorInt64, kTypedContinuousVectorInt64Ptr);
FXRT_REGISTER_CONST_DECODER(ContinuousVectorVectorInt32, kGertContinuousVectorVectorInt32Ptr);
FXRT_REGISTER_CONST_DECODER(TaskConfig, kRtTaskCfgInfoTPtr);
FXRT_REGISTER_CONST_DECODER(DfxExeArg, kAiCoreDfxExeArgPtr);
FXRT_REGISTER_CONST_DECODER(ComputeNodeDesc, kAiCoreComputeNodeDescPtr);
FXRT_REGISTER_CONST_DECODER(ArgsInfosDesc, kAiCoreArgsInfosDescPtr);
FXRT_REGISTER_CONST_DECODER(LaunchKernelAttrs, kAclrtLaunchKernelAttrPtr);
FXRT_REGISTER_CONST_DECODER(LaunchKernelConfig, kAclrtLaunchKernelCfgPtr);
FXRT_REGISTER_CONST_DECODER(MixCoreArgs, kAiCoreMixCoreArgsPtr);
FXRT_REGISTER_CONST_DECODER(AllocatorsPtr, kGertAllocatorsPtr);
FXRT_REGISTER_CONST_DECODER(TensorDataVectorPtr, kGertTensorDataVectorPtr);
FXRT_REGISTER_CONST_DECODER(TilingFwkDataPtr, kGertTilingFwkDataPtr);

#undef FXRT_REGISTER_CONST_DECODER
