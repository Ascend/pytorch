#include "exe_graph/const_value_serialization.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "exe_graph/runtime/runtime_tensor.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/op_execute_context.h"
#include "exe_graph/runtime/tensor_data.h"
#include "acl/acl_rt.h"
#include "lowering/aicore/aicore_rt_layout.h"
#include "runtime/rt_external_base.h"

namespace fx_converter {
namespace {

template <typename T>
bool Read(const uint8_t *data, const size_t size, T *value) {
  if (data == nullptr || value == nullptr || size != sizeof(T)) {
    return false;
  }
  std::memcpy(value, data, sizeof(T));
  return true;
}

template <typename T>
bool Write(const T &value, std::vector<uint8_t> *bytes) {
  if (bytes == nullptr) {
    return false;
  }
  bytes->resize(sizeof(T));
  std::memcpy(bytes->data(), &value, sizeof(T));
  return true;
}

bool CopyRaw(const uint8_t *data, const size_t size, std::vector<uint8_t> *bytes) {
  if (bytes == nullptr || (data == nullptr && size != 0U)) {
    return false;
  }
  if (size == 0U) {
    bytes->clear();
    return true;
  }
  bytes->assign(data, data == nullptr ? data : data + size);
  return true;
}

struct RawShape {
  size_t dim_num;
  int64_t dims[gert::Shape::kMaxDimNum];
  uint8_t reserved[40];
};

struct RawStorageShape {
  RawShape origin_shape;
  RawShape storage_shape;
  uint8_t reserved[40];
};

bool SerializeStorageShape(const uint8_t *data, const size_t size, std::vector<uint8_t> *bytes) {
  if (data == nullptr || size != sizeof(gert::StorageShape) || bytes == nullptr) {
    return false;
  }
  RawStorageShape raw{};
  static_assert(sizeof(raw) == sizeof(gert::StorageShape), "StorageShape layout changed");
  std::memcpy(&raw, data, sizeof(raw));
  if (raw.origin_shape.dim_num > gert::Shape::kMaxDimNum ||
      raw.storage_shape.dim_num > gert::Shape::kMaxDimNum) {
    return false;
  }
  constexpr size_t kSerializedSize = sizeof(uint32_t) * 2U + sizeof(int64_t) * gert::Shape::kMaxDimNum * 2U;
  bytes->assign(kSerializedSize, 0U);
  const auto write_shape = [bytes](const RawShape &shape, const size_t offset) {
    const auto count = static_cast<uint32_t>(shape.dim_num);
    std::memcpy(bytes->data() + offset, &count, sizeof(count));
    for (size_t i = 0U; i < shape.dim_num; ++i) {
      std::memcpy(bytes->data() + offset + sizeof(uint32_t) + i * sizeof(int64_t), &shape.dims[i], sizeof(int64_t));
    }
  };
  write_shape(raw.origin_shape, 0U);
  write_shape(raw.storage_shape, sizeof(uint32_t) + sizeof(int64_t) * gert::Shape::kMaxDimNum);
  return true;
}

struct TensorDataPayload {
  uint64_t address;
  uint64_t size;
  int32_t placement;
  int64_t stream_id;
};

struct RawTensorData {
  void *address;
  void *manager;
  size_t size;
  gert::TensorPlacement placement;
  uint32_t reserved_0;
  uint8_t reserved_1[40];
};

struct RawGertTensorData {
  uint8_t tensor_data[sizeof(gert::TensorData)];
  uintptr_t gert_mem_block;
  int64_t stream_id;
};

bool SerializeTensorData(const uint8_t *data, const size_t size, const bool with_wrapper,
                         std::vector<uint8_t> *bytes) {
  if (data == nullptr || bytes == nullptr) {
    return false;
  }
  RawTensorData raw{};
  int64_t stream_id = -1;
  if (with_wrapper) {
    if (size != sizeof(gert::GertTensorData)) {
      return false;
    }
    RawGertTensorData wrapper{};
    static_assert(sizeof(wrapper) == sizeof(gert::GertTensorData), "GertTensorData layout changed");
    std::memcpy(&wrapper, data, sizeof(wrapper));
    std::memcpy(&raw, wrapper.tensor_data, sizeof(raw));
    stream_id = wrapper.stream_id;
  } else {
    if (size != sizeof(gert::TensorData)) {
      return false;
    }
    std::memcpy(&raw, data, sizeof(raw));
  }
  const bool no_data = raw.address == nullptr && raw.size == 0U;
  const int32_t placement = static_cast<int32_t>(raw.placement);
  if (!((placement >= static_cast<int32_t>(gert::kOnDeviceHbm) &&
         placement < static_cast<int32_t>(gert::kTensorPlacementEnd)) ||
        (placement == static_cast<int32_t>(gert::kTensorPlacementEnd) && no_data))) {
    return false;
  }
  const TensorDataPayload payload{static_cast<uint64_t>(reinterpret_cast<uintptr_t>(raw.address)),
                                  static_cast<uint64_t>(raw.size),
                                  no_data ? static_cast<int32_t>(gert::kTensorPlacementEnd) : placement, stream_id};
  return Write(payload, bytes);
}

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

struct RawExpandDimsType {
  uint64_t bits;
  uint8_t reserved[40];
};

struct RawStorageFormat {
  int32_t origin_format;
  int32_t storage_format;
  RawExpandDimsType expand_dims_type;
  uint8_t reserved[40];
};

struct RawTensor {
  RawStorageShape storage_shape;
  RawStorageFormat storage_format;
  uint8_t version;
  uint8_t reserved[3];
  ge::DataType data_type;
  RawTensorData tensor_data;
  uint8_t reserved_field[40];
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
bool CopyObject(const uint8_t *data, const size_t size, std::vector<uint8_t> *bytes) {
  T value{};
  return Read(data, size, &value) && Write(value, bytes);
}

bool SerializeTensor(const uint8_t *data, const size_t size, std::vector<uint8_t> *bytes) {
  if (data == nullptr || size != sizeof(gert::Tensor) || bytes == nullptr) {
    return false;
  }
  RawTensor raw{};
  static_assert(sizeof(RawTensorData) == sizeof(gert::TensorData), "TensorData layout changed");
  static_assert(sizeof(RawTensor) == sizeof(gert::Tensor), "Tensor layout changed");
  std::memcpy(&raw, data, sizeof(raw));
  if (raw.version != static_cast<uint8_t>(gert::kTensorV1) || raw.tensor_data.manager != nullptr ||
      raw.storage_shape.origin_shape.dim_num > gert::Shape::kMaxDimNum ||
      raw.storage_shape.storage_shape.dim_num > gert::Shape::kMaxDimNum ||
      (raw.storage_format.expand_dims_type.bits >> gert::ExpandDimsType::kMaxExpandSize) > 0xffU) {
    return false;
  }
  const bool owns_no_data = raw.tensor_data.address == nullptr && raw.tensor_data.size == 0U;
  const auto placement = static_cast<int32_t>(raw.tensor_data.placement);
  const bool valid_placement =
      (placement >= static_cast<int32_t>(gert::kOnDeviceHbm) &&
       placement < static_cast<int32_t>(gert::kTensorPlacementEnd)) ||
      (placement == static_cast<int32_t>(gert::kTensorPlacementEnd) && owns_no_data);
  if (!valid_placement || raw.tensor_data.size > std::numeric_limits<uint64_t>::max()) {
    return false;
  }
  TensorConstPayload payload{};
  payload.address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(raw.tensor_data.address));
  payload.size = static_cast<uint64_t>(raw.tensor_data.size);
  payload.placement = owns_no_data ? static_cast<int32_t>(gert::kTensorPlacementEnd) : placement;
  payload.data_type = static_cast<int32_t>(raw.data_type);
  payload.origin_format = raw.storage_format.origin_format;
  payload.storage_format = raw.storage_format.storage_format;
  payload.expand_dims_size = raw.storage_format.expand_dims_type.bits >> gert::ExpandDimsType::kMaxExpandSize;
  payload.expand_dims_mask = raw.storage_format.expand_dims_type.bits & ((uint64_t{1} << 56U) - 1U);
  payload.origin_dim_num = static_cast<uint32_t>(raw.storage_shape.origin_shape.dim_num);
  payload.storage_dim_num = static_cast<uint32_t>(raw.storage_shape.storage_shape.dim_num);
  std::copy(raw.storage_shape.origin_shape.dims, raw.storage_shape.origin_shape.dims + gert::Shape::kMaxDimNum,
            payload.origin_dims);
  std::copy(raw.storage_shape.storage_shape.dims, raw.storage_shape.storage_shape.dims + gert::Shape::kMaxDimNum,
            payload.storage_dims);
  return Write(payload, bytes);
}

bool SerializeContinuousVector(const uint8_t *data, const size_t size, const size_t element_size,
                               std::vector<uint8_t> *bytes) {
  if (data == nullptr || size < sizeof(gert::ContinuousVector) || bytes == nullptr) {
    return false;
  }
  size_t capacity = 0U;
  size_t count = 0U;
  std::memcpy(&capacity, data, sizeof(capacity));
  std::memcpy(&count, data + sizeof(size_t), sizeof(count));
  if (count > capacity || capacity > (std::numeric_limits<size_t>::max() - sizeof(gert::ContinuousVector)) / element_size ||
      sizeof(gert::ContinuousVector) + capacity * element_size != size) {
    return false;
  }
  const size_t value_bytes = count * element_size;
  bytes->resize(sizeof(uint64_t) + value_bytes);
  const uint64_t count_wire = static_cast<uint64_t>(count);
  std::memcpy(bytes->data(), &count_wire, sizeof(count_wire));
  if (value_bytes > 0U) {
    constexpr size_t kDataOffset = sizeof(gert::ContinuousVector) - sizeof(size_t);
    std::memcpy(bytes->data() + sizeof(count_wire), data + kDataOffset, value_bytes);
  }
  return true;
}

bool SerializeContinuousVectorVectorInt32(const uint8_t *data, const size_t size, std::vector<uint8_t> *bytes) {
  if (data == nullptr || bytes == nullptr || size < sizeof(gert::ContinuousVectorVector)) {
    return false;
  }
  size_t capacity = 0U;
  size_t count = 0U;
  std::memcpy(&capacity, data, sizeof(capacity));
  std::memcpy(&count, data + sizeof(size_t), sizeof(count));
  if (capacity != count) {
    return false;
  }

  const size_t overhead = gert::ContinuousVectorVector::GetOverHeadLength(capacity);
  if (overhead > size) {
    return false;
  }
  constexpr size_t kFirstOffsetPosition = sizeof(gert::ContinuousVectorVector) - sizeof(size_t);
  constexpr size_t kVectorDataOffset = sizeof(gert::ContinuousVector) - sizeof(size_t);
  size_t expected_offset = overhead;
  size_t serialized_size = sizeof(uint64_t);
  for (size_t i = 0U; i < count; ++i) {
    size_t offset_position = 0U;
    size_t vector_offset = 0U;
    if (__builtin_mul_overflow(i, sizeof(size_t), &offset_position) ||
        __builtin_add_overflow(kFirstOffsetPosition, offset_position, &offset_position) ||
        offset_position > size || sizeof(vector_offset) > size - offset_position) {
      return false;
    }
    std::memcpy(&vector_offset, data + offset_position, sizeof(vector_offset));
    if (vector_offset != expected_offset || vector_offset > size ||
        sizeof(gert::ContinuousVector) > size - vector_offset) {
      return false;
    }
    size_t inner_capacity = 0U;
    size_t inner_count = 0U;
    std::memcpy(&inner_capacity, data + vector_offset, sizeof(inner_capacity));
    std::memcpy(&inner_count, data + vector_offset + sizeof(size_t), sizeof(inner_count));
    size_t element_bytes = 0U;
    size_t vector_size = 0U;
    size_t serialized_vector_size = 0U;
    if (inner_count > inner_capacity ||
        __builtin_mul_overflow(inner_capacity, sizeof(int32_t), &element_bytes) ||
        __builtin_add_overflow(sizeof(gert::ContinuousVector), element_bytes, &vector_size) ||
        __builtin_add_overflow(vector_offset, vector_size, &expected_offset) || expected_offset > size ||
        __builtin_mul_overflow(inner_count, sizeof(int32_t), &element_bytes) ||
        __builtin_add_overflow(sizeof(uint64_t), element_bytes, &serialized_vector_size) ||
        __builtin_add_overflow(serialized_size, serialized_vector_size, &serialized_size)) {
      return false;
    }
    const size_t inner_data_offset = vector_offset + kVectorDataOffset;
    const size_t inner_data_size = inner_count * sizeof(int32_t);
    if (inner_data_offset > expected_offset || inner_data_size > expected_offset - inner_data_offset) {
      return false;
    }
  }
  if (expected_offset != size) {
    return false;
  }

  bytes->resize(serialized_size);
  const uint64_t count_wire = static_cast<uint64_t>(count);
  std::memcpy(bytes->data(), &count_wire, sizeof(count_wire));
  size_t cursor = sizeof(count_wire);
  for (size_t i = 0U; i < count; ++i) {
    size_t vector_offset = 0U;
    std::memcpy(&vector_offset, data + kFirstOffsetPosition + i * sizeof(size_t), sizeof(vector_offset));
    size_t inner_capacity = 0U;
    size_t inner_count = 0U;
    std::memcpy(&inner_capacity, data + vector_offset, sizeof(inner_capacity));
    std::memcpy(&inner_count, data + vector_offset + sizeof(size_t), sizeof(inner_count));
    const uint64_t inner_count_wire = static_cast<uint64_t>(inner_count);
    std::memcpy(bytes->data() + cursor, &inner_count_wire, sizeof(inner_count_wire));
    cursor += sizeof(inner_count_wire);
    const size_t value_bytes = inner_count * sizeof(int32_t);
    if (value_bytes > 0U) {
      std::memcpy(bytes->data() + cursor, data + vector_offset + kVectorDataOffset, value_bytes);
      cursor += value_bytes;
    }
  }
  return cursor == serialized_size;
}

bool SerializeLaunchKernelAttrs(const uint8_t *data, const size_t size, std::vector<uint8_t> *bytes) {
  constexpr size_t kAttrCount = 5U;
  if (data == nullptr || size != kAttrCount * sizeof(AclRtLaunchKernelAttrCompat) || bytes == nullptr) {
    return false;
  }
  const int32_t expected_ids[] = {kAclRtLaunchKernelAttrSchemMode, kAclRtLaunchKernelAttrDynUbufSize,
                                  kAclRtLaunchKernelAttrEngineType, kAclRtLaunchKernelAttrBlockDimOffset,
                                  kAclRtLaunchKernelAttrEnableProfiling};
  LaunchKernelAttrPayload normalized[kAttrCount]{};
  for (size_t i = 0U; i < kAttrCount; ++i) {
    AclRtLaunchKernelAttrCompat attr{};
    std::memcpy(&attr, data + i * sizeof(attr), sizeof(attr));
    if (static_cast<int32_t>(attr.id) != expected_ids[i]) {
      return false;
    }
    uint32_t value = 0U;
    switch (attr.id) {
      case kAclRtLaunchKernelAttrSchemMode: value = attr.value.schemMode; break;
      case kAclRtLaunchKernelAttrDynUbufSize: value = attr.value.dynUBufSize; break;
      case kAclRtLaunchKernelAttrEngineType: value = static_cast<uint32_t>(attr.value.engineType); break;
      case kAclRtLaunchKernelAttrBlockDimOffset: value = attr.value.blockDimOffset; break;
      case kAclRtLaunchKernelAttrEnableProfiling: value = attr.value.enableProfiling; break;
      default: return false;
    }
    normalized[i] = {static_cast<int32_t>(attr.id), value};
  }
  bytes->resize(sizeof(normalized));
  std::memcpy(bytes->data(), normalized, sizeof(normalized));
  return true;
}

bool SerializeComputeNodeDesc(const uint8_t *data, const size_t size, std::vector<uint8_t> *bytes) {
  using ComputeNodeDesc = aicore_rt::ComputeNodeDesc;
  if (data == nullptr || size < sizeof(ComputeNodeDesc) || bytes == nullptr) {
    return false;
  }
  ComputeNodeDesc desc{};
  std::memcpy(&desc, data, sizeof(desc));
  size_t expected_size = 0U;
  if (__builtin_add_overflow(sizeof(ComputeNodeDesc), desc.compiled_args_size, &expected_size) || expected_size != size) {
    return false;
  }
  return CopyRaw(data, size, bytes);
}

bool SerializeArgsInfosDesc(const uint8_t *data, const size_t size, std::vector<uint8_t> *bytes) {
  using ArgsInfosDesc = aicore_rt::ArgsInfosDesc;
  if (data == nullptr || size < sizeof(ArgsInfosDesc) || bytes == nullptr) {
    return false;
  }
  std::vector<uint8_t> aligned(size);
  std::memcpy(aligned.data(), data, size);
  const auto *const desc = reinterpret_cast<const ArgsInfosDesc *>(aligned.data());
  size_t args_num = 0U;
  size_t args_bytes = 0U;
  size_t expected_size = 0U;
  if (__builtin_add_overflow(desc->input_args_info_num_, desc->output_args_info_num_, &args_num) ||
      __builtin_mul_overflow(args_num, sizeof(aicore_rt::ArgsInfo), &args_bytes) ||
      __builtin_add_overflow(sizeof(ArgsInfosDesc), args_bytes, &expected_size) ||
      desc->args_infos_size_ != args_bytes || expected_size != size) {
    return false;
  }
  return CopyRaw(data, size, bytes);
}

}  // namespace

bool IsFxrtConstInlineInt64(const FxrtConstType type) {
  return type == FxrtConstType::kInt64T;
}

bool IsFxrtConstPassedThrough(const FxrtConstType type, const bool is_string) {
  return is_string || type == FxrtConstType::kCStringPtr;
}

const char *FxrtConstDecoderType(const FxrtConstType type) {
  switch (type) {
    case FxrtConstType::kBytePtr: return "FxDeserializeConstBytePtr";
    case FxrtConstType::kBool: return "FxDeserializeConstBool";
    case FxrtConstType::kInt32T: return "FxDeserializeConstInt32";
    case FxrtConstType::kInt64T: return "FxDeserializeConstInt64";
    case FxrtConstType::kUInt32T: return "FxDeserializeConstUInt32";
    case FxrtConstType::kUInt64T: return "FxDeserializeConstUInt64";
    case FxrtConstType::kSizeT: return "FxDeserializeConstSizeT";
    case FxrtConstType::kGeDataType: return "FxDeserializeConstGeDataType";
    case FxrtConstType::kGertTensorPlacement: return "FxDeserializeConstTensorPlacement";
    case FxrtConstType::kGertCoreTypeIndexPtr: return "FxDeserializeConstCoreTypeIndex";
    case FxrtConstType::kVoidPtr: return "FxDeserializeConstVoidPtr";
    case FxrtConstType::kKernelFuncPtr: return "FxDeserializeConstKernelFuncPtr";
    case FxrtConstType::kRtStreamTPtr: return "FxDeserializeConstRtStreamPtr";
    case FxrtConstType::kGertStorageShapePtr: return "FxDeserializeConstStorageShape";
    case FxrtConstType::kGertTensorPtr: return "FxDeserializeConstTensor";
    case FxrtConstType::kGertTensorDataPtr: return "FxDeserializeConstGertTensorData";
    case FxrtConstType::kGertTensorDataObjectPtr: return "FxDeserializeConstTensorData";
    case FxrtConstType::kGertOpExecuteOptionsPtr: return "FxDeserializeConstOpExecuteOptions";
    case FxrtConstType::kTypedContinuousVectorSizeTPtr: return "FxDeserializeConstContinuousVectorSizeT";
    case FxrtConstType::kTypedContinuousVectorInt64Ptr: return "FxDeserializeConstContinuousVectorInt64";
    case FxrtConstType::kGertContinuousVectorVectorInt32Ptr: return "FxDeserializeConstContinuousVectorVectorInt32";
    case FxrtConstType::kRtTaskCfgInfoTPtr: return "FxDeserializeConstTaskConfig";
    case FxrtConstType::kAiCoreDfxExeArgPtr: return "FxDeserializeConstDfxExeArg";
    case FxrtConstType::kAiCoreComputeNodeDescPtr: return "FxDeserializeConstComputeNodeDesc";
    case FxrtConstType::kAiCoreArgsInfosDescPtr: return "FxDeserializeConstArgsInfosDesc";
    case FxrtConstType::kAclrtLaunchKernelAttrPtr: return "FxDeserializeConstLaunchKernelAttrs";
    case FxrtConstType::kAclrtLaunchKernelCfgPtr: return "FxDeserializeConstLaunchKernelConfig";
    case FxrtConstType::kAiCoreMixCoreArgsPtr: return "FxDeserializeConstMixCoreArgs";
    case FxrtConstType::kGertAllocatorsPtr: return "FxDeserializeConstAllocatorsPtr";
    case FxrtConstType::kGertTensorDataVectorPtr: return "FxDeserializeConstTensorDataVectorPtr";
    case FxrtConstType::kGertTilingFwkDataPtr: return "FxDeserializeConstTilingFwkDataPtr";
    case FxrtConstType::kCStringPtr:
    case FxrtConstType::kUnknown:
    default: return nullptr;
  }
}

bool SerializeFxrtConst(const FxrtConstType type, const uint8_t *data, const size_t size,
                        std::vector<uint8_t> *serialized) {
  if (serialized == nullptr || (data == nullptr && size != 0U)) {
    return false;
  }
  switch (type) {
    case FxrtConstType::kBytePtr:
      return CopyRaw(data, size, serialized);
    case FxrtConstType::kBool: {
      bool value = false;
      return Read(data, size, &value) && Write(value, serialized);
    }
    case FxrtConstType::kInt32T: {
      int32_t value = 0;
      return Read(data, size, &value) && Write(value, serialized);
    }
    case FxrtConstType::kInt64T: {
      int64_t value = 0;
      return Read(data, size, &value) && Write(value, serialized);
    }
    case FxrtConstType::kUInt32T:
    case FxrtConstType::kGeDataType:
    case FxrtConstType::kGertTensorPlacement:
    case FxrtConstType::kGertCoreTypeIndexPtr: {
      uint32_t value = 0U;
      return Read(data, size, &value) && Write(value, serialized);
    }
    case FxrtConstType::kUInt64T: {
      uint64_t value = 0U;
      if (size == sizeof(uint32_t)) {
        uint32_t narrow = 0U;
        if (!Read(data, size, &narrow)) return false;
        value = narrow;
      } else if (!Read(data, size, &value)) {
        return false;
      }
      return Write(value, serialized);
    }
    case FxrtConstType::kSizeT: {
      size_t value = 0U;
      if (size == sizeof(bool)) {
        bool narrow = false;
        if (!Read(data, size, &narrow)) return false;
        value = narrow ? 1U : 0U;
      } else if (size == sizeof(uint32_t)) {
        uint32_t narrow = 0U;
        if (!Read(data, size, &narrow)) return false;
        value = narrow;
      } else if (!Read(data, size, &value)) {
        return false;
      }
      return Write(value, serialized);
    }
    case FxrtConstType::kVoidPtr:
    case FxrtConstType::kKernelFuncPtr:
    case FxrtConstType::kRtStreamTPtr: {
      uintptr_t value = 0U;
      return Read(data, size, &value) && Write(value, serialized);
    }
    case FxrtConstType::kGertStorageShapePtr:
      return SerializeStorageShape(data, size, serialized);
    case FxrtConstType::kGertTensorPtr:
      return SerializeTensor(data, size, serialized);
    case FxrtConstType::kGertTensorDataPtr:
      return SerializeTensorData(data, size, true, serialized);
    case FxrtConstType::kGertTensorDataObjectPtr:
      return SerializeTensorData(data, size, false, serialized);
    case FxrtConstType::kGertOpExecuteOptionsPtr:
      return CopyObject<gert::OpExecuteOptions>(data, size, serialized);
    case FxrtConstType::kTypedContinuousVectorSizeTPtr:
      return SerializeContinuousVector(data, size, sizeof(size_t), serialized);
    case FxrtConstType::kTypedContinuousVectorInt64Ptr:
      return SerializeContinuousVector(data, size, sizeof(int64_t), serialized);
    case FxrtConstType::kGertContinuousVectorVectorInt32Ptr:
      return SerializeContinuousVectorVectorInt32(data, size, serialized);
    case FxrtConstType::kRtTaskCfgInfoTPtr: {
      if (data == nullptr || size != sizeof(rtTaskCfgInfo_t) ||
          !std::all_of(data, data + size, [](uint8_t value) { return value == 0U; })) {
        return false;
      }
      const rtTaskCfgInfo_t value{};
      return Write(value, serialized);
    }
    case FxrtConstType::kAiCoreDfxExeArgPtr: {
      aicore_rt::DfxExeArg source{};
      if (!Read(data, size, &source)) return false;
      const DfxExeArgPayload payload{source.need_assert, source.need_print, source.buffer_size};
      return Write(payload, serialized);
    }
    case FxrtConstType::kAiCoreComputeNodeDescPtr:
      return SerializeComputeNodeDesc(data, size, serialized);
    case FxrtConstType::kAiCoreArgsInfosDescPtr:
      return SerializeArgsInfosDesc(data, size, serialized);
    case FxrtConstType::kAclrtLaunchKernelAttrPtr:
      return SerializeLaunchKernelAttrs(data, size, serialized);
    case FxrtConstType::kAclrtLaunchKernelCfgPtr: {
      aclrtLaunchKernelCfg source{};
      if (!Read(data, size, &source) || source.attrs != nullptr || source.numAttrs != 5U) return false;
      return Write(source.numAttrs, serialized);
    }
    case FxrtConstType::kAiCoreMixCoreArgsPtr: {
      aicore_rt::MixCoreArgs source{};
      if (!Read(data, size, &source) ||
          (source.mix_type != aicore_rt::MixType::kMixAiCore &&
           source.mix_type != aicore_rt::MixType::kMixVectorCore)) {
        return false;
      }
      const MixCoreArgsPayload payload{static_cast<uint32_t>(source.mix_type), source.all_core_num,
                                       source.vec_core_num, source.ai_core_num};
      return Write(payload, serialized);
    }
    case FxrtConstType::kGertAllocatorsPtr:
    case FxrtConstType::kGertTensorDataVectorPtr:
    case FxrtConstType::kGertTilingFwkDataPtr: {
      uintptr_t value = 0U;
      return Read(data, size, &value) && value == 0U && Write(value, serialized);
    }
    default:
      return false;
  }
}

}  // namespace fx_converter
