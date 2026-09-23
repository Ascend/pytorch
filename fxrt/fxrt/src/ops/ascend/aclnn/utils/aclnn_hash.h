#ifndef __OPS_ASCEND_ACLNN_UTILS_ACLNN_HASH_H__
#define __OPS_ASCEND_ACLNN_UTILS_ACLNN_HASH_H__

#include <string>
#include <vector>
#include <optional>

#include "ir/value/value.h"
#include "ops/ascend/aclnn/utils/hash_buf.h"

namespace fxrt {
namespace ops {
// Gather tensor
void GatherHash(const ir::TensorPtr& tensor);

inline void GatherHash(const std::vector<ir::TensorPtr>& tensorList) {
  for (auto& tensor : tensorList) {
    GatherHash(tensor);
  }
}

// Gather scalar, int64_t/bool/float/double, etc.
template <typename T>
void GatherHash(const T& value) {
  MemcpyToBuf(&value, sizeof(T));
}

// Gather vector scalar.
template <typename T>
void GatherHash(const std::vector<T>& values) {
  MemcpyToBuf(values.data(), values.size() * sizeof(T));
}

template <typename T>
void GatherHash(std::vector<T>& values) {
  MemcpyToBuf(values.data(), values.size() * sizeof(T));
}

// Gather pair.
template <typename K, typename V>
void GatherHash(const std::pair<K, V>& value) {
  GatherHash(value.first);
  GatherHash(value.second);
}

inline void GatherHash(const std::string& str) {
  MemcpyToBuf(str.c_str(), str.size());
}

// Gather value
inline void GatherHash(const ir::ValuePtr& value) {
  if (value == nullptr || value->IsNone()) {
    return;
  }
  if (value->IsTensor()) {
    GatherHash(value->ToTensor());
  } else if (value->IsInt()) {
    GatherHash(value->ToInt());
  } else if (value->IsSymbol()) {
    GatherHash(value->ToInt());
  } else if (value->IsDouble()) {
    GatherHash(value->ToDouble());
  } else if (value->IsBool()) {
    GatherHash(value->ToBool());
  } else if (value->IsString()) {
    GatherHash(value->ToString());
  } else {
    RT_GLOG(EXCEPTION) << "Invalid value type: " << value << " for hash from tuple";
  }
}

inline void GatherHash(const ir::Value* value) {
  if (value == nullptr)
    return;
  if (value->IsTensor()) {
    GatherHash(value->ToTensor());
  } else if (value->IsInt()) {
    GatherHash(value->ToInt());
  } else if (value->IsSymbol()) {
    GatherHash(value->ToInt());
  } else if (value->IsDouble()) {
    GatherHash(value->ToDouble());
  } else if (value->IsBool()) {
    GatherHash(value->ToBool());
  } else if (value->IsString()) {
    GatherHash(value->ToString());
  }
}

// GatherHash for vector of raw pointers
inline void GatherHash(const std::vector<const ir::Value*>& values) {
  for (const auto* value : values) {
    GatherHash(value);
  }
}

// Gather tuple
inline void GatherHash(const ir::TuplePtr& tuple) {
  if (tuple == nullptr || tuple->Size() == 0) {
    return;
  }
  for (const auto& value : *tuple) {
    GatherHash(value);
  }
}

inline void GatherHash() {}

template <typename T>
void GatherHash(const std::optional<T>& value) {
  if (value.has_value()) {
    GatherHash(value.value());
  }
}

template <typename T, typename... Args>
void GatherHash(const T& arg, const Args&... args) {
  GatherHash(arg);
  GatherHash(args...);
}

// Main entry for calculate hash
template <typename... Args>
uint64_t CalcAclnnHash(const std::string& opName, const Args&... args) {
  gHashOffset = 0;
  GatherHash(opName, args...);
  return CalcHashId();
}

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_UTILS_ACLNN_HASH_H__
