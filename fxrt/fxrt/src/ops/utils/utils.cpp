#include "ops/utils/utils.h"
#include <string>
#include <unordered_set>

namespace fxrt {
namespace ops {

namespace {
constexpr size_t kDefaultAtbCacheCapacity = 64;
constexpr size_t kDefaultAclnnCacheCapacity = 10000;

size_t ResolveCacheCapacity(const char* envName, size_t defaultCapacity, const char* cacheNameForLog) {
  size_t resolved = defaultCapacity;
  auto env = GetEnv(envName);
  if (!env.empty() && IsPositiveInteger(env)) {
    try {
      size_t value = std::stoull(env);
      if (value != 0) {
        resolved = value;
      }
    } catch (...) {
      resolved = defaultCapacity;
    }
  }
  RT_VLOG(VL_OPS) << cacheNameForLog << " cache capacity : " << resolved;
  return resolved;
}
} // namespace

size_t GetAtbCacheCapacity() {
  static const size_t capacity = ResolveCacheCapacity("FXRT_ATB_CACHE_CAPACITY", kDefaultAtbCacheCapacity, "ATB");
  return capacity;
}

size_t GetAclnnCacheCapacity() {
  static const size_t capacity = ResolveCacheCapacity("FXRT_ACLNN_CACHE_CAPACITY", kDefaultAclnnCacheCapacity, "ACLNN");
  return capacity;
}

static const std::unordered_set<MemoryFormat> BaseFormatSet = {
    MemoryFormat::FORMAT_ND,
    MemoryFormat::FORMAT_NCHW,
    MemoryFormat::FORMAT_NHWC,
    MemoryFormat::FORMAT_NCDHW,
};

bool IsBaseFormat(MemoryFormat format) {
  return BaseFormatSet.count(format) != 0;
}

bool IsTensorBaseFormat(const ir::TensorPtr& tensor) {
  return IsBaseFormat(tensor->Format());
}

MemoryFormat GetBaseFormat(MemoryFormat format) {
  switch (format) {
    case MemoryFormat::FORMAT_ND:
      return MemoryFormat::FORMAT_ND;
    case MemoryFormat::FORMAT_NCHW:
      return MemoryFormat::FORMAT_NCHW;
    case MemoryFormat::FORMAT_NHWC:
      return MemoryFormat::FORMAT_NHWC;
    case MemoryFormat::FORMAT_NC1HWC0:
      return MemoryFormat::FORMAT_NCHW;
    case MemoryFormat::FORMAT_FRACTAL_Z:
      return MemoryFormat::FORMAT_NCHW;
    case MemoryFormat::FORMAT_FRACTAL_NZ:
      return MemoryFormat::FORMAT_ND;
    case MemoryFormat::FORMAT_NCDHW:
      return MemoryFormat::FORMAT_NCDHW;
    case MemoryFormat::FORMAT_NDHWC:
      return MemoryFormat::FORMAT_NCDHW;
    case MemoryFormat::FORMAT_NDC1HWC0:
      return MemoryFormat::FORMAT_NCDHW;
    case MemoryFormat::FORMAT_FRACTAL_Z_3D:
      return MemoryFormat::FORMAT_NCDHW;
    default:
      RT_GLOG(EXCEPTION) << "unknown format type: " << static_cast<int>(format);
      return MemoryFormat::FORMAT_ND;
  }
}

bool IsDefiniteTensorWhenMetaDataChanges(const ir::TensorPtr& tensor, const std::vector<int64_t>& shape) {
  const auto baseFormat = GetBaseFormat(tensor->Format());
  if (baseFormat == MemoryFormat::FORMAT_NCHW && shape.size() >= 5) {
    return true;
  }
  if (baseFormat == MemoryFormat::FORMAT_NCDHW && shape.size() != 5) {
    return true;
  }
  return false;
}

void CalBroadCastShape(
    const std::vector<int64_t>& xShape,
    const std::vector<int64_t>& yShape,
    std::vector<int64_t>* broadcastShape) {
  if (xShape == yShape) {
    *broadcastShape = xShape;
    return;
  }

  auto xLength = xShape.size();
  auto yLength = yShape.size();
  auto res = xLength > yLength;
  size_t maxLen = res ? xLength : yLength;
  size_t minLen = res ? yLength : xLength;
  const std::vector<int64_t>& maxShape = res ? xShape : yShape;
  const std::vector<int64_t>& minShape = res ? yShape : xShape;

  *broadcastShape = maxShape;
  auto lengthDiff = maxLen - minLen;
  for (size_t i = 0; i < minLen; ++i) {
    auto dsti = lengthDiff + i;
    if (maxShape[dsti] == 1) {
      (*broadcastShape)[dsti] = minShape[i];
    } else if (maxShape[dsti] != minShape[i] && minShape[i] != 1) {
      RT_GLOG(EXCEPTION) << "xShape[" << xLength + i << "] or yShape[" << yLength + i
                         << "] must be when they are not equal"
                         << ", but got xShape=" << ir::ShapeToString(xShape)
                         << ", yShape=" << ir::ShapeToString(yShape);
    }
  }
}

} // namespace ops
} // namespace fxrt
