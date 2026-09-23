#include "hardware/cpu/res_manager/cpu_res_manager.h"
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cstring>

namespace fxrt {
namespace device {
namespace cpu {
void CPUResManager::Initialize() {
  RT_VLOG(VL_HARDWARE) << "Unimplemented interface.";
}

void CPUResManager::Destroy() {
  RT_VLOG(VL_HARDWARE) << "Unimplemented interface.";
}

void* CPUResManager::AllocateMemory(size_t size, uint32_t streamId) const {
  void* ptr = std::malloc(size);
  if (ptr == nullptr) {
    RT_GLOG(ERROR) << "Memory allocate failed";
    return nullptr;
  }
  return ptr;
}

void CPUResManager::FreeMemory(void* ptr) const {
  CHECK_IF_NULL(ptr);
  std::free(ptr);
}

void CPUResManager::FreePartMemorys(
    const std::vector<void*>& freeAddrs,
    const std::vector<void*>& keepAddrs,
    const std::vector<size_t>& keepAddrSizes) const {
  RT_VLOG(VL_HARDWARE) << "Unimplemented interface.";
  return;
}

bool CPUResManager::AsyncCopy(void* dst, const void* src, uint64_t size, CopyType kind, void* stream) const {
  RT_GLOG(ERROR) << "Not support async copy for CPU platform";
  return false;
}

bool CPUResManager::SyncCopy(void* dst, const void* src, uint64_t size, CopyType kind) const {
  CHECK_IF_NULL(dst);
  CHECK_IF_NULL(src);
  if (size == 0) {
    return true;
  }
  // The destination size always equals the copy size here, so the only check
  // left to do before memcpy is that the two buffers do not overlap.
  auto* dstBytes = static_cast<uint8_t*>(dst);
  const auto* srcBytes = static_cast<const uint8_t*>(src);
  if (dstBytes < srcBytes + size && srcBytes < dstBytes + size) {
    RT_GLOG(ERROR) << "Sync copy source and destination overlap, size:" << size;
    return false;
  }
  std::memcpy(dst, src, size);
  return true;
}

namespace {

// clang-format off
#define FOR_EACH_TYPE_BASE(M)                    \
  M(kNumberTypeBool, bool)                       \
  M(kNumberTypeUInt8, uint8_t)                   \
  M(kNumberTypeInt4, int8_t)                     \
  M(kNumberTypeInt8, int8_t)                     \
  M(kNumberTypeInt16, int16_t)                   \
  M(kNumberTypeInt32, int32_t)                   \
  M(kNumberTypeInt64, int64_t)                   \
  M(kNumberTypeUInt16, uint16_t)                 \
  M(kNumberTypeUInt32, uint32_t)                 \
  M(kNumberTypeUInt64, uint64_t)                 \
  M(kNumberTypeFloat16, float16)                 \
  M(kNumberTypeFloat32, float)                   \
  M(kNumberTypeFloat64, double)                  \
  M(kNumberTypeFloat8E4M3FN, float8_e4m3fn)      \
  M(kNumberTypeFloat8E5M2, float8_e5m2)          \
  M(kNumberTypeHiFloat8, hifloat8)               \
  M(kNumberTypeComplex64, ComplexStorage<float>) \
  M(kNumberTypeComplex128, ComplexStorage<double>)


#define FOR_EACH_TYPE_EXTRA(M) M(kNumberTypeBFloat16, bfloat16)

#define FOR_EACH_TYPE(M) \
  FOR_EACH_TYPE_BASE(M)  \
  FOR_EACH_TYPE_EXTRA(M)

#define REGISTER_SIZE(addressTypeId, addressType) { addressTypeId, sizeof(addressType) },


#undef FOR_EACH_TYPE
#undef FOR_EACH_TYPE_BASE
#undef FOR_EACH_TYPE_EXTRA
#undef REGISTER_SIZE
// clang-format on
} // namespace

} // namespace cpu
} // namespace device
} // namespace fxrt
