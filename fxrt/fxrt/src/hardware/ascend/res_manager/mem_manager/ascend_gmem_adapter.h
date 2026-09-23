#ifndef FXRT_SRC_HARDWARE_ASCEND_ASCEND_GMEM_ADAPTER_H_
#define FXRT_SRC_HARDWARE_ASCEND_ASCEND_GMEM_ADAPTER_H_

#include <atomic>
#include <memory>

#include "acl/acl.h"
#include "hardware/hardware_abstract/dlopen_macro.h"

namespace fxrt {
namespace device {
namespace ascend {
#define CONCAT(l, r) l##r
// Function Object definition marco.
#define LIB_FUNC(funcName) CONCAT(funcName, FunObj)
// Function definition marco, and then can use `LIB_FUNC(funcName)`.
#define DEFINE_LIB_METHOD(funcName, ...) ORIGIN_METHOD(funcName, __VA_ARGS__)

// GMem mem free eager function name. Need to use origin name when export symbol from lib.
#define GMEM_FREE_EAGER gmemFreeEager
// Definition for GMem lib function : GMEM_FREE_EAGER.
DEFINE_LIB_METHOD(GMEM_FREE_EAGER, size_t, uint64_t, size_t, void*);

using DeviceMemPtr = void(*);
class AscendGmemAdapter {
 public:
  static AscendGmemAdapter& GetInstance() {
    static AscendGmemAdapter instance{};
    return instance;
  }

  AscendGmemAdapter() {
    LoadGMemLib();
  }
  ~AscendGmemAdapter() {
    UnloadGMemLib();
  }

 public:
  const size_t GetRoundUpAlignSize(size_t inputSize) const;
  const size_t GetRoundDownAlignSize(size_t inputSize) const;

  size_t AllocDeviceMem(size_t size, DeviceMemPtr* addr) const;
  size_t EagerFreeDeviceMem(const DeviceMemPtr addr, const size_t size) const;

  uint8_t* MmapMemory(size_t size, void* addr) const;
  bool MunmapMemory(void* addr, const size_t size) const;

  inline const bool is_eager_free_enabled() const {
    return isEagerFreeEnabled_;
  }

 private:
  void LoadGMemLib() noexcept;
  void UnloadGMemLib() noexcept;

  bool isEagerFreeEnabled_{false};
  void* gmemHandle_{nullptr};
  // Function for eager free.
  LIB_FUNC(GMEM_FREE_EAGER) freeEager_;
};
} // namespace ascend
} // namespace device
} // namespace fxrt

#endif
