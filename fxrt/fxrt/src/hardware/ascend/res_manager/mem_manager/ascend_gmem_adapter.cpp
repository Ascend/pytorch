#include "hardware/ascend/res_manager/mem_manager/ascend_gmem_adapter.h"
#include <pthread.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <tuple>
#include "common/common.h"
#include "hardware/ascend/res_manager/ascend_stream_manager.h"

namespace fxrt {
namespace device {
namespace ascend {
static constexpr const char kGMemLibName[] = "libgmem.so";
static constexpr const char kFxrtEnableGmem[] = "FXRT_ENABLE_GMEM";
constexpr uint64_t kAscendMmapAlignSize = 1 << 21;
constexpr int kMapPeerShared = 0x8000000;

const size_t AscendGmemAdapter::GetRoundUpAlignSize(size_t inputSize) const {
  return (inputSize + kAscendMmapAlignSize - 1) & ~(kAscendMmapAlignSize - 1);
}

const size_t AscendGmemAdapter::GetRoundDownAlignSize(size_t inputSize) const {
  return inputSize & ~(kAscendMmapAlignSize - 1);
}

size_t AscendGmemAdapter::AllocDeviceMem(size_t size, DeviceMemPtr* addr) const {
  size_t alignSize = GetRoundUpAlignSize(size);
  uint8_t* allocAddr = MmapMemory(alignSize, nullptr);
  if (allocAddr == nullptr) {
    RT_VLOG(VL_HARDWARE) << "Malloc memory failed.";
    return 0;
  }
  *addr = allocAddr;
  return alignSize;
}

size_t AscendGmemAdapter::EagerFreeDeviceMem(const DeviceMemPtr addr, const size_t size) const {
  CHECK_IF_NULL(addr);
  RT_VLOG(VL_HARDWARE) << "Enter ascend eager free device mem, addr : " << addr << ", size : " << size << ".";
  if (size == 0) {
    RT_VLOG(VL_HARDWARE) << "Eager free device mem, addr : " << addr << ", size is zero.";
    return 0;
  }
  size_t addrSizeT = reinterpret_cast<size_t>(addr);
  // Adjust addr -> round up addr, size -> round down size.
  size_t fromAddr = GetRoundUpAlignSize(addrSizeT);
  size_t endAddr = GetRoundDownAlignSize(addrSizeT + size);
  if (endAddr <= fromAddr) {
    RT_VLOG(VL_HARDWARE) << "End addr : " << endAddr << " is not bigger than fromAddr : " << fromAddr << ".";
    return 0;
  }
  size_t realSize = endAddr - fromAddr;
  int ret = freeEager_(fromAddr, SizeToUlong(realSize), nullptr);
  return ret != 0 ? 0 : realSize;
}

uint8_t* AscendGmemAdapter::MmapMemory(size_t size, void* addr) const {
  RT_VLOG(VL_HARDWARE) << "Enter mmap memory, size : " << size << ".";
  if (size == 0) {
    RT_GLOG(ERROR) << "Mmap memory, addr : " << addr << ", size is zero.";
    return nullptr;
  }

  int flags = MAP_PRIVATE | MAP_ANONYMOUS | kMapPeerShared;
  int prot = PROT_READ | PROT_WRITE;
  void* mappedAddr = mmap(addr, size, prot, flags, -1, 0);
  if (mappedAddr == MAP_FAILED) {
    RT_GLOG(ERROR) << "Mmap failed.";
  }
  return static_cast<uint8_t*>(mappedAddr);
}

bool AscendGmemAdapter::MunmapMemory(void* addr, const size_t size) const {
  RT_VLOG(VL_HARDWARE) << "Enter munmap memory, addr : " << addr << ", size : " << size << ".";
  auto ret = munmap(addr, size);
  return ret != -1;
}

void AscendGmemAdapter::LoadGMemLib() noexcept {
  RT_VLOG(VL_HARDWARE) << "FXRT_ENABLE_GMEM is set, try to open gmem.";
  gmemHandle_ = dlopen(kGMemLibName, RTLD_NOW);
  if (gmemHandle_ != nullptr) {
    RT_VLOG(VL_HARDWARE) << "Open GMem lib success, fxrt will use gmem to optimize memory usage.";
    LIB_FUNC(GMEM_FREE_EAGER) gmemFreeEager = DlsymFuncObj(gmemFreeEager, gmemHandle_);
    if (gmemFreeEager != nullptr) {
      isEagerFreeEnabled_ = true;
      freeEager_ = gmemFreeEager;
    } else {
      RT_VLOG(VL_HARDWARE) << "Load gmem free eager failed.";
      if (dlclose(gmemHandle_) != 0) {
        RT_GLOG(ERROR) << "Close GMem lib failed, detail : " << dlerror() << ".";
      }
    }
  } else {
    RT_VLOG(VL_HARDWARE) << "Open GMem lib failed.";
  }
}

void AscendGmemAdapter::UnloadGMemLib() noexcept {
  if (gmemHandle_ != nullptr) {
    RT_VLOG(VL_HARDWARE) << "Close GMem lib.";
    if (dlclose(gmemHandle_) != 0) {
      RT_GLOG(ERROR) << "Close GMem lib failed, detail : " << dlerror() << ".";
    }
    gmemHandle_ = nullptr;
  }
}
} // namespace ascend
} // namespace device
} // namespace fxrt
