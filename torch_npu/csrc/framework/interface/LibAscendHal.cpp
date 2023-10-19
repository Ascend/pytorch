#include <c10/util/Exception.h>

#include "torch_npu/csrc/framework/interface/LibAscendHal.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

namespace at_npu {
namespace native {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libascend_hal, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)              \
  GET_FUNCTION(libascend_hal, funcName)

REGISTER_LIBRARY(libascend_hal)
LOAD_FUNCTION(halGetDeviceInfo)
LOAD_FUNCTION(halGetAPIVersion)
constexpr uint32_t KHZTOMHZ = 1000U;
constexpr uint32_t RES_OK = 0;
constexpr uint32_t ERR_FREQ = 1;
constexpr uint32_t ERR_VER = 0;
constexpr uint32_t FREQ_CONFIG = 24;

int64_t getFreq() {
  using getReqFun = int32_t (*)(uint32_t, int32_t, int32_t, int64_t*);
  static getReqFun getFreqInfo = nullptr;
  if (getFreqInfo == nullptr) {
      getFreqInfo = (getReqFun)GET_FUNC(halGetDeviceInfo);
  }
  int64_t freq = ERR_FREQ;
  if (getFreqInfo == nullptr) {
    TORCH_WARN("Failed to find function halGetDeviceInfo.");
    return ERR_FREQ;
  }
  int32_t ret = getFreqInfo(0, 0, FREQ_CONFIG, &freq);
  if (ret != RES_OK) {
    TORCH_WARN("Failed to find freq.");
    return ERR_FREQ;
  }
  return freq / KHZTOMHZ;
}
int64_t getVer() {
  using getReqFun = int32_t (*)(int32_t*);
  static getReqFun getVerInfo = nullptr;
  if (getVerInfo == nullptr) {
      getVerInfo = (getReqFun)GET_FUNC(halGetAPIVersion);
  }
  int32_t ver = ERR_VER;
  if (getVerInfo == nullptr) {
    TORCH_WARN("Failed to find function halGetAPIVersion.");
    return ERR_VER;
  }
  int32_t ret = getVerInfo(&ver);
  if (ret != RES_OK) {
    TORCH_WARN("Failed to find version.");
    return ERR_VER;
  }
  return ver;
}
bool isSyscntEnable() {
  constexpr int32_t supportVersion = 0x071905;
  return getVer() >= supportVersion;
}

} // namespace native
} // namespace at_npu

