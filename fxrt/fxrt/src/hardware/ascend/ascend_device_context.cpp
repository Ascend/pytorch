#include "hardware/ascend/ascend_device_context.h"
#include <unistd.h>
#include <tuple>
#include <algorithm>
#include <sstream>
#include <map>
#include <thread>
#include <set>
#include "config/device/ascend/common.h"
#include "config/device/ascend/op_precision_conf.h"
#include "hardware/hardware_abstract/device_context_manager.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_compiler_symbol.h"
#include "common/common.h"

namespace fxrt {
namespace device {
namespace ascend {
namespace {
using SocVersion = config::ascend::SocVersion;
constexpr auto kSaturationMode = "Saturation";
constexpr auto kINFNANMode = "INFNAN";
const char kAscendDevice[] = "Ascend";

const std::map<std::string, SocVersion> kAscendSocVersionMap = {
    {"Ascend910PremiumA", SocVersion::k910PremiumA},
    {"Ascend910ProA", SocVersion::k910ProA},
    {"Ascend910A", SocVersion::k910A},
    {"Ascend910ProB", SocVersion::k910ProB},
    {"Ascend910B", SocVersion::k910B},
    {"Ascend310P1", SocVersion::k310P1},
    {"Ascend310P2", SocVersion::k310P2},
    {"Ascend310P3", SocVersion::k310P3},
    {"Ascend310P4", SocVersion::k310P4},
    {"Ascend310P5", SocVersion::k310P5},
    {"Ascend310P7", SocVersion::k310P7},
    {"Ascend910B1", SocVersion::k910B1},
    {"Ascend910B2", SocVersion::k910B2},
    {"Ascend910B2C", SocVersion::k910B2C},
    {"Ascend910B3", SocVersion::k910B3},
    {"Ascend910B4", SocVersion::k910B4},
    {"Ascend910B4-1", SocVersion::k910B4_1},
    {"Ascend310B1", SocVersion::k310B1},
    {"Ascend310B2", SocVersion::k310B2},
    {"Ascend310B3", SocVersion::k310B3},
    {"Ascend310B4", SocVersion::k310B4},
    {"Ascend910_9391", SocVersion::k910_9391},
    {"Ascend910_9392", SocVersion::k910_9392},
    {"Ascend910_9381", SocVersion::k910_9381},
    {"Ascend910_9382", SocVersion::k910_9382},
    {"Ascend910_9372", SocVersion::k910_9372},
    {"Ascend910_9362", SocVersion::k910_9362},
};

const SocVersion& GetSocVersion() {
  auto socVersion = GetAscendSocVersion();
  if (socVersion == nullptr) {
    RT_GLOG(EXCEPTION) << "Get soc version failed.";
  }
  auto it = kAscendSocVersionMap.find(socVersion);
  if (it == kAscendSocVersionMap.end()) {
    RT_GLOG(EXCEPTION) << "Unsupported soc version: " << std::string(socVersion);
  }
  return it->second;
}
} // namespace

void AscendDeviceContext::InitializeForAclop() const {
  if (initializedAclop_) {
    return;
  }

  RT_VLOG(VL_HARDWARE) << "Start initializing for acl.";
  LoadAscendApiSymbols();

  RT_VLOG(VL_HARDWARE) << "End initializing for acl.";
}

void AscendDeviceContext::Initialize() {
  std::lock_guard<std::mutex> lock(initMutex_);
  if (initialized_) {
    return;
  }

  RT_VLOG(VL_HARDWARE) << "Start initializing device context.";
  LoadAscendApiSymbols();

  CHECK_IF_NULL(deviceResManager_);
  deviceResManager_->Initialize();

  initialized_ = true;
  pid_ = getpid(); // set the pid when first initialize

  config::ascend::OpPrecisionConf::Instance().SetSocVersion(GetSocVersion());
  RT_VLOG(VL_HARDWARE) << "End initializing device context.";
}

void AscendDeviceContext::Destroy() {
  if (pid_ != getpid()) {
    // Check whether the device context needs to be released.
    // The device context is copied by the dataset independent process, but does not need to be released
    // in the dataset independent process.
    // The device context is copied from main process by fork
    RT_VLOG(VL_HARDWARE)
        << "The device context is not initialized by current process, it doesn't need to be destroyed.";
    return;
  }

  if (deviceResManager_ == nullptr) {
    return;
  }
  // Device resource manager must be destroyed before 'FinalizeGe' unless some runtime APIs will throw exception.
  // for ge, has destropy in graph_executor->finalize
  deviceResManager_->Destroy();

  initialized_ = false;
}

FXRT_REGISTER_DEVICE(kAscendDevice, AscendDeviceContext);
} // namespace ascend
} // namespace device
} // namespace fxrt
