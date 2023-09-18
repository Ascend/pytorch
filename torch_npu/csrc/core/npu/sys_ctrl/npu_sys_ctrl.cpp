// Copyright (c) 2022 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "third_party/acl/inc/acl/acl_op_compiler.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#ifdef SUCCESS
#undef SUCCESS
#endif
#ifdef FAILED
#undef FAILED
#endif

#if defined(_MSC_VER)
#include <direct.h>
#define GetCurrentDirPath _getcwd
#define Mkdir(path, mode) _mkdir(path)
#elif defined(__unix__)
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#define GetCurrentDirPath getcwd
#define Mkdir(path, mode) mkdir(path, mode)
#else
#endif

namespace {
const size_t kMaxPathLen = 4096U;
std::string GetCurDirPath() {
  char buff[kMaxPathLen] = {'\0'};
  GetCurrentDirPath(buff, kMaxPathLen);
  return std::string(buff);
}

void MakeCompileCacheDirAndSetOption() {
  char* compile_cache_mode_val = std::getenv("ACL_OP_COMPILER_CACHE_MODE");
  char* compile_cache_dir_val = std::getenv("ACL_OP_COMPILER_CACHE_DIR");

  std::string compile_cache_mode = (compile_cache_mode_val == nullptr) ? std::string("enable")
                                                                       : std::string(compile_cache_mode_val);
  std::string compile_cache_dir = (compile_cache_dir_val == nullptr) ? GetCurDirPath() + "/cache"
                                                                     : std::string(compile_cache_dir_val);
  // mode : 750
  auto ret = Mkdir(compile_cache_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP);
  if (ret == -1) {
    if (errno != EEXIST) {
      TORCH_WARN("make compile cache directory error: ", strerror(errno));
      return;
    }
  }
  if (compile_cache_mode != "enable" && compile_cache_mode != "disable" && compile_cache_mode != "force") {
    compile_cache_mode = std::string("enable");
  }
  c10_npu::option::register_options::OptionRegister::GetInstance()->Set("ACL_OP_COMPILER_CACHE_MODE", compile_cache_mode);
  c10_npu::option::register_options::OptionRegister::GetInstance()->Set("ACL_OP_COMPILER_CACHE_DIR", compile_cache_dir);
}

void GetAndSetDefaultJitCompileByAcl() {
  auto opt_size = at_npu::native::AclGetCompileoptSize(ACL_OP_JIT_COMPILE);
  if (!opt_size.has_value()) {
    ASCEND_LOGW("Get ACL JitCompile default value size failed, use PTA default value: True");
    return;
  }
  TORCH_CHECK(opt_size.value() != 0, "AclGetCompileoptSize opt_size.value() = 0 !");
  char value_name[opt_size.value()];
  auto ret = at_npu::native::AclGetCompileopt(ACL_OP_JIT_COMPILE, value_name, opt_size.value());
  // Get func success but get value failed, throw error
  TORCH_CHECK(ret == ACL_SUCCESS, "Get ACL JitCompile default value failed.");
  std::string value_str(value_name);
  c10_npu::option::SetOption("jitCompile", value_str);
  ASCEND_LOGI("Get ACL JitCompile default value %s and set", value_str.c_str());
}

void SetHF32DefaultValue() {
  // The default value of the flag used to control whether HF32 is allowed on conv is True.
  // The default value of the flag used to control whether HF32 is allowed on matmul is True,
  // and this flag defaults to False in all PyTorch versions.

  // When the flag of matmul is False, and the flag of conv is True,
  // the value of option "ACL_ALLOW_HF32" should be set to "10";
  std::string allow_hf32 = "10";
  auto ret = at_npu::native::AclSetCompileopt(aclCompileOpt::ACL_ALLOW_HF32, allow_hf32.c_str());
  if (ret == ACL_SUCCESS) {
    ASCEND_LOGI("Set ACL option ACL_ALLOW_HF32 default value to %s.", allow_hf32.c_str());
  } else if (ret == ACL_ERROR_INTERNAL_ERROR) {
    // Used to solve version compatibility issues, when ASCEND have not been updated.
    ASCEND_LOGW("Failed to set default value of ACL option ACL_ALLOW_HF32, which is unsupported by current version.");
  } else {
    TORCH_CHECK(0, "Failed to set compile option ACL_ALLOW_HF32, result = ", ret, ", set value ", allow_hf32);
  }
}
} // namespace

namespace c10_npu {

NpuSysCtrl::NpuSysCtrl() : init_flag_(false), device_id_(0), is_soc_match(true) {}

// Get NpuSysCtrl singleton instance
 NpuSysCtrl& NpuSysCtrl::GetInstance() {
  static NpuSysCtrl instance;
  return instance;
}

// Environment Initialize, return Status: SUCCESS, FAILED
 NpuSysCtrl::SysStatus NpuSysCtrl::Initialize(int device_id) {
    if (init_flag_) {
        if (!is_soc_match) {
          AT_ERROR("Unsupported soc version: ", soc_name_);
        }
        return INIT_SUCC;
    }
    NPU_CHECK_ERROR(aclInit(nullptr));

    if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()){
        NPU_CHECK_ERROR(aclmdlInitDump());
        NPU_LOGD("dump init success");
    }

    c10_npu::NPUCachingAllocator::init();
    NPU_LOGD("Npu caching allocator initialize successfully");

    auto ret = aclrtGetDevice(&device_id_);
    if (ret != ACL_ERROR_NONE) {
        device_id_ = (device_id == -1) ? 0 : device_id;
        NPU_CHECK_ERROR(aclrtSetDevice(device_id_));
    }else{
        NPU_LOGE("Npu device %d has been set before global init.", device_id_);
    }

    if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()) {
      const char *aclConfigPath = "acl.json";
      NPU_CHECK_ERROR(aclmdlSetDump(aclConfigPath));
      NPU_LOGD("set dump config success");
    }

  auto soc_name = c10_npu::acl::AclGetSocName();
  // set global soc name
  if (!c10_npu::SetSocVersion(soc_name)) {
    is_soc_match = false;
    soc_name_ = soc_name;
    AT_ERROR("Unsupported soc version: ", soc_name_);
  }

  if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1) {
    if (c10_npu::IsSupportInfNan()) {
      c10_npu::acl::AclrtSetDeviceSatMode(aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_INFNAN);
    } else {
      c10_npu::acl::AclrtSetDeviceSatMode(aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_SATURATION);
    }
  }

  // set ACL_PRECISION_MODE by SocVersion("allow_fp32_to_fp16" or "must_keep_origin_dtype").
  auto precision_mode = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 ?
      "must_keep_origin_dtype" : "allow_fp32_to_fp16";
  at_npu::native::AclSetCompileopt(aclCompileOpt::ACL_PRECISION_MODE, precision_mode);

  // set default compile cache mode and dir for users to improve op compile time
  MakeCompileCacheDirAndSetOption();
  // set default jit_Compile value from Get acl defalut value
  GetAndSetDefaultJitCompileByAcl();

  SetHF32DefaultValue();

  NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, 0));
    
  init_flag_ = true;
  NPU_LOGD("Npu sys ctrl initialize successfully.");
  return INIT_SUCC;
}

 NpuSysCtrl::SysStatus NpuSysCtrl::ExchangeDevice(int pre_device, int device) {
    NPU_CHECK_ERROR(aclrtResetDevice(pre_device));
    NPU_CHECK_ERROR(aclrtSetDevice(device));
    device_id_= device;
    return INIT_SUCC;
}

 NpuSysCtrl::SysStatus NpuSysCtrl::BackwardsInit() {
    NPU_CHECK_ERROR(aclrtSetDevice(device_id_));
    return INIT_SUCC;
}

NpuSysCtrl::SysStatus NpuSysCtrl::OverflowSwitchEnable() {
    if (!c10_npu::option::OptionsManager::CheckInfNanModeEnable() &&
        (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1)) {
      c10_npu::acl::AclrtSetStreamOverflowSwitch(c10_npu::getCurrentNPUStream(), 1);
     ASCEND_LOGI("Npu overflow check switch set successfully.");
    }
    return INIT_SUCC;
}

// Environment Finalize, return SysStatus
 NpuSysCtrl::SysStatus NpuSysCtrl::Finalize() {
    if (!init_flag_) {
        return FINALIZE_SUCC;
    }

    this->RegisterReleaseFn([=]() ->void {
          c10_npu::NPUEventManager::GetInstance().ClearEvent();
          auto stream = c10_npu::getCurrentNPUStream();
          NPU_CHECK_WARN(c10_npu::acl::AclrtDestroyStreamForce(stream));
          NPU_CHECK_WARN(aclrtResetDevice(device_id_));
          NPU_CHECK_WARN(aclFinalize());
        }, ReleasePriority::PriorityLast);

    init_flag_ = false;

    if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()) {
        NPU_CHECK_WARN(aclmdlFinalizeDump());
    }

    // call release fn by priotity
    for (const auto& iter : release_fn_) {
        const auto& fn_vec = iter.second;
        for (const auto& fn : fn_vec) {
            fn();
        }
    }
    release_fn_.clear();

    NPU_LOGD("Npu sys ctrl finalize successfully.");
    return FINALIZE_SUCC;
}

 bool NpuSysCtrl::GetInitFlag() {
    return init_flag_;
}

void NpuSysCtrl::RegisterReleaseFn(ReleaseFn release_fn,
                                   ReleasePriority priority) {
    const auto& iter = this->release_fn_.find(priority);
    if (iter != release_fn_.end()) {
        release_fn_[priority].emplace_back(release_fn);
    } else {
        release_fn_[priority] = (std::vector<ReleaseFn>({release_fn}));
    }
}

} // namespace c10_npu
