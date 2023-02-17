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
#include <Python.h>
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
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
#include <third_party/acl/inc/ge/ge_api.h>

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
  auto compile_cache_dir = GetCurDirPath() + "/cache";
  // mode : 750
  auto ret = Mkdir(compile_cache_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP);
  if (ret == -1) {
    if (errno != EEXIST) {
      TORCH_WARN("make compile cache directory error: ", strerror(errno));
      return;
    }
  }
  c10_npu::option::register_options::OptionRegister::GetInstance()->Set("ACL_OP_COMPILER_CACHE_MODE", "enable");
  c10_npu::option::register_options::OptionRegister::GetInstance()->Set("ACL_OP_COMPILER_CACHE_DIR", compile_cache_dir);
}
} // namespace

namespace c10_npu {

NpuSysCtrl::NpuSysCtrl() : init_flag_(false), device_id_(0) {}

// Get NpuSysCtrl singleton instance
 NpuSysCtrl& NpuSysCtrl::GetInstance() {
  static NpuSysCtrl instance;
  return instance;
}

// GE Environment Initialize, return Status: SUCCESS, FAILED
 NpuSysCtrl::SysStatus NpuSysCtrl::Initialize(int device_id) {
    if (init_flag_) {
        return INIT_SUCC;
    }
    C10_NPU_CHECK(aclInit(nullptr));

    if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()){
        C10_NPU_CHECK(aclmdlInitDump());
        NPU_LOGD("dump init success");
    }

    auto ret = aclrtGetDevice(&device_id_);
    if (ret != ACL_ERROR_NONE) {
        device_id_ = (device_id == -1) ? 0 : device_id;
        C10_NPU_CHECK(aclrtSetDevice(device_id_));
    }else{
        NPU_LOGE("Npu device %d has been set before global init.", device_id_);
    }

    init_flag_ = true;
    NPU_LOGD("Npu sys ctrl initialize successfully.");

    if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()) {
      const char *aclConfigPath = "acl.json";
      C10_NPU_CHECK(aclmdlSetDump(aclConfigPath));
      NPU_LOGD("set dump config success");
    }

  auto npu_device_id = std::to_string(device_id_);
  std::map<ge::AscendString, ge::AscendString> config = {
      {ge::AscendString(ge::OPTION_EXEC_DEVICE_ID),
       ge::AscendString(npu_device_id.data())},
      {ge::AscendString(ge::OPTION_GRAPH_RUN_MODE), "0"},
      {ge::AscendString(ge::VARIABLE_MEMORY_MAX_SIZE), "1048576"},
      {ge::AscendString(ge::OP_SELECT_IMPL_MODE.data()), "high_precision"}
  };

  config["ge.session_device_id"] = ge::AscendString(npu_device_id.data());
  config["ge.exec.reuseZeroCopyMemory"] = ge::AscendString("1");

  static std::map<const std::string, const std::string>
      STRING_TO_COMPILE_OPT_MAP = {
      {"ACL_OP_DEBUG_LEVEL", ge::OP_DEBUG_LEVEL},
      {"ACL_DEBUG_DIR", ge::DEBUG_DIR},
      {"ACL_OP_COMPILER_CACHE_MODE", ge::OP_COMPILER_CACHE_MODE},
      {"ACL_OP_COMPILER_CACHE_DIR", ge::OP_COMPILER_CACHE_DIR},
      {"ACL_OP_SELECT_IMPL_MODE", ge::OP_SELECT_IMPL_MODE},
      {"ACL_OPTYPELIST_FOR_IMPLMODE", ge::OPTYPELIST_FOR_IMPLMODE}
  };

  for (const auto& iter : STRING_TO_COMPILE_OPT_MAP) {
    auto val = c10_npu::option::GetOption(iter.first);
    if (val.has_value() && (!val.value().empty())) {
      config.emplace(iter.second.data(), val.value().data());
    }
  }

  auto soc_name = c10_npu::acl::AclGetSocName();
  if (soc_name != nullptr) {
    config.emplace(ge::AscendString(ge::SOC_VERSION.data()), soc_name);
  }

  // set global soc name
  c10_npu::SetSocVersion(soc_name);

  if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1){
    auto init_mode = aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_SATURATION;
    c10_npu::acl::AclrtSetDeviceSatMode(init_mode);
  }


  if (c10_npu::acl::IsExistQueryEventRecordedStatus()) {
    static const std::string HCOM_OPTIONS = "ge.exec.isUseHcom";
    config.emplace(HCOM_OPTIONS.data(), "1");
  }

  C10_NPU_CHECK(ge::GEInitialize(config));

  // set ACL_PRECISION_MODE by SocVersion("allow_fp32_to_fp16" or "must_keep_origin_dtype").
  auto precision_mode = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 ?
      "allow_fp32_to_fp16" : "allow_fp32_to_fp16";
  at_npu::native::AclSetCompileopt(aclCompileOpt::ACL_PRECISION_MODE, precision_mode);

  // set default compile cache mode and dir for users to improve op compile time
  MakeCompileCacheDirAndSetOption();

  return INIT_SUCC;
}

 NpuSysCtrl::SysStatus NpuSysCtrl::ExchangeDevice(int pre_device, int device) {
    C10_NPU_CHECK(aclrtResetDevice(pre_device));
    C10_NPU_CHECK(aclrtSetDevice(device));
    device_id_= device;
    return INIT_SUCC;
}

 NpuSysCtrl::SysStatus NpuSysCtrl::BackwardsInit() {
    C10_NPU_CHECK(aclrtSetDevice(device_id_));
    return INIT_SUCC;
}

// GE Environment Finalize, return SysStatus
 NpuSysCtrl::SysStatus NpuSysCtrl::Finalize() {
    if (!init_flag_) {
        return FINALIZE_SUCC;
    }

    this->RegisterReleaseFn([=]() ->void {
          c10_npu::NPUEventManager::GetInstance().ClearEvent();
          auto stream = c10_npu::getCurrentNPUStream();
          (void)aclrtDestroyStream(stream);
          C10_NPU_CHECK(ge::GEFinalize());
          C10_NPU_CHECK(aclrtResetDevice(device_id_));
          C10_NPU_CHECK(aclFinalize());
        }, ReleasePriority::PriorityLast);

    init_flag_ = false;

    if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()) {
        C10_NPU_CHECK(aclmdlFinalizeDump());
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
