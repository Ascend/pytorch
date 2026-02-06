#include "torch_npu/csrc/framework/LazyInitAclops.h"

#include <ATen/record_function.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"

#ifndef BUILD_LIBTORCH
#include <Python.h>
#endif
#include <atomic>
#if defined(_MSC_VER)
#include <direct.h>
#define GetCurrentDirPath _getcwd
#define Mkdir(path, mode) _mkdir(path)
#elif defined(__unix__)
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <mutex>
#include <vector>
#include <string>
#define GetCurrentDirPath getcwd
#define Mkdir(path, mode) mkdir(path, mode)
#else
#endif

namespace at_npu {
namespace aclops {

std::mutex LazyAclopSet::lazy_set_mutex_;
bool LazyAclopSet::acl_op_call_ = false;
std::vector<std::pair<aclCompileOpt, std::string>> LazyAclopSet::lazy_acl_opt_;
std::atomic<bool> encounteredAclops(false);

aclError LazyAclopSet::LazyAclSetCompileopt(aclCompileOpt opt, const char *value)
{
    static auto acl_op_init_mode = c10_npu::option::OptionsManager::GetAclOpInitMode();
    ASCEND_LOGI("[LazyAclops] LazyAclSetCompileopt called, acl_op_init_mode=%d", acl_op_init_mode)
    if (acl_op_init_mode == 0) {
        return at_npu::native::AclSetCompileopt(opt, value);
    } else {
        if (value == nullptr) {
            return ACL_ERROR_INVALID_PARAM;
        }
        std::lock_guard<std::mutex> lock(lazy_set_mutex_);
        if (acl_op_call_) {
            return at_npu::native::AclSetCompileopt(opt, value);
        }
        lazy_acl_opt_.emplace_back(std::make_pair(opt, std::string(value)));
        ASCEND_LOGI("Cache ACL compile %d with value %s", static_cast<int>(opt), value)
        return ACL_ERROR_NONE;
    }
}

void LazyAclopSet::SetCompileopt()
{
    static auto acl_op_init_mode = c10_npu::option::OptionsManager::GetAclOpInitMode();
    ASCEND_LOGI("[LazyAclops] SetCompileopt called, acl_op_init_mode=%d", acl_op_init_mode)
    if (acl_op_init_mode == 0) {
        return;
    } else {
        std::lock_guard<std::mutex> lock(lazy_set_mutex_);
        if (!acl_op_call_) {
            for (const auto &iter: lazy_acl_opt_) {
                aclError ret = at_npu::native::AclSetCompileopt(iter.first, iter.second.c_str());
                TORCH_CHECK(ret == ACL_SUCCESS,
                    "Failed to set compile opt ", static_cast<int>(iter.first),
                    " with value '", iter.second, "'");
            }
            ASCEND_LOGI("Apply %zu cached ACL compile options", lazy_acl_opt_.size())
            lazy_acl_opt_.clear();
            acl_op_call_ = true;
        }
    }
}

void SetHF32DefaultValue()
{
    // The default value of the flag used to control whether HF32 is allowed on
    // conv is True. The default value of the flag used to control whether HF32
    // is allowed on matmul is True, but this flag defaults to False in
    // PyTorch 1.12 and later.

    // When the flag of matmul is False, and the flag of conv is True,
    // the value of option "ACL_ALLOW_HF32" should be set to "10";
    static const std::string mm_hf32_option_name = "ALLOW_MATMUL_HF32";
    auto mm_hf32_val = c10_npu::option::GetOption(mm_hf32_option_name);
    // default value is False;
    std::string mm_hf32 = "0";
    if (mm_hf32_val.has_value() && (mm_hf32_val.value() == "enable")) {
        mm_hf32 = "1";
    }
    static const std::string conv_hf32_option_name = "ALLOW_CONV_HF32";
    auto conv_hf32_val = c10_npu::option::GetOption(conv_hf32_option_name);
    // default value is True;
    std::string conv_hf32 = "1";
    if (conv_hf32_val.has_value() && (conv_hf32_val.value() == "disable")) {
        conv_hf32 = "0";
    }

    std::string allow_hf32 = conv_hf32 + mm_hf32;
    auto ret = at_npu::native::AclSetCompileopt(aclCompileOpt::ACL_ALLOW_HF32, allow_hf32.c_str());
    if (ret == ACL_SUCCESS) {
        ASCEND_LOGI("Set ACL option ACL_ALLOW_HF32 default value to %s.", allow_hf32.c_str());
    } else if (ret == ACL_ERROR_INTERNAL_ERROR) {
        // Used to solve version compatibility issues, when ASCEND have not been
        // updated.
        ASCEND_LOGW(
            "Failed to set default value of ACL option ACL_ALLOW_HF32, which is "
            "unsupported by current version.");
    } else {
        TORCH_CHECK(0, "Failed to set compile option ACL_ALLOW_HF32, result = ", ret, ", set value ", allow_hf32,
                    PTA_ERROR(ErrCode::ACL));
    }
}

// set default compile cache mode and dir to improve op compile time
void MakeCompileCacheDirAndSetOption()
{
    char *compile_cache_mode_val = std::getenv("ACL_OP_COMPILER_CACHE_MODE");
    std::string compile_cache_mode =
        (compile_cache_mode_val == nullptr) ? std::string("enable") : std::string(compile_cache_mode_val);
    if (compile_cache_mode != "enable" && compile_cache_mode != "disable" && compile_cache_mode != "force") {
        compile_cache_mode = std::string("enable");
    }
    auto compile_mode = c10_npu::option::GetOption("ACL_OP_COMPILER_CACHE_MODE");
    if (!compile_mode.has_value() || compile_mode.value() == "") {
        c10_npu::option::register_options::OptionRegister::GetInstance()->Set(
            "ACL_OP_COMPILER_CACHE_MODE", compile_cache_mode);
    }

    char *compile_cache_dir_val = std::getenv("ACL_OP_COMPILER_CACHE_DIR");
    if (compile_cache_dir_val != nullptr) {
        std::string compile_cache_dir = std::string(compile_cache_dir_val);
        // mode : 750
        auto ret = Mkdir(compile_cache_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP);
        if (ret == -1) {
            if (errno != EEXIST) {
                TORCH_NPU_WARN("make compile cache directory error: ", strerror(errno));
                return;
            }
        }
        auto compile_dir = c10_npu::option::GetOption("ACL_OP_COMPILER_CACHE_DIR");
        if (!compile_dir.has_value() || compile_dir.value() == "") {
            c10_npu::option::register_options::OptionRegister::GetInstance()->Set(
                "ACL_OP_COMPILER_CACHE_DIR", compile_cache_dir);
        }
    }
}

bool IsJitCompileModeSetted()
{
    auto jit_compile = c10_npu::option::GetOption("jitCompile");
    if (jit_compile.has_value() && jit_compile.value() != "") {
        return true;
    }
    return false;
}

std::string GetJitCompileMode()
{
    auto opt_size = at_npu::native::AclGetCompileoptSize(ACL_OP_JIT_COMPILE);
    if (!opt_size.has_value()) {
        ASCEND_LOGW(
            "Get ACL JitCompile default value size failed, use PTA "
            "default value: "
            "True");
        return "";
    }
    TORCH_CHECK(opt_size.value() != 0, "AclGetCompileoptSize opt_size.value() = 0 !", PTA_ERROR(ErrCode::PARAM));

    char value_name[opt_size.value()];
    auto ret = at_npu::native::AclGetCompileopt(ACL_OP_JIT_COMPILE, value_name, opt_size.value());
    // Get func success but get value failed, throw error
    TORCH_CHECK(ret == ACL_SUCCESS, "Get ACL JitCompile default value failed.", PTA_ERROR(ErrCode::ACL));

    return std::string(value_name);
}

void InitializeJitCompilationMode()
{
    if (IsJitCompileModeSetted()) {
        return;
    }
    std::string value_str = GetJitCompileMode();
    if (value_str != "") {
        c10_npu::option::SetOption("jitCompileInit", value_str);
        ASCEND_LOGI("Set jitCompileInit option to %s", value_str.c_str());
    } else {
        c10_npu::option::SetOption("jitCompileInit", "disable");
        ASCEND_LOGI("Set jitCompileInit option to default value: disable");
    }
}

// set default jit_Compile value from Get acl defalut value
void GetAndSetDefaultJitCompileByAcl()
{
    if (IsJitCompileModeSetted()) {
        return;
    }

    std::string value_str = GetJitCompileMode();
    if (value_str != "") {
        c10_npu::option::SetOption("jitCompile", value_str);
    }
}

void SetPrecisionMode()
{
    // set ACL_PRECISION_MODE by SocVersion("allow_fp32_to_fp16" or
    // "must_keep_origin_dtype").
    auto precision_mode = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 ? "must_keep_origin_dtype"
                                                                                        : "allow_fp32_to_fp16";
    static const std::string precision_mode_ = "ACL_PRECISION_MODE";
    auto precision_mode_val = c10_npu::option::GetOption(precision_mode_);
    if (precision_mode_val.has_value()) {
        precision_mode = precision_mode_val->c_str();
    }
    NPU_CHECK_ERROR(at_npu::native::AclSetCompileopt(aclCompileOpt::ACL_PRECISION_MODE, precision_mode));
}

void LazyInitAclopsCore()
{
#ifndef BUILD_LIBTORCH
    PyThreadState *gilState = nullptr;
    if (PyGILState_Check()) {
        gilState = PyEval_SaveThread();
    }
#endif
    SetPrecisionMode();
    SetHF32DefaultValue();
    MakeCompileCacheDirAndSetOption();
    GetAndSetDefaultJitCompileByAcl();
#ifndef BUILD_LIBTORCH
    if (gilState) {
        PyEval_RestoreThread(gilState);
    }
#endif
}

void LazyInitAclops()
{
    static auto acl_op_init_mode = c10_npu::option::OptionsManager::GetAclOpInitMode();
    if (acl_op_init_mode == 0) {
        return;
    }
    TORCH_CHECK(acl_op_init_mode != 2,
                "Acl op is disabled! Please check the environment variable ACL_OP_INIT_MODE.",
                PTA_ERROR(ErrCode::NOT_SUPPORT));

    if (!encounteredAclops.exchange(true) && c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        RECORD_FUNCTION("LazyInitAclops", std::vector<c10::IValue>({}));
        LazyInitAclopsCore();
        ASCEND_LOGI("Lazy init for aclops finished.")
    }
}

void InitAclops()
{
    RECORD_FUNCTION("InitAclops", std::vector<c10::IValue>({}));
    LazyInitAclopsCore();
    ASCEND_LOGI("Init for aclops finished.")
}

}  // namespace aclops
}  // namespace at_npu
