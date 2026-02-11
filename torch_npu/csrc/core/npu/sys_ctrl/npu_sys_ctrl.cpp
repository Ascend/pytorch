#ifndef BUILD_LIBTORCH
#include <torch/csrc/python_headers.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_numbers.h>
#endif

#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUAffinityController.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "third_party/acl/inc/acl/acl_op_compiler.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/framework/LazyInitAclops.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"
#ifdef SUCCESS
#undef SUCCESS
#endif
#ifdef FAILED
#undef FAILED
#endif


namespace {
const uint32_t kMaxOpExecuteTimeOut = 547U;
const size_t kMaxPathLen = 4096U;

void SetDefaultAllowInternalFromatDisable()
{
    auto allow_internal_format = c10_npu::option::GetOption("ALLOW_INTERNAL_FORMAT");
    if (allow_internal_format.has_value() && allow_internal_format.value() != "") {
        return;
    }

    c10_npu::option::SetOption("ALLOW_INTERNAL_FORMAT", "disable");
    ASCEND_LOGI("Set ALLOW_INTERNAL_FORMAT default value disable.");
}

void SetDeterministicFromLevel()
{
    const static bool isAclStrongConsistencyExist = []() {
        const std::string kMinRuntimeVersion = "8.5.0";
        if (IsGteCANNVersion(kMinRuntimeVersion, "RUNTIME")) {
            return true;
        }
        return false;
    }();
    if (!isAclStrongConsistencyExist) {
        NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, 0));
        return;
    }

    uint32_t level = c10_npu::GetDeterministicLevel();
    if (level == 1) {
        NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, 1));
        NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_STRONG_CONSISTENCY, 0));
        return;
    } else if (level == 2) {
        NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, 1));
        NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_STRONG_CONSISTENCY, 1));
        return;
    } else if (level != 0) {
        TORCH_CHECK(false, "'torch_npu.npu.set_deterministic_level' currently only supports configuring 0/1/2 !", PTA_ERROR(ErrCode::VALUE));
    }
    NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, 0));
    NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_STRONG_CONSISTENCY, 0));
}

#ifndef BUILD_LIBTORCH
std::string GetTorchNpuFile()
{
    PyObject *file_attr = nullptr;
    {
        pybind11::gil_scoped_acquire get_gil;
        auto torch = THPObjectPtr(PyImport_ImportModule("torch"));
        if (!torch) {
            throw python_error();
        }
        file_attr = PyObject_GetAttrString(torch, "__file__");
    }
    if (file_attr) {
        const char *file_path = PyUnicode_AsUTF8(file_attr);
        std::string file_path_str = std::string(file_path);
        std::string key_word = "torch";
        size_t pos = file_path_str.rfind(key_word);
        if (pos != std::string::npos) {
            return file_path_str.substr(0, pos);
        }
    }
    ASCEND_LOGW("Failed to get __file__ attribute.");
    return "";
}
#endif

std::string GetAclConfigJsonPath()
{
#ifndef BUILD_LIBTORCH
    std::string npu_path = GetTorchNpuFile();
    if (npu_path == "") {
        ASCEND_LOGW("Failed to get npu path!");
        return "";
    }
    std::string json_path = "";
    if (c10_npu::is_lazy_set_device()) {
        json_path = npu_path.append("torch_npu/acl_default.json");
    } else {
        json_path = npu_path.append("torch_npu/acl.json");
    }
    std::string json_path_str = torch_npu::toolkit::profiler::Utils::RealPath(json_path);
    if (json_path_str == "") {
        ASCEND_LOGW("this path:%s is not exist!", json_path.c_str());
    }
    return json_path_str;
#else
    return "";
#endif
}
} // namespace

namespace c10_npu {

NpuSysCtrl::NpuSysCtrl() : repeat_init_acl_flag_(true), init_flag_(false), lazy_init_flag_(false), device_id_(0) {}

// Get NpuSysCtrl singleton instance
NpuSysCtrl &NpuSysCtrl::GetInstance()
{
    static NpuSysCtrl instance;
    return instance;
}

// Environment Initialize, return Status: SUCCESS, FAILED
NpuSysCtrl::SysStatus NpuSysCtrl::Initialize(int device_id)
{
    if (init_flag_) {
        return INIT_SUCC;
    }
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (init_flag_) {
        return INIT_SUCC;
    }
    std::string json_path = GetAclConfigJsonPath();
    const char *json_path_ptr = json_path == "" ? nullptr : json_path.c_str();
    ASCEND_LOGD("get acl json path:%s.", json_path_ptr);
    auto init_ret = aclInit(json_path_ptr);
    // The aclinit function is an important low-level aclInit interface that can only be called once in PTA.
    // However, because of the different business codes, aclinit may be successfully invoked by other frameworks or components.
    // ACL_ERROR_REPEAT_INITIALIZE means that aclInit is not called by PTA, so we save the flag variable to control aclFinalize.
    if (init_ret == ACL_ERROR_REPEAT_INITIALIZE) {
        repeat_init_acl_flag_ = false;
        ASCEND_LOGI("acl has allready init by other component.");
    } else if (init_ret != ACL_ERROR_NONE) {
        NPU_CHECK_ERROR(init_ret, "aclInit");
    }

    if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()) {
        NPU_CHECK_ERROR(aclmdlInitDump());
        ASCEND_LOGD("dump init success");
    }

    c10_npu::NPUCachingAllocator::init();
    ASCEND_LOGD("Npu caching allocator initialize successfully");
    c10_npu::NPUWorkspaceAllocator::init();
    ASCEND_LOGD("Npu workspace allocator initialize successfully");
    c10_npu::option::OptionsManager::IsOomSnapshotEnable();

    if (!c10_npu::is_lazy_set_device()) {
        // There's no need to call c10_npu::GetDevice at the start of the process, because device 0 may not be needed
        auto ret = aclrtGetDevice(&device_id_);
        if (ret != ACL_ERROR_NONE) {
            device_id_ = (device_id == -1) ? 0 : device_id;
            NPU_CHECK_ERROR(c10_npu::SetDevice(device_id_));
        } else {
            ASCEND_LOGW("Npu device %d has been set before global init.", device_id_);
        }
    } else {
        if (device_id >= 0) {
            NPU_CHECK_ERROR(c10_npu::SetDevice(device_id));
        }
    }

    if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()) {
        const char *aclConfigPath = "acl.json";
        NPU_CHECK_ERROR(aclmdlSetDump(aclConfigPath));
        ASCEND_LOGD("set dump config success");
    }

    auto soc_name = c10_npu::acl::AclGetSocName();
    // set global soc name
    c10_npu::SetSocVersion(soc_name);

    if (!c10_npu::is_lazy_set_device()) {
        if (c10_npu::IsSupportInfNan()) {
            c10_npu::acl::AclrtSetDeviceSatMode(aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_INFNAN);
        } else {
            c10_npu::acl::AclrtSetDeviceSatMode(aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_SATURATION);
        }
    }

    auto acl_op_init_mode = c10_npu::option::OptionsManager::GetAclOpInitMode();
    if (acl_op_init_mode == 0) {
        at_npu::aclops::InitAclops();
    } else {
        at_npu::aclops::InitializeJitCompilationMode();
    }

    // set default allow_internal_format value
    if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910_9391) {
        SetDefaultAllowInternalFromatDisable();
    }

    if (!c10_npu::is_lazy_set_device()) {
        SetDeterministicFromLevel();
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSetOpExecuteTimeOut(kMaxOpExecuteTimeOut));
    }

    // lazy call for the setoption
    for (const auto &iter: lazy_fn_) {
        ASCEND_LOGD("start setoption for the lazy call.");
        const auto &call_ = iter.first;
        const auto &in = iter.second;
        call_(in);
    }

    lazy_fn_.clear();

    SetMainThread();
    if (SetThreadAffinityInInitialize()) {
        SetThreadAffinity(device_id_);
    }

    init_flag_ = true;
    ASCEND_LOGD("Npu sys ctrl initialize successfully.");

    return INIT_SUCC;
}

NpuSysCtrl::SysStatus NpuSysCtrl::LazyInitialize(int device_id)
{
    if (!c10_npu::is_lazy_set_device()) {
        return INIT_SUCC;
    }

    if (lazy_init_flag_) {
        return INIT_SUCC;
    }
    std::lock_guard<std::mutex> lock(lazy_init_mutex_);
    if (lazy_init_flag_) {
        return INIT_SUCC;
    }

    // There's no need to call c10_npu::GetDevice at the start of the process, because device 0 may not be needed
    auto ret = aclrtGetDevice(&device_id_);

    if (c10_npu::IsSupportInfNan()) {
        c10_npu::acl::AclrtSetDeviceSatMode(aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_INFNAN);
    } else {
        c10_npu::acl::AclrtSetDeviceSatMode(aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_SATURATION);
    }

    SetDeterministicFromLevel();
    NPU_CHECK_ERROR(c10_npu::acl::AclrtSetOpExecuteTimeOut(kMaxOpExecuteTimeOut));

    lazy_init_flag_ = true;
    ASCEND_LOGD("Npu sys ctrl Lazyinitialize successfully.");

    return INIT_SUCC;
}

NpuSysCtrl::SysStatus NpuSysCtrl::ExchangeDevice(int device)
{
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::SetDevice(device));
    device_id_ = device;
    return INIT_SUCC;
}

NpuSysCtrl::SysStatus NpuSysCtrl::BackwardsInit()
{
    NPU_CHECK_ERROR(c10_npu::SetDevice(device_id_));
    return INIT_SUCC;
}

NpuSysCtrl::SysStatus NpuSysCtrl::OverflowSwitchEnable()
{
    if (!c10_npu::IsSupportInfNan()) {
        c10_npu::acl::AclrtSetStreamOverflowSwitch(c10_npu::getCurrentNPUStream(), 1);
        ASCEND_LOGI("Npu overflow check switch set successfully.");
    }
    return INIT_SUCC;
}

// Environment Finalize, return SysStatus
NpuSysCtrl::SysStatus NpuSysCtrl::Finalize()
{
    if (!init_flag_) {
        return FINALIZE_SUCC;
    }

    this->RegisterReleaseFn(
        [=]() -> void {
            c10_npu::NPUEventManager::GetInstance().ClearEvent();
            NPU_CHECK_WARN(c10_npu::DestroyUsedStreams());
            NPU_CHECK_WARN(c10_npu::ResetUsedDevices());
            // Maintain a basic point of view, who applies for the resource, the resource is released by whom.
            // If aclInit is not a PTA call, then aclFinalize should not be a PTA call either.
            if (repeat_init_acl_flag_) {
                NPU_CHECK_WARN(aclFinalize());
            }
        },
        ReleasePriority::PriorityLast);

    init_flag_ = false;

    if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()) {
        NPU_CHECK_WARN(aclmdlFinalizeDump());
    }

    // call release fn by priotity
    for (const auto &iter: release_fn_) {
        const auto &fn_vec = iter.second;
        for (const auto &fn: fn_vec) {
            fn();
        }
    }
    release_fn_.clear();

    ASCEND_LOGD("Npu sys ctrl finalize successfully.");
    return FINALIZE_SUCC;
}

bool NpuSysCtrl::GetInitFlag()
{
    return init_flag_;
}

bool NpuSysCtrl::GetLazyInitFlag()
{
    return lazy_init_flag_;
}

int NpuSysCtrl::InitializedDeviceID()
{
    if (GetInitFlag()) {
        return device_id_;
    }
    TORCH_CHECK(false, "no npu device has been initialized!", PTA_ERROR(ErrCode::INTERNAL));
    return -1;
}

void NpuSysCtrl::RegisterLazyFn(const option::OptionCallBack& call_, const std::string& in)
{
    lazy_fn_.emplace_back(std::make_pair(call_, in));
}

void NpuSysCtrl::RegisterReleaseFn(ReleaseFn release_fn, ReleasePriority priority)
{
    const auto& iter = this->release_fn_.find(priority);
    if (iter != release_fn_.end()) {
        release_fn_[priority].emplace_back(release_fn);
    } else {
        release_fn_[priority] = (std::vector<ReleaseFn>({release_fn}));
    }
}

aclError SetCurrentDevice()
{
    if (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        c10_npu::SetDevice(c10_npu::NpuSysCtrl::GetInstance().InitializedDeviceID());
        return ACL_SUCCESS;
    }
    TORCH_CHECK(false, "npu device has not been inited.", PTA_ERROR(ErrCode::INTERNAL));
}

} // namespace c10_npu
