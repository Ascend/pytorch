#include <atomic>
#include <mutex>
#include <sstream>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <ATen/Context.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUAffinityController.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif

namespace c10_npu {

static uint32_t dev_count = 0;
static thread_local int local_device = -1;
static std::unordered_map<int8_t, aclrtContext> used_devices;
std::recursive_mutex mtx;
thread_local int targetDeviceIndex = -1;
static std::atomic<uint32_t> deterministic_level = {0};
static std::mutex deterministic_state_mutex;

namespace {
constexpr uint32_t kMaxDeterministicLevel = 3;
constexpr const char* kLevel3MinRuntimeVersion = "9.2.0";
constexpr const char* kLevel3MinOppVersion = "9.2.0";
constexpr const char* kLevel3MinOppKernelVersion = "9.2.0";
constexpr const char* kLevel3MinOpsPkgVersion = "9.2.0";

struct DeterministicVersionFailure {
    std::string module;
    std::string current_version;
    std::string required_version;
};

struct DeterministicLevel3VersionCheck {
    bool supported = false;
    std::vector<DeterministicVersionFailure> failures;
};

bool IsModuleVersionGt(const std::string& module, const std::string& required_version)
{
    return IsGteCANNVersion(required_version, module);
}

void RecordVersionFailure(
    std::vector<DeterministicVersionFailure>& failures,
    const std::string& module,
    const std::string& required_version)
{
    failures.push_back({module, GetCANNVersion(module), required_version});
}

bool CheckVersionGroup(
    const std::vector<std::pair<std::string, std::string>>& requirements,
    std::vector<DeterministicVersionFailure>& failures)
{
    bool supported = true;
    for (const auto& requirement : requirements) {
        if (!IsModuleVersionGt(requirement.first, requirement.second)) {
            RecordVersionFailure(failures, requirement.first, requirement.second);
            supported = false;
        }
    }
    return supported;
}

DeterministicLevel3VersionCheck CheckDeterministicLevel3Version()
{
    DeterministicLevel3VersionCheck check;
    std::vector<DeterministicVersionFailure> runtime_failures;
    const bool runtime_supported = CheckVersionGroup(
        {{"RUNTIME", kLevel3MinRuntimeVersion}},
        runtime_failures);

    std::vector<DeterministicVersionFailure> legacy_pkg_failures;
    const bool legacy_pkg_supported = CheckVersionGroup(
        {
            {"OPP", kLevel3MinOppVersion},
            {"OPP_KERNEL", kLevel3MinOppKernelVersion},
        },
        legacy_pkg_failures);

    std::vector<DeterministicVersionFailure> split_pkg_failures;
    const bool split_pkg_supported = CheckVersionGroup(
        {
            {"ops_math", kLevel3MinOpsPkgVersion},
            {"ops_nn", kLevel3MinOpsPkgVersion},
            {"ops_transformer", kLevel3MinOpsPkgVersion},
            {"ops_cv", kLevel3MinOpsPkgVersion},
            {"ops_legacy", kLevel3MinOpsPkgVersion},
        },
        split_pkg_failures);

    check.supported = runtime_supported && (legacy_pkg_supported || split_pkg_supported);
    if (check.supported) {
        return check;
    }

    check.failures.insert(check.failures.end(), runtime_failures.begin(), runtime_failures.end());
    if (!legacy_pkg_supported && !split_pkg_supported) {
        check.failures.insert(check.failures.end(), legacy_pkg_failures.begin(), legacy_pkg_failures.end());
        check.failures.insert(check.failures.end(), split_pkg_failures.begin(), split_pkg_failures.end());
    }
    return check;
}

const DeterministicLevel3VersionCheck& GetDeterministicLevel3VersionCheck()
{
    static const DeterministicLevel3VersionCheck check = CheckDeterministicLevel3Version();
    return check;
}

void ThrowLevel3UnsupportedError()
{
    const auto& check = GetDeterministicLevel3VersionCheck();
    std::ostringstream oss;
    oss << "'torch_npu.npu.set_deterministic_level(3)' requires CANN runtime and operator packages "
        << "that support batch consistency. Please upgrade CANN runtime and operator packages.";
    if (!check.failures.empty()) {
        oss << " Unsatisfied versions:";
        for (const auto& failure : check.failures) {
            oss << " " << failure.module << "(current="
                << (failure.current_version.empty() ? "unavailable" : failure.current_version)
                << ", required>" << failure.required_version << ");";
        }
    }
    TORCH_CHECK(false, oss.str(), PTA_ERROR(ErrCode::VALUE));
}
} //


bool is_lazy_set_device()
{
    static bool is_lazy_set = []() {
        const std::string baseCannversion = "8.3.RC1";
        const std::string baseCannModule = "CANN";
        bool lazy_val = IsGteCANNVersion(baseCannversion, baseCannModule);
        ASCEND_LOGW("is_lazy_set_device %d", lazy_val);
        return lazy_val;
    }();
    return is_lazy_set;
}

c10::DeviceIndex device_count() noexcept
{
    // initialize number of devices only once
    if (dev_count == 0) {
        aclError error = aclrtGetDeviceCount(&dev_count);
        if (error != ACL_ERROR_NONE) {
            CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(error);
            ASCEND_LOGE("get device count of NPU failed");
            return 0;
        }
        return static_cast<c10::DeviceIndex>(dev_count);
    }
    return static_cast<c10::DeviceIndex>(dev_count);
}

bool hasPrimaryContext(c10::DeviceIndex device_index)
{
    /* This interface is implemented to align with 'c10::cuda::hasPrimaryContext(device_index)' interface,
     * but it has performance overhead due to calling AclrtGetPrimaryCtxState API.
     * For internal usage, it's recommended to use the more performant isDeviceCtxActive
     * function which checks the locally cached device context state.
     */
    TORCH_CHECK(device_index >= 0 && device_index < device_count(),
        "hasPrimaryContext expects a valid device index, but got device_index=", device_index, PTA_ERROR(ErrCode::VALUE));
    int32_t ctx_is_active = 0;
    NPU_CHECK_ERROR_WITHOUT_UCE(acl::AclrtGetPrimaryCtxState(device_index, nullptr, &ctx_is_active));
    return ctx_is_active == 1;
}

c10::DeviceIndex device_count_ensure_non_zero()
{
    unsigned int count = 0;

    NPU_CHECK_ERROR_WITHOUT_UCE(aclrtGetDeviceCount(&count));
    TORCH_CHECK(count, "No NPUs are available", PTA_ERROR(ErrCode::UNAVAIL));

    return static_cast<c10::DeviceIndex>(count);
}

aclError GetDevice(int32_t *device)
{
    if (targetDeviceIndex >= 0) {
        *device = targetDeviceIndex;
        return ACL_ERROR_NONE;
    }

    if (local_device >= 0) {
        *device = local_device;
        return ACL_ERROR_NONE;
    }
    aclError err =  aclrtGetDevice(device);
    if (err != ACL_ERROR_NONE) {
        CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
    }
    if (!is_lazy_set_device()) {
        if (err == ACL_ERROR_NONE) {
            local_device = *device;
        }
    }
    // before call aclinit with defaultdevice
    if (err == ACL_ERROR_RT_CONTEXT_NULL) {
        *device = 0;
        return ACL_ERROR_NONE;
    }
    return err;
}

aclError GetDeviceWithoutSet(int32_t *device)
{
    if (targetDeviceIndex >= 0) {
        *device = targetDeviceIndex;
        return ACL_ERROR_NONE;
    }

    if (local_device >= 0) {
        *device = local_device;
        return ACL_ERROR_NONE;
    }
    aclError err =  aclrtGetDevice(device);
    if (err != ACL_ERROR_NONE) {
        CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
    }
    if (!is_lazy_set_device()) {
        if (err == ACL_ERROR_NONE) {
            local_device = *device;
        }
    }
    // before call aclinit with defaultdevice
    if (err == ACL_ERROR_RT_CONTEXT_NULL) {
        *device = -1;
        return ACL_ERROR_NONE;
    }
    return err;
}

aclError SetDevice(c10::DeviceIndex device)
{
    TORCH_CHECK(device >= 0, "device id must be positive!", PTA_ERROR(ErrCode::VALUE));
    targetDeviceIndex = -1;
    if (local_device == device) {
        return ACL_ERROR_NONE;
    }

    if (c10_npu::NeedMainThreadBind()) {
        c10_npu::SetThreadAffinity(device);
    }

    aclError err = aclrtSetDevice(device);
    if (err == ACL_ERROR_NONE) {
        local_device = device;
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (used_devices.find(local_device) == used_devices.end()) {
            NPU_CHECK_ERROR_WITHOUT_UCE(aclrtGetCurrentContext(&used_devices[local_device]));
        }
    }
    return err;
}

aclError MaybeSetDevice(c10::DeviceIndex device)
{
    if (isDeviceCtxActive(device)) {
        NPU_CHECK_ERROR_WITHOUT_UCE(SetDevice(device));
    } else {
        ASCEND_LOGI("MaybeSetDevice: NPU device %d has not been initialized! We will set targetDeviceIndex.", device);
        targetDeviceIndex = device;
    }
    return ACL_ERROR_NONE;
}

std::vector<int8_t> GetUsedDevices()
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    std::vector<int8_t> used_devices_list;
    for (const auto it : used_devices) {
        used_devices_list.emplace_back(it.first);
    }
    return used_devices_list;
}

aclError ResetUsedDevices()
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    for (const auto it : used_devices) {
        aclError err = aclrtResetDevice(it.first);
        if (err != ACL_ERROR_NONE) {
            return err;
        }
    }
    used_devices.clear();
    return ACL_ERROR_NONE;
}

aclError DestroyUsedStreams()
{
    int32_t cur_device = 0;
    NPU_CHECK_ERROR_WITHOUT_UCE(GetDevice(&cur_device));
    std::lock_guard<std::recursive_mutex> lock(mtx);
    for (const auto it : used_devices) {
        if (c10_npu::StreamInitFlag(it.first)) {
            NPU_CHECK_ERROR_WITHOUT_UCE(SetDevice(it.first));
            NPUStream stream = getCurrentNPUStream(it.first);
            aclError acl_ret = acl::AclrtDestroyStreamForce(stream.stream(false));
            if (acl_ret != ACL_ERROR_NONE) {
                return acl_ret;
            }
        }
    }
    NPU_CHECK_ERROR_WITHOUT_UCE(MaybeSetDevice(cur_device));
    return ACL_ERROR_NONE;
}

aclError SynchronizeUsedDevices()
{
    int32_t cur_device = 0;
    NPU_CHECK_ERROR_WITHOUT_UCE(GetDevice(&cur_device));
    std::lock_guard<std::recursive_mutex> lock(mtx);
    for (const auto it : used_devices) {
        NPU_CHECK_ERROR_WITHOUT_UCE(SetDevice(it.first));
        aclError acl_ret = c10_npu::acl::AclrtSynchronizeDeviceWithTimeout();
        if (acl_ret != ACL_ERROR_NONE) {
            CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(acl_ret);
            return acl_ret;
        }
#ifndef BUILD_LIBTORCH
        const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuDeviceSynchronization();
        }
#endif
    }
    NPU_CHECK_ERROR_WITHOUT_UCE(MaybeSetDevice(cur_device));
    return ACL_ERROR_NONE;
}

aclrtContext GetDeviceContext(int32_t device)
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if (used_devices.find(device) == used_devices.end()) {
        ASCEND_LOGE("NPU device %d has not been initialized! Can not get context", device);
        return nullptr;
    }
    return used_devices[device];
}

bool isDeviceCtxActive(int32_t device)
{
    if (at_npu::native::env::CheckCompatibleImpl()) {
        return hasPrimaryContext(device);
    }
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if (used_devices.find(device) == used_devices.end()) {
        return false;
    }
    return used_devices[device] != nullptr;
}

std::optional<c10::DeviceIndex> getDeviceIndexWithPrimaryContext()
{
    // check current device first
    auto cur_device = current_device();
    if (cur_device >= 0) {
        if (isDeviceCtxActive(cur_device)) {
            return cur_device;
        }
    }
    for (const auto device_index : c10::irange(device_count())) {
        if (device_index == cur_device) {
            continue;
        }
        if (isDeviceCtxActive(device_index)) {
            return device_index;
        }
    }
    return std::nullopt;
}

c10::DeviceIndex current_device()
{
    int cur_device = 0;
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::GetDevice(&cur_device));
    return static_cast<c10::DeviceIndex>(cur_device);
}

void set_device(c10::DeviceIndex device)
{
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::SetDevice(device));
}

void device_synchronize()
{
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::acl::AclrtSynchronizeDeviceWithTimeout());
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuDeviceSynchronization();
    }
#endif
}

int ExchangeDevice(int device)
{
    targetDeviceIndex = -1;
    NPU_CHECK_ERROR_WITHOUT_UCE(SetDevice(device));

    return device;
}

int MaybeExchangeDevice(int to_device)
{
    int cur_device = -1;
    NPU_CHECK_ERROR_WITHOUT_UCE(GetDeviceWithoutSet(&cur_device));
    if (to_device == cur_device) {
        return cur_device;
    }
    if (isDeviceCtxActive(to_device)) {
        NPU_CHECK_ERROR_WITHOUT_UCE(SetDevice(to_device));
    } else {
        ASCEND_LOGI("NPU device %d has not been initialized! We will set targetDeviceIndex.", to_device);
        targetDeviceIndex = to_device;
    }
    return cur_device;
}

void SetTargetDevice()
{
    if (targetDeviceIndex >= 0) {
        NPU_CHECK_ERROR_WITHOUT_UCE(SetDevice(targetDeviceIndex));
    }
}

bool IsContextInitialized()
{
    if (local_device >= 0) {
        return true;
    }

    if (is_lazy_set_device()) {
        return false;
    }

    int32_t device = -1;
    aclError err = aclrtGetDevice(&device);
    if (err == ACL_ERROR_NONE) {
        return true;
    } else {
        CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
        if (err == ACL_ERROR_RT_CONTEXT_NULL) {
            return false;
        }
        NPU_CHECK_ERROR_WITHOUT_UCE(err);
        return false;
    }
}

int GetLocalDevice()
{
    return local_device;
}

void LazySetDevice(c10::DeviceIndex device)
{
    if (local_device != device) {
        aclError err = aclrtSetDevice(device);
        if (err == ACL_ERROR_NONE) {
            local_device = device;
            std::lock_guard<std::recursive_mutex> lock(mtx);
            if (used_devices.find(local_device) == used_devices.end()) {
                NPU_CHECK_ERROR_WITHOUT_UCE(aclrtGetCurrentContext(&used_devices[local_device]));
            }
        }
        NPU_CHECK_ERROR_WITHOUT_UCE(err);
    }
}

void warn_or_error_on_sync()
{
    if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
        TORCH_CHECK(false, "called a synchronizing NPU operation", PTA_ERROR(ErrCode::ACL));
    } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
        TORCH_NPU_WARN("called a synchronizing NPU operation");
    }
}

void stream_synchronize(aclrtStream stream)
{
    if (C10_UNLIKELY(warning_state().get_sync_debug_mode() != SyncDebugMode::L_DISABLED)) {
        warn_or_error_on_sync();
    }
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger *trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuStreamSynchronization(reinterpret_cast<uintptr_t>(stream));
    }
#endif
    NPU_CHECK_ERROR(aclrtSynchronizeStream(stream));
}

aclError SetDeviceResLimit(int32_t device, int32_t type, uint32_t value)
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if (used_devices.find(device) == used_devices.end()) {
        TORCH_CHECK(false, "NPU device ", device, " has not been initialized! Can not get device resource limit");
    }
    TORCH_CHECK(device >= 0, "device id must be positive!", PTA_ERROR(ErrCode::VALUE));
    c10_npu::acl::aclrtDevResLimitType restype = static_cast<c10_npu::acl::aclrtDevResLimitType>(type);
    aclError err = c10_npu::acl::AclrtSetDeviceResLimit(device, restype, value);
    NPU_CHECK_ERROR(err);
    return err;
}

uint32_t GetDeviceResLimit(int32_t device, int32_t type)
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if (used_devices.find(device) == used_devices.end()) {
        TORCH_CHECK(false, "NPU device ", device, " has not been initialized! Can not get device resource limit");
    }
    TORCH_CHECK(device >= 0, "device id must be positive!", PTA_ERROR(ErrCode::VALUE));
    c10_npu::acl::aclrtDevResLimitType restype = static_cast<c10_npu::acl::aclrtDevResLimitType>(type);
    uint32_t value;
    NPU_CHECK_ERROR(c10_npu::acl::AclrtGetDeviceResLimit(device, restype, &value));
    return value;
}

aclError ResetDeviceResLimit(int32_t device)
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if (used_devices.find(device) == used_devices.end()) {
        TORCH_CHECK(false, "NPU device ", device, " has not been initialized! Can not reset device resource limit");
    }
    TORCH_CHECK(device >= 0, "device id must be positive!", PTA_ERROR(ErrCode::VALUE));
    aclError err = c10_npu::acl::AclrtResetDeviceResLimit(device);
    NPU_CHECK_ERROR(err);
    return err;
}

aclError SetStreamResLimit(NPUStream npu_stream, int32_t type, uint32_t value)
{
    c10_npu::acl::aclrtDevResLimitType restype = static_cast<c10_npu::acl::aclrtDevResLimitType>(type);
    aclError err = c10_npu::acl::AclrtSetStreamResLimit(npu_stream.stream(), restype, value);
    enable_core_control.store(true, std::memory_order_relaxed);
    NPU_CHECK_ERROR(err);
    return err;
}

aclError ResetStreamResLimit(NPUStream npu_stream)
{
    aclError err = c10_npu::acl::AclrtResetStreamResLimit(npu_stream.stream());
    NPU_CHECK_ERROR(err);
    return err;
}

uint32_t GetStreamResLimit(NPUStream npu_stream, int32_t type)
{
    c10_npu::acl::aclrtDevResLimitType restype = static_cast<c10_npu::acl::aclrtDevResLimitType>(type);
    uint32_t value;
    NPU_CHECK_ERROR(c10_npu::acl::AclrtGetStreamResLimit(npu_stream.stream(false), restype, &value));
    return value;
}

aclError UseStreamResInCurrentThread(aclrtStream stream)
{
    aclError err = c10_npu::acl::AclrtUseStreamResInCurrentThread(stream);
    NPU_CHECK_ERROR(err);
    return err;
}

aclError UnuseStreamResInCurrentThread(aclrtStream stream)
{
    aclError err = c10_npu::acl::AclrtUnuseStreamResInCurrentThread(stream);
    NPU_CHECK_ERROR(err);
    return err;
}

uint32_t GetResInCurrentThread(int32_t type)
{
    c10_npu::acl::aclrtDevResLimitType restype = static_cast<c10_npu::acl::aclrtDevResLimitType>(type);
    uint32_t value;
    NPU_CHECK_ERROR(c10_npu::acl::AclrtGetResInCurrentThread(restype, &value));
    return value;
}

void SetDeterministicLevel(uint32_t level)
{
    TORCH_CHECK(level <= kMaxDeterministicLevel,
        "'torch_npu.npu.set_deterministic_level' supports configuring 0/1/2/3.",
        PTA_ERROR(ErrCode::VALUE));
    if (level == kMaxDeterministicLevel && !IsSupportDeterministicLevel3()) {
        ThrowLevel3UnsupportedError();
    }

    std::lock_guard<std::mutex> lock(deterministic_state_mutex);
    deterministic_level.store(level, std::memory_order_release);
}

uint32_t GetDeterministicLevel()
{
    return deterministic_level.load(std::memory_order_acquire);
}

DeterministicSnapshot CaptureDeterministicSnapshot()
{
    DeterministicSnapshot snapshot;
    snapshot.deterministic_algorithms_enabled = at::globalContext().deterministicAlgorithms();
    snapshot.requested_level = GetDeterministicLevel();
    snapshot.effective_level = 0;
    if (snapshot.deterministic_algorithms_enabled) {
        snapshot.effective_level = snapshot.requested_level == 0 ? 1 : snapshot.requested_level;
    }
    snapshot.backend = GetDeterministicBackend();
    return snapshot;
}

uint32_t GetEffectiveDeterministicLevel()
{
    return CaptureDeterministicSnapshot().effective_level;
}

DeterministicBackend GetDeterministicBackend()
{
    return IsSupportDeterministicLevel3() ? DeterministicBackend::V2 : DeterministicBackend::Legacy;
}

bool IsSupportDeterministicLevel3()
{
    return GetDeterministicLevel3VersionCheck().supported;
}
} // namespace c10_npu
