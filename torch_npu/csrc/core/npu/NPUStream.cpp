#include <array>
#include <climits>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>
#include <sys/time.h>
#include <unistd.h>
#include <sstream>
#include <iostream>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUQueue.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif

namespace c10_npu {

std::atomic<bool> enable_core_control{false};

namespace {
struct LeakyStreamInternals {
    LeakyStreamInternals()
    {
        repo = ::std::make_unique<Repository>();
    }
    C10_DISABLE_COPY_AND_ASSIGN(LeakyStreamInternals);

    ~LeakyStreamInternals()
    {
        // NB: this code is invoked only in the destruction of global variables
        // (since we never shrink the corresponding vectors). At this point the NPU
        // runtime might be already destroyed and invoking npuStreamDestroy leads
        // to a crash. It's likely an issue in NPU, but to be safe - let's just
        // "forget" the destruction.
    }

    c10::DeviceIndex device_index = -1;
    int32_t stream_id = -1;
    aclrtStream stream = nullptr;
    ::std::unique_ptr<NPUQueueBase> repo = nullptr;
    bool is_data_preprocess_stream = false;
    bool is_repo_stop = false;
    bool is_sync_launch = false;
    aclrtStream prev_stream = nullptr;
};
// Global stream state and constants
static c10::DeviceIndex num_npus = -1;
static constexpr int kStreamsPerPoolBits = 5;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
static constexpr int kMaxStreamPriorities = 2;
static constexpr int kSyncLaunchStreamsPerPool = 4;
// Default streams init flags
static bool initialize_flag[C10_COMPILE_TIME_MAX_NPUS] = {false};
std::mutex mtx[C10_COMPILE_TIME_MAX_NPUS];
// initrepo mutex
std::mutex init_repo_mutex_;
// The stream that delivers the compute task.
static LeakyStreamInternals default_streams[C10_COMPILE_TIME_MAX_NPUS];
// In a specific scenario, the two operators have no value dependence
// and different execution hardware, so they can be executed in parallel
// on the default stream and the secondary stream respectively.
static LeakyStreamInternals secondary_streams[C10_COMPILE_TIME_MAX_NPUS];
// npu streams pool init flags
static std::once_flag device_priority_flags[C10_COMPILE_TIME_MAX_NPUS][kMaxStreamPriorities];
// SyncLaunch streams pool init flags
static std::once_flag device_sync_launch_flags[C10_COMPILE_TIME_MAX_NPUS];
static std::array<
    std::array<std::atomic<uint32_t>, kMaxStreamPriorities>,
    C10_COMPILE_TIME_MAX_NPUS>
    npu_counters;
static std::atomic<uint32_t> sync_stream_counters[C10_COMPILE_TIME_MAX_NPUS];
// npu_streams is a stream pool, each device has a stream pool,
// and 8 streams are created in each pool.
static std::array<
    std::array<
        std::array<LeakyStreamInternals, kStreamsPerPool>,
        kMaxStreamPriorities>,
    C10_COMPILE_TIME_MAX_NPUS>
    npu_streams;
static thread_local std::unique_ptr<LeakyStreamInternals* []> current_streams = nullptr;

static std::array<LeakyStreamInternals, kSyncLaunchStreamsPerPool> sync_launch_streams[C10_COMPILE_TIME_MAX_NPUS];

thread_local aclrtStream tls_prev_stream = nullptr;

enum class StreamIdType : uint8_t {
    DEFAULT = 0x0,
    SECONDARY = 0x1,
    SYNCLAUNCH = 0x2,
    NORMAL = 0x3,
    HIGH = 0x4,
};

std::ostream& operator<<(std::ostream& stream, StreamIdType s)
{
    switch (s) {
        case StreamIdType::DEFAULT:
            stream << "DEFAULT";
            break;
        case StreamIdType::NORMAL:
            stream << "NORMAL";
            break;
        case StreamIdType::HIGH:
            stream << "HIGH";
            break;
        case StreamIdType::SECONDARY:
            stream << "SECONDARY";
            break;
        case StreamIdType::SYNCLAUNCH:
            stream << "SYNCLAUNCH";
            break;
        default:
            stream << static_cast<uint8_t>(s);
            break;
    }
    return stream;
}

int GetStreamsPerPoolBits()
{
    const static int StreamsPerPoolBits = []() -> int {
        if (c10_npu::option::OptionsManager::GetStreamsPerDevice() == 8) {
            return 3;
        }
        return kStreamsPerPoolBits;
    }();
    return StreamsPerPoolBits;
}

int GetStreamsPerPool()
{
    const static int StreamsPerPool = []() -> int {
        if (c10_npu::option::OptionsManager::GetStreamsPerDevice() == 8) {
            return 8;
        }
        return kStreamsPerPool;
    }();
    return StreamsPerPool;
}

static inline StreamIdType streamIdType(c10::StreamId s)
{
    static int StreamsPerPoolBits = GetStreamsPerPoolBits();
    return static_cast<StreamIdType>((uint32_t)s >> StreamsPerPoolBits);
}

static inline size_t streamIdIndex(c10::StreamId s)
{
    static int StreamsPerPoolBits = GetStreamsPerPoolBits();
    return static_cast<size_t>((uint32_t)s & ((1 << StreamsPerPoolBits) - 1));
}

c10::StreamId makeStreamId(StreamIdType st, size_t si)
{
    static int StreamsPerPoolBits = GetStreamsPerPoolBits();
    return static_cast<c10::StreamId>((static_cast<size_t>(st) << StreamsPerPoolBits) | si);
}

template <typename T, typename A>
static bool pointer_within(const T* ptr, const A& arr)
{
    return std::greater_equal<const T*>()(ptr, arr.data()) &&
        std::less<const T*>()(ptr, arr.data() + arr.size());
}

static c10::StreamId NPUStream_getStreamId(const LeakyStreamInternals* ptr)
{
    c10::DeviceIndex device_index = ptr->device_index;
    if (ptr == &default_streams[device_index]) {
        return makeStreamId(StreamIdType::DEFAULT, 0);
    }
    for (const auto p : c10::irange(kMaxStreamPriorities)) {
        if (pointer_within<LeakyStreamInternals>(ptr, npu_streams[device_index][p])) {
            return makeStreamId(StreamIdType(static_cast<uint8_t>(StreamIdType::NORMAL) + p),
                                ptr - npu_streams[device_index][p].data());
        }
    }
    if (pointer_within<LeakyStreamInternals>(ptr, sync_launch_streams[device_index])) {
        return makeStreamId(
            StreamIdType::SYNCLAUNCH, ptr - sync_launch_streams[device_index].data());
    }
    if (ptr == &secondary_streams[device_index]) {
        return makeStreamId(StreamIdType::SECONDARY, 0);
    }
    AT_ASSERTM(
        0,
        "Could not compute stream ID for ",
        ptr,
        " on device ",
        +device_index,
        " (something has gone horribly wrong!)", PTA_ERROR(ErrCode::PTR));
}

static void initGlobalStreamState()
{
    num_npus = c10_npu::device_count();
    // Check if the number of GPUs matches the expected compile-time max number
    // of GPUs.
    AT_ASSERTM(
        num_npus <= C10_COMPILE_TIME_MAX_NPUS,
        "Number of NPU devices on the machine is larger than the compiled "
        "max number of npus expected (",
        C10_COMPILE_TIME_MAX_NPUS,
        "). Increase that and recompile.", PTA_ERROR(ErrCode::VALUE));

    int device_id = 0;
    auto ret = c10_npu::GetDevice(&device_id);
    if (ret != ACL_ERROR_NONE) {
        ASCEND_LOGE("Device has not been set");
    }
    // Initializes default streams
    default_streams[device_id].device_index = device_id;
    for (const auto p : c10::irange(kMaxStreamPriorities)) {
        npu_counters[device_id][p] = 0;
    }
    auto& default_streamsi = default_streams[device_id];
    NPU_CHECK_ERROR(
        acl::AclrtCreateStreamWithConfig(&default_streamsi.stream, 0, (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC)));
    if (c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
        default_streamsi.repo->InitRepo(device_id);
    }
    // Initializes secondary streams
    secondary_streams[device_id].device_index = device_id;
    auto &secondary_streamsi = secondary_streams[device_id];
    NPU_CHECK_ERROR(
        acl::AclrtCreateStreamWithConfig(&secondary_streamsi.stream, 0, (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC)));
}

static void initDeviceStreamState(c10::DeviceIndex device_index, int p)
{
    // Switches to the requested device so streams are properly associated
    // with it.
    NPUGuard device_guard{device_index};
    static int StreamsPerPool = GetStreamsPerPool();
    for (auto i = decltype(StreamsPerPool){0}; i < StreamsPerPool; ++i) {
        auto& npu_streami = npu_streams[device_index][p][i];

        npu_streami.device_index = device_index;

        NPU_CHECK_ERROR(acl::AclrtCreateStreamWithConfig(
            &npu_streami.stream, 0, (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC)));
    }
}

static void initNPUStreamsOnce()
{
    // Inits default and secondary streams (once, globally)
    c10::DeviceIndex device_index = current_device();
    // makesure on real devcie
    SetTargetDevice();
    LazySetDevice(device_index);
    c10_npu::NpuSysCtrl::GetInstance().LazyInitialize();
    if (!initialize_flag[device_index]) {
        std::lock_guard<std::mutex> lock(mtx[device_index]);
        if (!initialize_flag[device_index]) {
            initGlobalStreamState();
            initialize_flag[device_index] = true;
        }
    }

    if (current_streams) {
        return;
    }

    // Inits current streams (thread local) to default streams
    current_streams = std::make_unique<LeakyStreamInternals* []>(num_npus);
    for (auto i = decltype(num_npus){0}; i < num_npus; ++i) {
        default_streams[i].device_index = i;
        current_streams[i] = &default_streams[i];
    }
}

static inline void check_npu(c10::DeviceIndex device_index)
{
    AT_ASSERT(device_index >= 0 && device_index < num_npus, "Invalid device_index : ", device_index,
              ", valid device_index range is [0, ", num_npus, ")", PTA_ERROR(ErrCode::VALUE));
}

static uint32_t get_idx(std::atomic<uint32_t>& counter)
{
    auto raw_idx = counter++;
    static int StreamsPerPool = GetStreamsPerPool();
    return raw_idx % static_cast<uint32_t>(StreamsPerPool);
}

static uint32_t get_sync_launch_stream_idx(std::atomic<uint32_t>& counter)
{
    auto raw_idx = counter++;
    return raw_idx % kSyncLaunchStreamsPerPool;
}

LeakyStreamInternals* NPUStream_internals(NPUStream s)
{
    c10::DeviceIndex device_index = s.device_index();
    StreamIdType st = streamIdType(s.unwrap().id());
    size_t si = streamIdIndex(s.unwrap().id());
    switch (st) {
        case StreamIdType::DEFAULT:
            AT_ASSERTM(
                si == 0,
                "Unrecognized stream ",
                s.unwrap(),
                " (I think this should be the default stream, but I got a non-zero index ",
                si,
                ").",
                " Did you manufacture the StreamId yourself?  Don't do that; use the",
                " official API like c10::cuda::getStreamFromPool() to get a new stream.", PTA_ERROR(ErrCode::PARAM));
            return &default_streams[device_index];
        case StreamIdType::NORMAL:
        case StreamIdType::HIGH:
            return &npu_streams[device_index][static_cast<uint8_t>(st) - static_cast<uint8_t>(StreamIdType::NORMAL)][si];
        case StreamIdType::SECONDARY:
            return &secondary_streams[device_index];
        case StreamIdType::SYNCLAUNCH:
            return &sync_launch_streams[device_index][si];
        default:
            AT_ASSERTM(
                0,
                "Unrecognized stream ",
                s.unwrap(),
                " (I didn't recognize the stream type, ",
                st,
                ")", PTA_ERROR(ErrCode::PARAM));
    }
}

NPUStream NPUStream_fromInternals(const LeakyStreamInternals* ptr)
{
    return NPUStream(
        NPUStream::UNCHECKED,
        c10::Stream(
            c10::Stream::UNSAFE,
            c10::Device(c10::DeviceType::PrivateUse1, ptr->device_index),
            NPUStream_getStreamId(ptr)));
}
} // namespace

bool NPUStream::query() const
{
    c10::DeviceGuard guard{stream_.device()};
    acl::aclrtStreamStatus status = acl::ACL_STREAM_STATUS_RESERVED;
    NPU_CHECK_ERROR(acl::AclrtStreamQuery(stream(), &status));
    if (status == acl::ACL_STREAM_STATUS_COMPLETE) {
        return true;
    }
    return false;
}

void NPUStream::synchronize() const
{
    c10::DeviceGuard guard{stream_.device()};
    NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream()));
}

aclrtStream NPUStream::stream() const
{
    if (c10_npu::option::OptionsManager::GetPerStreamQueue()) {
        auto cur_ptr = NPUStream_internals(*this);
        AT_ASSERT(cur_ptr, PTA_ERROR(ErrCode::PTR));
        if (!this->isSyncLaunchStream() && cur_ptr->repo->CheckInit()) {
            NPUStatus ret = cur_ptr->repo->MakeSureQueueEmpty();
            if (ret != NPU_STATUS_SUCCESS) {
                ASCEND_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
                return nullptr;
            }
        }
        return cur_ptr->stream;
    } else {
        auto ptr = NPUStream_internals(getDefaultNPUStream());
        AT_ASSERT(ptr, PTA_ERROR(ErrCode::PTR));
        if (!this->isSyncLaunchStream() && ptr->repo->CheckInit()) {
            NPUStatus ret = ptr->repo->MakeSureQueueEmpty();
            if (ret != NPU_STATUS_SUCCESS) {
                ASCEND_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
                return nullptr;
            }
        }
    }
    auto cur_ptr = NPUStream_internals(*this);
    AT_ASSERT(cur_ptr, PTA_ERROR(ErrCode::PTR));
    return cur_ptr->stream;
}

NPUStream getStreamFromPool(const int priority, c10::DeviceIndex device_index)
{
    initNPUStreamsOnce();
    if (device_index == -1) {
        device_index = current_device();
    }
    check_npu(device_index);

    auto pri_idx = std::clamp(-priority, 0, kMaxStreamPriorities - 1);
    // Initializes the stream pools (once)
    std::call_once(
        device_priority_flags[device_index][pri_idx], initDeviceStreamState, device_index, pri_idx);
    const auto idx = get_idx(npu_counters[device_index][pri_idx]);
    return NPUStream_fromInternals(&npu_streams[device_index][pri_idx][idx]);
}

NPUStream getNPUStreamFromPool(c10::DeviceIndex device_index)
{
    return getStreamFromPool(0, device_index);
}

NPUStream getStreamFromPool(const bool isHighPriority, c10::DeviceIndex device_index)
{
    initNPUStreamsOnce();
    int priority = isHighPriority ? -kMaxStreamPriorities + 1 : 0;
    return getStreamFromPool(priority, device_index);
}

NPUStream getDefaultNPUStream(c10::DeviceIndex device_index)
{
    initNPUStreamsOnce();
    if (device_index == -1) {
        device_index = current_device();
    }
    return NPUStream_fromInternals(&default_streams[device_index]);
}

NPUStream getCurrentNPUStream(c10::DeviceIndex device_index)
{
    initNPUStreamsOnce();
    if (device_index == -1) {
        device_index = current_device();
    }
    check_npu(device_index);
    return NPUStream_fromInternals(current_streams[device_index]);
}

NPUStream getCurrentSecondaryStream(c10::DeviceIndex device_index)
{
    initNPUStreamsOnce();
    if (device_index == -1) {
        device_index = current_device();
    }
    check_npu(device_index);
    return NPUStream_fromInternals(&secondary_streams[device_index]);
}

aclrtStream getCurrentNPUStreamNoWait(c10::DeviceIndex device_index)
{
    initNPUStreamsOnce();
    if (device_index == -1) {
        device_index = current_device();
    }
    check_npu(device_index);
    LeakyStreamInternals* ptr = current_streams[device_index];
    return ptr->stream;
}

NPUStatus emptyAllNPUStream(bool check_error)
{
    NPUStatus ret;
    for (auto i = decltype(num_npus){0}; i < num_npus; ++i) {
        auto& default_streamsi = default_streams[i];
        if (default_streamsi.stream == nullptr) {
            continue;
        }
        if (default_streamsi.stream != nullptr && default_streamsi.repo->CheckInit()) {
            ret = default_streamsi.repo->MakeSureQueueEmpty(check_error);
            if (ret != NPU_STATUS_SUCCESS) {
                return ret;
            }
        }
    }

    if (c10_npu::option::OptionsManager::GetPerStreamQueue()) {
        for (auto i = decltype(num_npus){0}; i < num_npus; ++i) {
            auto& secondary_streamsi = secondary_streams[i];
            if (secondary_streamsi.stream == nullptr) {
                continue;
            }
            if (secondary_streamsi.stream != nullptr && secondary_streamsi.repo->CheckInit()) {
                ret = secondary_streamsi.repo->MakeSureQueueEmpty(check_error);
                if (ret != NPU_STATUS_SUCCESS) {
                    return ret;
                }
            }
        }

        static int StreamsPerPool = GetStreamsPerPool();
        for (auto device_index = decltype(num_npus){0}; device_index < num_npus; ++device_index) {
            for (auto i = decltype(StreamsPerPool){0}; i < StreamsPerPool; ++i) {
                for (const auto p : c10::irange(kMaxStreamPriorities)) {
                    auto& npu_streami = npu_streams[p][device_index][i];

                    if (npu_streami.stream == nullptr) {
                        continue;
                    }
                    if (npu_streami.stream != nullptr && npu_streami.repo->CheckInit()) {
                        ret = npu_streami.repo->MakeSureQueueEmpty(check_error);
                        if (ret != NPU_STATUS_SUCCESS) {
                            return ret;
                        }
                    }
                }
            }
        }
    }

    return NPU_STATUS_SUCCESS;
}

std::string getRepoInfo()
{
    std::stringstream repo_info;
    for (auto i = decltype(num_npus){0}; i < num_npus; ++i) {
        auto& default_streamsi = default_streams[i];
        if (default_streamsi.stream == nullptr) {
            continue;
        }
        if (default_streamsi.stream != nullptr &&default_streamsi.repo->CheckInit()) {
            repo_info << "device " << (int)i << ": " << default_streamsi.repo->GetPara() << ". ";
        }
    }
    return repo_info.str();
}

void setRepoErrMsg(const char* errmsg)
{
    for (auto i = decltype(num_npus){0}; i < num_npus; ++i) {
        auto& default_streamsi = default_streams[i];
        if (default_streamsi.stream == nullptr) {
            continue;
        }
        if (default_streamsi.stream != nullptr &&default_streamsi.repo->CheckInit()) {
            default_streamsi.repo->SetQueueErrMsg(errmsg);
        }
    }
}

void setDefaultStreamsStatus(c10::DeviceIndex device_index, RepoStatus status)
{
    if (status == c10_npu::RepoStatus::STOP_EXIT) {
        default_streams[device_index].is_repo_stop = true;
    } else {
        default_streams[device_index].is_repo_stop = false;
    }
    if (default_streams[device_index].repo->CheckInit()) {
        default_streams[device_index].repo->SetStatus(status);
    }
}

bool npuSynchronizeDevice(bool check_error)
{
    if (c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
        NPUStatus ret = c10_npu::emptyAllNPUStream(check_error);
        if (ret != NPU_STATUS_SUCCESS) {
            ASCEND_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
        }
    }
    auto acl_ret = c10_npu::acl::AclrtSynchronizeDeviceWithTimeout();
    if (acl_ret != ACL_ERROR_NONE) {
        CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(acl_ret);
    }
#ifndef BUILD_LIBTORCH
    if (acl_ret == ACL_ERROR_NONE) {
        const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuDeviceSynchronization();
        }
    }
#endif
    if (check_error) {
        NPU_CHECK_ERROR(acl_ret, "AclrtSynchronizeDeviceWithTimeout");
    } else {
        NPU_CHECK_WARN(acl_ret);
    }
    return acl_ret == ACL_ERROR_NONE;
}

bool npuSynchronizeUsedDevices(bool check_error)
{
    if (c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
        NPUStatus ret = c10_npu::emptyAllNPUStream(check_error);
        if (ret != NPU_STATUS_SUCCESS) {
            ASCEND_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
        }
    }

    auto acl_ret = SynchronizeUsedDevices();
    if (check_error) {
        NPU_CHECK_ERROR(acl_ret);
    } else {
        NPU_CHECK_WARN(acl_ret);
    }
    return acl_ret == ACL_ERROR_NONE;
}

void enCurrentNPUStream(void* cur_paras, c10::DeviceIndex device_index, NPUStream *task_stream)
{
    initNPUStreamsOnce();
    if (device_index == -1) {
        device_index = current_device();
    }
    check_npu(device_index);
    c10_npu::queue::QueueParas* queueParam = static_cast<c10_npu::queue::QueueParas* >(cur_paras);
    queueParam->correlation_id = c10_npu::queue::QueueParas::g_correlation_id++;
    queueParam->paramStream = current_streams[device_index]->stream;
    if (c10_npu::option::OptionsManager::GetPerStreamQueue()) {
        LeakyStreamInternals* ptr = current_streams[device_index];
        if (task_stream != nullptr) {
            ptr = NPUStream_internals(*task_stream);
        }
        // To prevent all taskqueue threads from being created during stream initialization,
        // we initialize the taskqueue when enqueueing
        // each stream init repo once
        if (!ptr->repo->CheckInit()) {
            std::lock_guard<std::mutex> lock(init_repo_mutex_);
            if (!ptr->repo->CheckInit()) {
                ptr->repo->InitRepo(device_index);
            }
        }
        ptr->repo->Enqueue(cur_paras);
        if (ptr->repo->GetStatus() == RepoStatus::INIT) {
            ptr->repo->MakeSureQueueEmpty();
            ptr->repo->ChangeStatus(RepoStatus::INIT, RepoStatus::RUN);
        }
    } else {
        default_streams[device_index].repo->Enqueue(cur_paras);
        if (default_streams[device_index].repo->GetStatus() == RepoStatus::INIT) {
            default_streams[device_index].repo->MakeSureQueueEmpty();
            default_streams[device_index].repo->ChangeStatus(RepoStatus::INIT, RepoStatus::RUN);
        }
    }
}

void setCurrentNPUStream(NPUStream stream)
{
    initNPUStreamsOnce();
    auto ptr = NPUStream_internals(stream);
    AT_ASSERT(ptr, PTA_ERROR(ErrCode::PTR));
    if (current_streams[ptr->device_index]->stream != ptr->stream) {
        ASCEND_LOGI("Exchange NPU current stream from stream = %p to stream = %p",
            current_streams[ptr->device_index]->stream, ptr->stream);
    }

    current_streams[ptr->device_index] = ptr;
}

std::ostream& operator<<(std::ostream& stream, const NPUStream& s)
{
    return stream << s.unwrap();
}

NPUStream::NPUStream(c10::Stream stream) : stream_(stream)
{
    TORCH_CHECK(stream_.device_type() == c10::DeviceType::PrivateUse1, PTA_ERROR(ErrCode::TYPE));
}

void NPUStream::setDataPreprocessStream(bool is_data_preprocess_stream)
{
    auto ptr = NPUStream_internals(getCurrentNPUStream());
    AT_ASSERT(ptr, PTA_ERROR(ErrCode::PTR));
    ptr->is_data_preprocess_stream = is_data_preprocess_stream;
}

bool NPUStream::isDataPreprocessStream()
{
    auto ptr = NPUStream_internals(getCurrentNPUStream());
    AT_ASSERT(ptr, PTA_ERROR(ErrCode::PTR));
    return ptr->is_data_preprocess_stream;
}

bool NPUStream::getRepoStopFlag()
{
    auto ptr = NPUStream_internals(getCurrentNPUStream());
    AT_ASSERT(ptr, PTA_ERROR(ErrCode::PTR));
    return ptr->is_repo_stop;
}

bool NPUStream::isSyncLaunchStream() const
{
    auto ptr = NPUStream_internals(*this);
    AT_ASSERT(ptr, PTA_ERROR(ErrCode::PTR));
    return ptr->is_sync_launch;
}

aclrtStream NPUStream::stream(const bool need_empty) const
{
    if (!need_empty) {
        auto cur_ptr = NPUStream_internals(*this);
        AT_ASSERT(cur_ptr, PTA_ERROR(ErrCode::PTR));
        return cur_ptr->stream;
    }

    return stream();
}

void recovery_all_npu_streams(c10::DeviceIndex device_index)
{
    if (!initialize_flag[device_index]) {
        return;
    }
    NPUGuard device_guard{device_index};
    auto& default_streamsi = default_streams[device_index];
    default_streamsi.stream = nullptr;
    NPU_CHECK_ERROR(
        acl::AclrtCreateStreamWithConfig(&default_streamsi.stream, 0, (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC)));
    auto& secondary_streamsi = secondary_streams[device_index];
    secondary_streamsi.stream = nullptr;
    NPU_CHECK_ERROR(
        acl::AclrtCreateStreamWithConfig(&secondary_streamsi.stream, 0, (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC)));
    static int StreamsPerPool = GetStreamsPerPool();
    for (const auto p : c10::irange(kMaxStreamPriorities)) {
        for (auto i = decltype(StreamsPerPool){0}; i < StreamsPerPool; ++i) {
            auto& npu_streami = npu_streams[device_index][p][i];
            if (npu_streami.stream == nullptr) {
                continue;
            }
            NPU_CHECK_ERROR(acl::AclrtCreateStreamWithConfig(
                &npu_streami.stream, 0, (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC)));
        }
    }
}

static void initDeviceSyncLaunchStream(c10::DeviceIndex device_index)
{
    NPUGuard device_guard{device_index};
    for (int i = 0; i < kSyncLaunchStreamsPerPool; ++i) {
        auto& sync_streami = sync_launch_streams[device_index][i];

        sync_streami.device_index = device_index;
        sync_streami.is_sync_launch = true;

        NPU_CHECK_ERROR(
            acl::AclrtCreateStreamWithConfig(&sync_streami.stream, 0, ACL_STREAM_FAST_SYNC));
    }
}

NPUStream getNPUStreamFromSyncLaunchPool(c10::DeviceIndex device_index)
{
    // in order to init num_npus
    initNPUStreamsOnce();
    if (device_index == -1) {
        device_index = current_device();
    }
    check_npu(device_index);

    // Initializes the stream pools once
    std::call_once(
        device_sync_launch_flags[device_index], initDeviceSyncLaunchStream, device_index);

    const auto idx = get_sync_launch_stream_idx(sync_stream_counters[device_index]);
    return NPUStream_fromInternals(&sync_launch_streams[device_index][idx]);
}

bool StreamInitFlag(c10::DeviceIndex device_index)
{
    ASCEND_LOGI("Device %d, Npu StreamInitFlag Check is %d", device_index, initialize_flag[device_index]);
    return initialize_flag[device_index];
}

aclrtStream getPrevStream()
{
    auto ptr = NPUStream_internals(getDefaultNPUStream());
    AT_ASSERT(ptr, PTA_ERROR(ErrCode::PTR));
    return ptr->prev_stream;
}

void setPrevStream(aclrtStream stream)
{
    auto ptr = NPUStream_internals(getDefaultNPUStream());
    AT_ASSERT(ptr, PTA_ERROR(ErrCode::PTR));
    ptr->prev_stream = stream;
}

bool check_enqueue_need_use(aclrtStream stream)
{
    if (!enable_core_control.load(std::memory_order_relaxed)) {
        return false;
    }

    if (tls_prev_stream != stream) {
        tls_prev_stream = stream;
        return true;
    }
    return false;
}

bool check_dequeue_need_use(aclrtStream stream)
{
    if (!enable_core_control.load(std::memory_order_relaxed)) {
        return false;
    }

    aclrtStream prev_stream = getPrevStream();
    if (prev_stream != stream) {
        setPrevStream(stream);
        return true;
    }
    return false;
}

bool is_core_control_enabled()
{
    return enable_core_control.load(std::memory_order_relaxed);
}

bool IsTaskQueueEmpty()
{
    auto ptr = NPUStream_internals(getDefaultNPUStream());
    TORCH_CHECK(ptr, PTA_ERROR(ErrCode::PTR));
    return ptr->repo->IsEmptyRepo();
}
} // namespace c10_npu
