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
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif

namespace c10_npu {
namespace {
struct LeakyStreamInternals {
  LeakyStreamInternals() {
    repo = ::std::make_unique<Repository>();
  }
  C10_DISABLE_COPY_AND_ASSIGN(LeakyStreamInternals);

  ~LeakyStreamInternals() {
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
};

// Global stream state and constants
static c10::DeviceIndex num_npus = -1;
static constexpr int kStreamsPerPoolBits = 3;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
// static constexpr unsigned int kDefaultFlags = npuStreamNonBlocking;

// Default streams
static bool initialize_flag[C10_COMPILE_TIME_MAX_NPUS] = {false};
std::mutex mtx[C10_COMPILE_TIME_MAX_NPUS];
static LeakyStreamInternals default_streams[C10_COMPILE_TIME_MAX_NPUS];

// In a specific scenario, the two operators have no value dependence
// and different execution hardware, so they can be executed in parallel
// on the default stream and the secondary stream respectively.
static LeakyStreamInternals secondary_streams[C10_COMPILE_TIME_MAX_NPUS];

static std::once_flag device_flags[C10_COMPILE_TIME_MAX_NPUS];
static std::atomic<uint32_t> npu_counters[C10_COMPILE_TIME_MAX_NPUS];

static std::array<LeakyStreamInternals, kStreamsPerPool>
    npu_streams[C10_COMPILE_TIME_MAX_NPUS];

enum class StreamIdType : uint8_t {
  DEFAULT = 0x0,
  HCCL = 0x1,
  SECONDARY = 0x2,
};

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
    switch (s) {
        case StreamIdType::DEFAULT:
            stream << "DEFAULT";
            break;
        case StreamIdType::HCCL:
            stream << "HCCL";
            break;
        case StreamIdType::SECONDARY:
            stream << "SECONDARY";
            break;
        default:
            stream << static_cast<uint8_t>(s);
            break;
    }
    return stream;
}

static inline StreamIdType streamIdType(c10::StreamId s) {
  return static_cast<StreamIdType>((uint32_t)s >> kStreamsPerPoolBits);
}

static inline size_t streamIdIndex(c10::StreamId s) {
  return static_cast<size_t>((uint32_t)s & ((1 << kStreamsPerPoolBits) - 1));
}

c10::StreamId makeStreamId(StreamIdType st, size_t si) {
  return static_cast<c10::StreamId>((static_cast<size_t>(st) << kStreamsPerPoolBits) | si);
}

template <typename T, typename A>
static bool pointer_within(const T* ptr, const A& arr) {
  return std::greater_equal<const T*>()(ptr, arr.data()) &&
      std::less<const T*>()(ptr, arr.data() + arr.size());
}

static c10::StreamId NPUStream_getStreamId(const LeakyStreamInternals* ptr) {
    c10::DeviceIndex device_index = ptr->device_index;
    if (ptr == &default_streams[device_index]) {
        return makeStreamId(StreamIdType::DEFAULT, 0);
    }
    if (pointer_within<LeakyStreamInternals>(ptr, npu_streams[device_index])) {
        return makeStreamId(
            StreamIdType::HCCL, ptr - npu_streams[device_index].data());
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

static thread_local std::unique_ptr<LeakyStreamInternals* []> current_streams = nullptr;

static void initGlobalStreamState() {
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
    npu_counters[device_id] = 0;
    auto& default_streamsi = default_streams[device_id];
    NPU_CHECK_SUPPORTED_OR_ERROR(
        acl::AclrtCreateStreamWithConfig(&default_streamsi.stream, 0, (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC)));
    if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
        default_streamsi.repo->InitRepo(device_id);
    }
    // Initializes secondary streams
    secondary_streams[device_id].device_index = device_id;
    auto &secondary_streamsi = secondary_streams[device_id];
    NPU_CHECK_SUPPORTED_OR_ERROR(
        acl::AclrtCreateStreamWithConfig(&secondary_streamsi.stream, 0, (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC)));
}

static void initDeviceStreamState(c10::DeviceIndex device_index) {
  // Switches to the requested device so streams are properly associated
  // with it.
  NPUGuard device_guard{device_index};
  for (auto i = decltype(kStreamsPerPool){0}; i < kStreamsPerPool; ++i) {
    auto& npu_streami = npu_streams[device_index][i];

    npu_streami.device_index = device_index;

    NPU_CHECK_SUPPORTED_OR_ERROR(
        acl::AclrtCreateStreamWithConfig(&npu_streami.stream, 0, (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC)));
  }
}

static void initNPUStreamsOnce() {
    // Inits default and secondary streams (once, globally)
    c10::DeviceIndex device_index = current_device();
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

static inline void check_npu(c10::DeviceIndex device_index) {
    AT_ASSERT(device_index >= 0 && device_index < num_npus, "Invalid device_index : ", device_index,
              ", valid device_index range is [0, ", num_npus, ")", PTA_ERROR(ErrCode::VALUE));
}

static uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

LeakyStreamInternals* NPUStream_internals(NPUStream s) {
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
        case StreamIdType::HCCL:
            return &npu_streams[device_index][si];
        case StreamIdType::SECONDARY:
            return &secondary_streams[device_index];
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

NPUStream NPUStream_fromInternals(const LeakyStreamInternals* ptr) {
  return NPUStream(
      NPUStream::UNCHECKED,
      c10::Stream(
          c10::Stream::UNSAFE,
          c10::Device(c10::DeviceType::PrivateUse1, ptr->device_index),
          NPUStream_getStreamId(ptr)));
}
} // namespace

 aclrtStream NPUStream::stream() const {
    auto ptr = NPUStream_internals(getDefaultNPUStream());
    AT_ASSERT(ptr, PTA_ERROR(ErrCode::PTR));
    if (ptr->repo->CheckInit()) {
        NPUStatus ret = ptr->repo->MakeSureQueueEmpty();
        if (ret != SUCCESS) {
          ASCEND_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
          return nullptr;
        }
    }
    auto cur_ptr = NPUStream_internals(*this);
    AT_ASSERT(cur_ptr, PTA_ERROR(ErrCode::PTR));
    return cur_ptr->stream;
}

NPUStream getNPUStreamFromPool(c10::DeviceIndex device_index) {
  initNPUStreamsOnce();
  if (device_index == -1)
    device_index = current_device();
  check_npu(device_index);

  // Initializes the stream pools (once)
  std::call_once(
      device_flags[device_index], initDeviceStreamState, device_index);

  const auto idx = get_idx(npu_counters[device_index]);
  return NPUStream_fromInternals(&npu_streams[device_index][idx]);
}

NPUStream getStreamFromPool(
    const bool isHighPriority,
    c10::DeviceIndex device_index) {
  initNPUStreamsOnce();
  if (device_index == -1)
    device_index = current_device();
  check_npu(device_index);

  // Initializes the stream pools (once)
  std::call_once(
      device_flags[device_index], initDeviceStreamState, device_index);

  if (isHighPriority) {
    const auto idx = get_idx(npu_counters[device_index]);
    return NPUStream_fromInternals(&npu_streams[device_index][idx]);
  }

  const auto idx = get_idx(npu_counters[device_index]);
  return NPUStream_fromInternals(&npu_streams[device_index][idx]);
}

NPUStream getDefaultNPUStream(c10::DeviceIndex device_index) {
  initNPUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  return NPUStream_fromInternals(&default_streams[device_index]);
}

NPUStream getCurrentNPUStream(c10::DeviceIndex device_index) {
  initNPUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_npu(device_index);
  return NPUStream_fromInternals(current_streams[device_index]);
}

NPUStream getCurrentSecondaryStream(c10::DeviceIndex device_index) {
  initNPUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_npu(device_index);
  return NPUStream_fromInternals(&secondary_streams[device_index]);
}

aclrtStream getCurrentNPUStreamNoWait(c10::DeviceIndex device_index) {
  initNPUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_npu(device_index);
  LeakyStreamInternals* ptr = current_streams[device_index];
  return ptr->stream;
}

NPUStatus emptyAllNPUStream() {
  initNPUStreamsOnce();
  NPUStatus ret;
  for (auto i = decltype(num_npus){0}; i < num_npus; ++i) {
    auto& default_streamsi = default_streams[i];
    if (default_streamsi.stream == nullptr) {
      continue;
    }
    if (default_streamsi.stream != nullptr && default_streamsi.repo->CheckInit()) {
      ret = default_streamsi.repo->MakeSureQueueEmpty();
      if (ret != SUCCESS) {
        return ret;
      }
    }
  }
  return SUCCESS;
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

bool npuSynchronizeDevice(bool check_error) {
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
    NPUStatus ret = c10_npu::emptyAllNPUStream();
    if (ret != SUCCESS) {
      ASCEND_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
    }
  }
  auto acl_ret = aclrtSynchronizeDevice();
#ifndef BUILD_LIBTORCH
  if (acl_ret == ACL_ERROR_NONE) {
      const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
      if (C10_UNLIKELY(trigger)) {
          trigger->traceNpuDeviceSynchronization();
      }
  }
#endif
  if (check_error) {
    NPU_CHECK_ERROR(acl_ret, "aclrtSynchronizeDevice");
  } else {
    NPU_CHECK_WARN(acl_ret);
  }
  return acl_ret == ACL_ERROR_NONE;
}

bool npuSynchronizeUsedDevices(bool check_error) {
    if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
        NPUStatus ret = c10_npu::emptyAllNPUStream();
        if (ret != SUCCESS) {
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

void enCurrentNPUStream(
    void* cur_paras,
    c10::DeviceIndex device_index) {
  initNPUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_npu(device_index);
  c10_npu::queue::QueueParas* queueParam = static_cast<c10_npu::queue::QueueParas* >(cur_paras);
  queueParam->correlation_id = c10_npu::queue::QueueParas::g_correlation_id++;
  queueParam->paramStream = current_streams[device_index]->stream;
  default_streams[device_index].repo->Enqueue(cur_paras);
  if (default_streams[device_index].repo->GetStatus() == RepoStatus::INIT) {
    default_streams[device_index].repo->MakeSureQueueEmpty();
    default_streams[device_index].repo->ChangeStatus(RepoStatus::INIT, RepoStatus::RUN);
  }
}

void setCurrentNPUStream(NPUStream stream) {
    initNPUStreamsOnce();
    auto ptr = NPUStream_internals(stream);
    AT_ASSERT(ptr, PTA_ERROR(ErrCode::PTR));
    current_streams[ptr->device_index] = ptr;
}

std::ostream& operator<<(std::ostream& stream, const NPUStream& s) {
  return stream << s.unwrap();
}

void NPUStream::setDataPreprocessStream(bool is_data_preprocess_stream) {
    auto ptr = NPUStream_internals(getCurrentNPUStream());
    AT_ASSERT(ptr, PTA_ERROR(ErrCode::PTR));
    ptr->is_data_preprocess_stream = is_data_preprocess_stream;
}

bool NPUStream::isDataPreprocessStream() {
    auto ptr = NPUStream_internals(getCurrentNPUStream());
    AT_ASSERT(ptr, PTA_ERROR(ErrCode::PTR));
    return ptr->is_data_preprocess_stream;
}

 aclrtStream NPUStream::stream(const bool need_empty) const {
    if (!need_empty) {
        auto cur_ptr = NPUStream_internals(*this);
        AT_ASSERT(cur_ptr, PTA_ERROR(ErrCode::PTR));
        return cur_ptr->stream;
    }
    
    return stream();
}

} // namespace c10_npu
