// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
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

#include "NPUStream.h"
#include <c10/npu/NPUFunctions.h>
#include <c10/npu/NPUGuard.h>
#include <c10/npu/NPUQueue.h>
#include <c10/npu/OptionsManager.h>
#include <c10/npu/interface/AsyncTaskQueueInterface.h>
#include <c10/util/Exception.h>

#include <Python.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

#include <sys/time.h>
#include <unistd.h>
#include <iostream>

namespace c10 {
namespace npu {

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

  DeviceIndex device_index = -1;
  int32_t stream_id = -1;
  aclrtStream stream = nullptr;
  ::std::unique_ptr<NPUQueueBase> repo = nullptr;
};

// Global stream state and constants
static DeviceIndex num_npus = -1;
static constexpr int kStreamsPerPoolBits = 3;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
// static constexpr unsigned int kDefaultFlags = npuStreamNonBlocking;

// Default streams
static std::once_flag init_flag;
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

static inline StreamIdType streamIdType(StreamId s) {
  return static_cast<StreamIdType>((uint32_t)s >> kStreamsPerPoolBits);
}

static inline size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>((uint32_t)s & ((1 << kStreamsPerPoolBits) - 1));
}

StreamId makeStreamId(StreamIdType st, size_t si) {
  return ((uint32_t)static_cast<StreamId>(st) << kStreamsPerPoolBits) |
      static_cast<StreamId>(si);
}

template <typename T, typename A>
static bool pointer_within(const T* ptr, const A& arr) {
  return std::greater_equal<const T*>()(ptr, arr.data()) &&
      std::less<const T*>()(ptr, arr.data() + arr.size());
}

static StreamId NPUStream_getStreamId(const LeakyStreamInternals* ptr) {
  DeviceIndex device_index = ptr->device_index;
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
      device_index,
      " (something has gone horribly wrong!)");
}

static thread_local LeakyStreamInternals** current_streams = nullptr;

static void initGlobalStreamState() {
  // TODO device_count(), set to 1 temporarily.
  num_npus = c10::npu::device_count();
  // Check if the number of GPUs matches the expected compile-time max number
  // of GPUs.
  AT_ASSERTM(
      num_npus <= C10_COMPILE_TIME_MAX_NPUS,
      "Number of NPU devices on the machine is larger than the compiled "
      "max number of npus expected (",
      C10_COMPILE_TIME_MAX_NPUS,
      "). Increase that and recompile.");

  int device_id = 0;
  auto ret = aclrtGetDevice(&device_id);
  if (ret != ACL_ERROR_NONE) {
    NPU_LOGE("Device has not been set");
  }
  // Initializes default streams
  default_streams[device_id].device_index = device_id;
  npu_counters[device_id] = 0;
  auto& default_streamsi = default_streams[device_id];
  C10_NPU_CHECK(aclrtCreateStream(&default_streamsi.stream));
  if (OptionsManager::CheckQueueEnable()) {
    default_streamsi.repo->InitRepo(device_id);
  }
  // Initializes secondary streams
  secondary_streams[device_id].device_index = device_id;
  auto& secondary_streamsi = secondary_streams[device_id];
  C10_NPU_CHECK(aclrtCreateStream(&secondary_streamsi.stream));
}

static void initDeviceStreamState(DeviceIndex device_index) {
  // Switches to the requested device so streams are properly associated
  // with it.
  NPUGuard device_guard{device_index};
  for (auto i = decltype(kStreamsPerPool){0}; i < kStreamsPerPool; ++i) {
    auto& npu_streami = npu_streams[device_index][i];

    npu_streami.device_index = device_index;

    C10_NPU_CHECK(aclrtCreateStream(&npu_streami.stream));
  }
}

static void initNPUStreamsOnce() {
  // Inits default and secondary streams (once, globally)
  std::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to default streams
  current_streams =
      (LeakyStreamInternals**)malloc(num_npus * sizeof(LeakyStreamInternals*));
  if (current_streams == NULL){
    NPU_LOGE("current_streams malloc failed.");
    return;
  }
  for (auto i = decltype(num_npus){0}; i < num_npus; ++i) {
    current_streams[i] = &default_streams[i];
  }
}

static inline void check_npu(DeviceIndex device_index) {
  AT_ASSERT(device_index >= 0 && device_index < num_npus);
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
          " official API like c10::npu::getStreamFromPool() to get a new stream.");
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
          ")");
  }
}

NPUStream NPUStream_fromInternals(const LeakyStreamInternals* ptr) {
  return NPUStream(
      NPUStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::NPU, ptr->device_index),
          NPUStream_getStreamId(ptr)));
}
} // namespace

C10_API aclrtStream NPUStream::stream() const {
  auto ptr = NPUStream_internals(getDefaultNPUStream());
  AT_ASSERT(ptr);
  if (ptr->repo->CheckInit()) {
    NPUStatus ret = ptr->repo->MakeSureQueueEmpty();
    if (ret != SUCCESS) {
      NPU_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
      return nullptr;
    }
  }
  return ptr->stream;
}

NPUStream getNPUStreamFromPool(DeviceIndex device_index) {
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
    DeviceIndex device_index) {
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

NPUStream getDefaultNPUStream(DeviceIndex device_index) {
  initNPUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  return NPUStream_fromInternals(&default_streams[device_index]);
}

NPUStream getCurrentNPUStream(DeviceIndex device_index) {
  initNPUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_npu(device_index);
  return NPUStream_fromInternals(current_streams[device_index]);
}

NPUStream getCurrentSecondaryStream(DeviceIndex device_index) {
  initNPUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_npu(device_index);
  return NPUStream_fromInternals(&secondary_streams[device_index]);
}

aclrtStream getCurrentNPUStreamNoWait(DeviceIndex device_index) {
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
    NPUGuard device_guard{i};
    if (default_streamsi.stream != nullptr && default_streamsi.repo->CheckInit()) {
      ret = default_streamsi.repo->MakeSureQueueEmpty();
      if (ret != SUCCESS) {
        return ret;
      }
    }
  }
  return SUCCESS;
}

void npuSynchronizeDevice() {
  if (OptionsManager::CheckQueueEnable()) {
    NPUStatus ret = c10::npu::emptyAllNPUStream();
    if (ret != SUCCESS) {
      NPU_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
      return;
    }
  }
  C10_NPU_CHECK(aclrtSynchronizeDevice());
}

void enCurrentNPUStream(
    void* cur_paras,
    SmallVector<Storage, N>& needClearVec,
    DeviceIndex device_index) {
  initNPUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_npu(device_index);
  c10::npu::queue::QueueParas* queueParam = static_cast<c10::npu::queue::QueueParas* >(cur_paras);
  queueParam->paramStream = current_streams[device_index]->stream;
  default_streams[device_index].repo->Enqueue(cur_paras, needClearVec);
  if (default_streams[device_index].repo->GetStatus() == RepoStatus::INIT) {
    default_streams[device_index].repo->MakeSureQueueEmpty();
    default_streams[device_index].repo->ChangeStatus(RepoStatus::INIT, RepoStatus::RUN);
  }
}

void setCurrentNPUStream(NPUStream stream) {
  initNPUStreamsOnce();
  auto ptr = NPUStream_internals(stream);
  AT_ASSERT(ptr);
  current_streams[ptr->device_index] = ptr;
}

std::ostream& operator<<(std::ostream& stream, const NPUStream& s) {
  return stream << s.unwrap();
}

} // namespace npu
} // namespace c10
