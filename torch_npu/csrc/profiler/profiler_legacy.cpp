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

#include "torch_npu/csrc/profiler/profiler.h"
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/jit/frontend/code_template.h>

#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

#include <fstream>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <ATen/record_function.h>
#include <c10/core/Allocator.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <iostream>

namespace torch_npu {
namespace profiler{

std::vector<std::string> callstackStr(const std::vector<FileLineFunc>& cs) {
  std::vector<std::string> cs_str;
  cs_str.reserve(cs.size());
  for (const auto& entry : cs) {
    std::stringstream loc;
    loc << entry.filename << "(" << entry.line << "): " << entry.funcname;
    cs_str.push_back(loc.str());
  }
  return cs_str;
}

// We decompose the profiler logic into the following components:
//
// ThreadLocalDebugInfo:
//
// ThreadLocalDebugInfo is a thread local mapping from slots into
// the debug information structs.
// ThreadLocalDebugInfo is automatically propagated across thread
// boundaries, including the cases of:
//  - launching async jobs with at::launch
//  - executing JIT continuations
//  - moving from the forward threads into autograd (backward) threads
//
// Entries in ThreadLocalDebugInfo are managed by DebugInfoGuard
// which can be used to add or overwrite an entry in the thread local
// mapping. A corresponding entry is removed when the guard is destroyed,
// potentially revealing the previously set value for the same slot.
//
// For the async tasks, slots previuosly set in the main thread before
// launching of an async task are shared and visible in the async task.
//
// On the other hand, any adding or overwriting of the mapping by the
// async task is not visible to the main thread and any modification
// (including removal of the entries) in the main thread is not visible
// to the async task if it happends after launching the task.
//
// We use ThreadLocalDebugInfo (slot PROFILER_STATE) to store profiler config,
// as well as a list of events that happen during profiling.
// An instance of ThreadLocalDebugInfo is created each time we enter
// profiler (i.e. enter profiling context manager/call enableConfig) and
// uniquely identifies a profiling run.
//
// We automatically propagate ThreadLocalDebugInfo into async tasks,
// as well as across JIT continuations and autograd thread, so all
// the operations that happen between profiling start and end
// (not necessarily within the same thread) are recorded.
// Unless the profiling slot is overwritten as in the case of nested
// profiling ranges (in this case events for the subrange are handled
// by the nested profiler)
//
// When we exit a profiling range (either by exiting profiling context
// manager or by calling disableProfiler), we remove the previously set
// profiling entry for the given thread local mapping, and consolidate
// events in the profiling result
//
//
// ThreadLocalState:
//
// ThreadLocalState takes a 'snapshot' of thread local variables
// using provided getters. It is used together with ThreadLocalStateGuard
// to transfer the snapshot across thread boundary and set the thread local
// values as in the parent task.
//
// Profiler uses ThreadLocalState to propagate profiler's thread local state.
// ThreadLocalState also automatically propagates profiler callbacks.
//
//
// at::RecordFunction and observers
//
// Profiler uses observers mechanism to add a pair of thread local callbacks
// that are executed on a number of predetermined ranges, including:
//  - c10/ATen ops
//  - TorchScript functions/methods
//  - user defined named ranges (see `record_function` python context manager)
//
// Profiler setups a pair of callbacks that record profiling events and save
// them into the thread local profiler struct (ThreadLocalDebugInfo,
// PROFILER_STATE slot)
//
//
// Thus, the overall logic is:
//
// enableProfiler:
//  - checks that profiler is not enabled (otherwise throws)
//  - pushes new ThreadLocalDebugInfo (slot PROFILER_STATE) as the profiler
//    config for the current thread
//  - pushes profiling callbacks for the current thread
//
// disableProfiler:
//  - pops PROFILER_STATE slot from the current ThreadLocalDebugInfo and
//    consolidates events
//  - removes profiling callbacks
//
// ThreadLocalState:
//  - propagates ThreadLocalDebugInfo across threads
//  - propagates profiler callbacks across threads
//
// Profiler callbacks:
//  - get the current profiling state (PROFILER slot in ThreadLocalDebugInfo)
//  - save profiling events into the profiling state
//

namespace {
const DeviceStubs default_stubs;
constexpr const DeviceStubs* default_stubs_addr = &default_stubs;
// Constant initialization, so it is guaranteed to be initialized before
// static initialization calls which may invoke registerCUDAMethods
inline const DeviceStubs*& device_stubs() {
  static const DeviceStubs* stubs_ = default_stubs_addr;
  return stubs_;
}
}

// Profiler state
const ProfilerConfig& ProfilerThreadLocalState::config() const {
  return config_;
}

thread_event_lists ProfilerThreadLocalState::consolidate() {
  std::lock_guard<std::mutex> g(state_mutex_);
  thread_event_lists result;
  for (auto& kv : event_lists_map_) {
    auto& list = kv.second;
    result.emplace_back(list->consolidate());
  }
  // Consolidate remote events if applicable as well.
  if (remoteProfiledEvents_) {
    result.insert(
        result.end(),
        std::make_move_iterator(remoteProfiledEvents_->begin()),
        std::make_move_iterator(remoteProfiledEvents_->end()));
  }
  return result;
}

void ProfilerThreadLocalState::mark(std::string name, bool include_device) {
  if (config_.state == ProfilerState::Disabled) {
    return;
  }
  if (config_.state == ProfilerState::NVTX) {
    device_stubs()->nvtxMarkA(name.c_str());
  } else {
    LegacyEvent evt(
        EventKind::Mark,
        at::StringView(std::move(name)),
        at::RecordFunction::currentThreadId(),
        include_device && (config_.state == ProfilerState::CUDA || config_.state == ProfilerState::NPU),
        config_.state);
    evt.setNodeId(at::RecordFunction::getDefaultNodeId());
    getEventList().record(std::move(evt));
  }
}

void ProfilerThreadLocalState::setOrAddRemoteProfiledEvents(
    std::vector<LegacyEvent>&& remoteProfiledEvents) {
  // Lock to serialize access from multiple callback threads.
  std::lock_guard<std::mutex> guard(state_mutex_);
  if (remoteProfiledEvents_) {
    (*remoteProfiledEvents_).emplace_back(remoteProfiledEvents);
  } else {
    remoteProfiledEvents_ = {std::move(remoteProfiledEvents)};
  }
}

void ProfilerThreadLocalState::pushRange(
    const at::RecordFunction& fn,
    const bool record_device,
    const char* msg,
    std::vector<std::vector<int64_t>>&& shapes) {
  if (config_.state == ProfilerState::Disabled) {
    return;
  }
  if (config_.state == ProfilerState::NVTX) {
    device_stubs()->nvtxRangePushA(getNvtxStr(
        fn.name(), msg, fn.seqNr(), shapes).c_str());
  } else {
    LegacyEvent evt(
        EventKind::PushRange,
        fn.name(),
        at::RecordFunction::currentThreadId(),
        record_device,
        config_.state,
        fn.handle(),
        std::move(shapes),
        at::RecordFunction::getDefaultNodeId());
    evt.setSequenceNr(fn.seqNr());
    evt.setFwdThreadId(fn.forwardThreadId());
    evt.setScope((uint8_t)fn.scope());
    if (config_.with_flops) {
      evt.setExtraArgs(saveExtraArgs(fn));
      evt.setFlops(computeFlops(std::string(fn.name().str()), evt.extraArgs()));
    }
    getEventList().record(std::move(evt));
  }
}

void ProfilerThreadLocalState::popRange(const at::RecordFunction& fn, const bool record_device) {
  if (config_.state == ProfilerState::Disabled) {
    return;
  }
  if (config_.state == ProfilerState::NVTX) {
    device_stubs()->nvtxRangePop();
  } else {
    // In some cases RecordFunction (and popRange) may be
    // called on a different thread than pushRange
    // As a convention, we put the async pop on the original
    // thread and save current thread id in pop event
    LegacyEvent evt(
        EventKind::PopRange,
        at::StringView(""),
        at::RecordFunction::currentThreadId(),
        record_device,
        config_.state,
        fn.handle());
    evt.setNodeId(at::RecordFunction::getDefaultNodeId());
    getEventList(fn.threadId()).record(std::move(evt));
  }
}

void ProfilerThreadLocalState::reportMemoryUsage(
    void* /* unused */,
    int64_t alloc_size,
    c10::Device device) {
  if (config_.profile_memory && config_.state != ProfilerState::Disabled) {
    uint64_t thread_id = at::RecordFunction::currentThreadId();
    LegacyEvent evt(
        EventKind::MemoryAlloc,
        at::StringView(""),
        thread_id,
        config_.state == ProfilerState::CUDA || config_.state == ProfilerState::NPU,
        config_.state);
    evt.updateMemoryStats(alloc_size, device);
    getEventList(thread_id).record(std::move(evt));
  }
}

bool ProfilerThreadLocalState::memoryProfilingEnabled() const {
  return config_.profile_memory;
}

std::string ProfilerThreadLocalState::getNvtxStr(
    const at::StringView& name,
    const char* msg,
    int64_t sequence_nr,
    const std::vector<std::vector<int64_t>>& shapes) const {
    return name.str();
}

RangeEventList& ProfilerThreadLocalState::getEventList(int64_t thread_id) {
  if (thread_id < 0) {
    thread_id = at::RecordFunction::currentThreadId();
  }
  RangeEventList* list_ptr = nullptr;
  std::lock_guard<std::mutex> guard(state_mutex_);
  auto it = event_lists_map_.find(thread_id);
  if (it != event_lists_map_.end()) {
    list_ptr = it->second.get();
  } else {
    auto event_list = std::make_shared<RangeEventList>();
    event_lists_map_[thread_id] = event_list;
    list_ptr = event_list.get();
  }
  return *list_ptr;
}

std::vector<std::vector<int64_t>> inputSizes(const at::RecordFunction& fn) {
  std::vector<std::vector<int64_t>> sizes;
  sizes.reserve(fn.inputs().size());
  for (const c10::IValue& input : fn.inputs()) {
    if (!input.isTensor()) {
      sizes.emplace_back();
      continue;
    }
    const at::Tensor& tensor = input.toTensor();
    if (tensor.defined()) {
      sizes.push_back(input.toTensor().sizes().vec());
    } else {
      sizes.emplace_back();
    }
  }
  return sizes;
}

namespace {

enum EventIValueIdx {
  KIND = 0,
  NAME,
  THREAD_ID,
  HANDLE,
  NODE_ID,
  CPU_MEM_USAGE,
  CPU_NS,
  CUDA_RECORDED,
  CUDA_MEM_USAGE,
  CUDA_DEVICE,
  CUDA_US,
  SHAPES,
  NUM_EVENT_IVALUE_IDX // must be last in list
};

enum ProfilerIValueIdx {
  STATE = 0,
  REPORT_INPUT_SHAPES,
  PROFILE_MEMORY,
  NUM_PROFILER_CFG_IVALUE_IDX // must be last in list
};

const std::unordered_set<std::string> disable_cuda_profiling = {
  "aten::view",
  "aten::t",
  "aten::transpose",
  "aten::stride",
  "aten::empty",
  "aten::empty_like",
  "aten::empty_strided",
  "aten::as_strided",
  "aten::expand",
  "aten::resize_",
  "aten::squeeze",
  "aten::unsqueeze",
  "aten::slice",
  "aten::_unsafe_view",
  "aten::size"
};

ProfilerThreadLocalState* getProfilerTLSState() {
  return static_cast<ProfilerThreadLocalState*>(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE));
}

void pushProfilingCallbacksLegacy() {
  auto state_ptr = getProfilerTLSState();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  auto handle = at::addThreadLocalCallback(at::RecordFunctionCallback(
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        auto state_ptr = getProfilerTLSState();
        if (!state_ptr || state_ptr->config().state == ProfilerState::Disabled) {
          return nullptr;
        }
        bool record_cuda =
            state_ptr->config().state == ProfilerState::CUDA;
        bool record_npu =
            state_ptr->config().state == ProfilerState::NPU;
        if (record_cuda && disable_cuda_profiling.find(fn.name().str()) != disable_cuda_profiling.end()) {
          record_cuda = false;
        }

        auto* msg = (fn.seqNr() >= 0) ? ", seq = " : "";
        if (state_ptr->config().report_input_shapes) {
          auto sizes = inputSizes(fn);
          state_ptr->pushRange(fn, record_cuda || record_npu, msg, std::move(sizes));
        } else {
          state_ptr->pushRange(fn, record_cuda || record_npu, msg);
        }

        return nullptr;
      },
      [](const at::RecordFunction& fn, at::ObserverContext*) {
        auto state_ptr = getProfilerTLSState();
        if (!state_ptr || state_ptr->config().state == ProfilerState::Disabled) {
          return;
        }
        bool record_cuda =
            state_ptr->config().state == ProfilerState::CUDA;
        bool record_npu =
            state_ptr->config().state == ProfilerState::NPU;
        if (record_cuda && disable_cuda_profiling.find(fn.name().str()) != disable_cuda_profiling.end()) {
          record_cuda = false;
        }
        state_ptr->popRange(fn, record_cuda || record_npu);
      })
    .needsInputs(state_ptr->config().report_input_shapes)
    .needsIds(true));
  state_ptr->setCallbackHandle(handle);
}

const int kCUDAWarmupStart = 5;
const int kNPUWarmupStart = 5;

} // namespace

void registerDeviceMethods(DeviceStubs* stubs) {
  device_stubs() = stubs;
}

at::IValue ProfilerConfig::toIValue() const {
  c10::impl::GenericList eventIValueList(at::AnyType::get());
  eventIValueList.reserve(NUM_PROFILER_CFG_IVALUE_IDX);
  eventIValueList.emplace_back(static_cast<int64_t>(state));
  eventIValueList.emplace_back(report_input_shapes);
  eventIValueList.emplace_back(profile_memory);
  return eventIValueList;
}

ProfilerConfig ProfilerConfig::fromIValue(
    const at::IValue& profilerConfigIValue) {
  TORCH_INTERNAL_ASSERT(
      profilerConfigIValue.isList(),
      "Expected IValue to contain type c10::impl::GenericList");
  auto ivalues = profilerConfigIValue.toList();
  TORCH_INTERNAL_ASSERT(
      ivalues.size() == NUM_PROFILER_CFG_IVALUE_IDX,
      c10::str(
          "Expected exactly ",
          NUM_PROFILER_CFG_IVALUE_IDX,
          " ivalues to resconstruct ProfilerConfig."));
  return ProfilerConfig(
      static_cast<ProfilerState>(ivalues.get(ProfilerIValueIdx::STATE).toInt()),
      ivalues.get(ProfilerIValueIdx::REPORT_INPUT_SHAPES).toBool(),
      ivalues.get(ProfilerIValueIdx::PROFILE_MEMORY).toBool());
}

ProfilerConfig getProfilerConfig() {
  auto state_ptr = getProfilerTLSState();
  TORCH_CHECK(
      state_ptr,
      "Tried to access profiler config, but profiler is not enabled!");
  return state_ptr->config();
}

bool profilerEnabled() {
  auto state_ptr = getProfilerTLSState();
  return state_ptr && state_ptr->config().state != ProfilerState::Disabled;
}
