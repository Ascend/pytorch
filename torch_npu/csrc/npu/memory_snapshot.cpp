#include <ATen/Context.h>
#include <ATen/core/CachingHostAllocator.h>
#include <ATen/record_function.h>
#include <torch/csrc/profiler/combined_traceback.h>
#include <torch/csrc/jit/serialization/pickler.h>

#include "torch_npu/csrc/utils/LazyInit.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
#include "torch_npu/csrc/core/npu/NPUAllocatorConfig.h"
#include "torch_npu/csrc/profiler/combined_traceback.h"
#include "torch_npu/csrc/npu/memory_snapshot.h"
#include "torch_npu/csrc/_compat/version.h"

using c10_npu::NPUCachingAllocator::BlockInfo;
using c10_npu::NPUCachingAllocator::SegmentInfo;
using torch::jit::Pickler;

namespace torch_npu {

namespace {

class CallbackManager {
 public:
  // Constructor
  CallbackManager() = default;
  // Destructor
  ~CallbackManager() = default;
  // Methods to get and set the callback handles
  at::CallbackHandle getAnnotationHandle() const {
    return annotationHandle_;
  }
  void setAnnotationHandle(at::CallbackHandle handle) {
    annotationHandle_ = handle;
  }
  at::CallbackHandle getCompileContextHandle() const {
    return compileContextHandle_;
  }
  void setCompileContextHandle(at::CallbackHandle handle) {
    compileContextHandle_ = handle;
  }
  std::unique_lock<std::mutex> lockCallbackMutex() const {
    return std::unique_lock<std::mutex>(callbackMutex_);
  }

 private:
  mutable std::mutex callbackMutex_;
  at::CallbackHandle annotationHandle_{0};
  at::CallbackHandle compileContextHandle_{0};
};

CallbackManager callbackManager;

at::CallbackHandle _initRecordAnnotations(bool useGlobalCallback) {
  auto addCallback = useGlobalCallback ? at::addGlobalCallback : at::addThreadLocalCallback;
  return addCallback(at::RecordFunctionCallback(
                         [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
                           c10_npu::NPUCachingAllocator::recordAnnotation({{"name", fn.name()}, {"stage", "START"}});
                           return nullptr;
                         },
                         [](const at::RecordFunction& fn, at::ObserverContext* /*ctx_ptr*/) {
                           c10_npu::NPUCachingAllocator::recordAnnotation({{"name", fn.name()}, {"stage", "END"}});
                         })
                         .scopes({at::RecordScope::USER_SCOPE}));
}

at::CallbackHandle _initCompileContexts() {
  return at::addGlobalCallback(at::RecordFunctionCallback(
                                   [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
                                     std::string functionName = fn.name();
                                     const std::string functionNamePrefix = "Torch-Compiled Region";
                                     if (functionName.starts_with(functionNamePrefix)) {
                                       c10_npu::NPUCachingAllocator::pushCompileContext(functionName);
                                     }
                                     return nullptr;
                                   },
                                   [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
                                     std::string functionName = fn.name();
                                     const std::string functionNamePrefix = "Torch-Compiled Region";
                                     if (functionName.starts_with(functionNamePrefix)) {
                                       c10_npu::NPUCachingAllocator::popCompileContext();
                                     }
                                   })
                                   .scopes({at::RecordScope::FUNCTION}));
}

void setRecordFunctionCallbacks(bool enabled, bool compileContext, bool globalRecordAnnotations) {
  // Handle Callbacks under mutex
  auto lock = callbackManager.lockCallbackMutex();
  if (enabled) {
    if (callbackManager.getAnnotationHandle() == 0) {
      callbackManager.setAnnotationHandle(_initRecordAnnotations(globalRecordAnnotations));
    }
    if (compileContext && callbackManager.getCompileContextHandle() == 0) {
      callbackManager.setCompileContextHandle(_initCompileContexts());
    }
  } else {
    if (callbackManager.getAnnotationHandle() != 0) {
      at::removeCallback(callbackManager.getAnnotationHandle());
      callbackManager.setAnnotationHandle(0);
    }
    if (callbackManager.getCompileContextHandle() != 0) {
      at::removeCallback(callbackManager.getCompileContextHandle());
      callbackManager.setCompileContextHandle(0);
    }
  }
}

} // namespace

std::shared_ptr<c10::GatheredContext> gather() {
#if defined(__x86_64__)
  return torch::CapturedTraceback::gather(true, true, false);
#else
  return torch_npu::CapturedTraceback::gather(true, true, false);
#endif
}

std::shared_ptr<c10::GatheredContext> gather_with_cpp() {
#if defined(__x86_64__)
  return torch::CapturedTraceback::gather(true, true, true);
#else
  return torch_npu::CapturedTraceback::gather(true, true, true);
#endif
}

static void checkOptionIn(const std::string& option, std::initializer_list<std::string> valid, const char* error) {
  TORCH_CHECK(valid.end() != std::find(valid.begin(), valid.end(), option), error, PTA_ERROR(ErrCode::NOT_FOUND));
}

void _record_memory_history(
    c10::optional<std::string> enabled,
    c10::optional<std::string> context,
    const std::string& stacks,
    size_t max_entries,
    bool clearHistory,
    bool compileContext,
    const std::vector<std::string>& skip_actions,
    bool globalRecordAnnotations,
    bool record_pinned_host_memory,
    bool record_npu) {
  TORCH_CHECK(
      !record_pinned_host_memory || enabled.has_value(),
      "record_pinned_host_memory requires memory history recording to be enabled");
  TORCH_CHECK(
      !record_pinned_host_memory || !c10_npu::NPUCachingAllocator::NPUAllocatorConfig::pin_memory_expandable_segments(),
      "record_pinned_host_memory is not supported with pin_memory_expandable_segments");
  if (enabled) {
    checkOptionIn(*enabled, {"state", "all"}, "expected state to be 'state', 'all', or None");
  }
  if (context) {
    checkOptionIn(*context, {"state", "alloc", "all"}, "expected context to be 'state', 'alloc', 'all', or None");
  }
  checkOptionIn(stacks, {"python", "all"}, "expected stacks to be 'python', or 'all'");
  // Validate before updating either allocator's recording configuration.
  for (const auto& action : skip_actions) {
    c10_npu::NPUCachingAllocator::parseTraceEntryAction(action);
  }

  c10_npu::NPUCachingAllocator::CreateContextFn recorder = gather;
  if (enabled && context && stacks == "all") {
    recorder = gather_with_cpp;
    // warm up C++ stack unwinding
#if defined(__x86_64__)
    torch::unwind::unwind();
#else
    torch_npu::unwind::unwind();
#endif
  }
  max_entries = (enabled && *enabled == "all") ? max_entries : 1;
  auto when = c10_npu::NPUCachingAllocator::RecordContext::NEVER;
  if (context) {
    if (context == "all") {
      when = c10_npu::NPUCachingAllocator::RecordContext::ALL;
    } else if (context == "alloc") {
      when = c10_npu::NPUCachingAllocator::RecordContext::ALLOC;
    } else if (context == "state") {
      when = c10_npu::NPUCachingAllocator::RecordContext::STATE;
    }
  }
  torch_npu::utils::npu_lazy_init();
  setRecordFunctionCallbacks(enabled.has_value(), compileContext, globalRecordAnnotations);
  bool npu_enabled = enabled.has_value() && record_npu;
  c10_npu::NPUCachingAllocator::recordHistory(npu_enabled, recorder, max_entries, when, clearHistory, skip_actions);
  c10_npu::NPUWorkspaceAllocator::recordHistory(npu_enabled, recorder, when, skip_actions);
// COMPAT(>= 2.14): Host allocator history APIs.
// CAN REMOVE guard when MIN_SUPPORTED >= (2, 14).
#if TORCH_NPU_VERSION_GE(2, 14)
  bool host_enabled = enabled.has_value() && record_pinned_host_memory;
  auto* host_allocator = at::getHostAllocator(at::kPrivateUse1);
  if (host_allocator && (host_enabled || host_allocator->is_history_enabled())) {
    host_allocator->record_history(
        host_enabled,
        recorder,
        max_entries,
        static_cast<c10::CachingDeviceAllocator::RecordContext>(when),
        clearHistory);
  }
#endif
}

std::string write_pickle(const c10::IValue& v) {
  std::vector<char> result;
  {
    auto writer = [&](const char* data, size_t size) { result.insert(result.end(), data, data + size); };
    Pickler pickler(writer, nullptr, nullptr, nullptr, nullptr, false);
    pickler.protocol();
    pickler.pushIValue(v);
    pickler.stop();
  }
  return std::string(result.begin(), result.end());
}

c10::Dict<c10::IValue, c10::IValue> new_dict() {
  return c10::Dict<c10::IValue, c10::IValue>(c10::AnyType::get(), c10::AnyType::get());
}

c10::List<c10::IValue> new_list() {
  return c10::List<c10::IValue>(c10::AnyType::get());
}

std::vector<c10::IValue> ivalue_symbolize(std::vector<torch::CapturedTraceback*>& to_symbolize) {
  // we dedup repeated to_symbolize objects to prevent
  // creating a bunch of duplicated frame objects
  std::unordered_map<torch::CapturedTraceback*, uint64_t> cached_frames;
  std::vector<torch::CapturedTraceback*> unique_frames;
  for (const auto& sc : to_symbolize) {
    auto it = cached_frames.find(sc);
    if (it == cached_frames.end()) {
      cached_frames.insert({sc, unique_frames.size()});
      unique_frames.push_back(sc);
    }
  }
  auto s = symbolize(unique_frames);

  c10::IValue line_s = "line";
  c10::IValue name_s = "name";
  c10::IValue filename_s = "filename";
  std::vector<c10::IValue> all_frames;
  for (const auto& f : s.all_frames) {
    auto d = new_dict();
    d.insert(name_s, f.funcname);
    d.insert(filename_s, f.filename);
    d.insert(line_s, int64_t(f.lineno));
    all_frames.emplace_back(std::move(d));
  }

  std::vector<c10::IValue> py_unique_frames;
  for (const auto& t : s.tracebacks) {
    auto l = new_list();
    for (const auto& e : t) {
      l.push_back(all_frames.at(e));
    }
    py_unique_frames.emplace_back(std::move(l));
  }

  std::vector<c10::IValue> result;
  for (const auto& sc : to_symbolize) {
    result.push_back(py_unique_frames.at(cached_frames.at(sc)));
  }
  return result;
}

torch::CapturedTraceback* getFromContext(const std::shared_ptr<c10::GatheredContext>& x) {
  if (torch::CapturedTraceback* sc = dynamic_cast<torch::CapturedTraceback*>(x.get())) {
    return sc;
  }
  TORCH_CHECK(
      false, "attempting to gather stack context from the wrong StackContext type.", PTA_ERROR(ErrCode::NOT_FOUND));
}

std::string _memory_snapshot_pickled() {
  c10::IValue device_s = "device";
  c10::IValue address_s = "address";
  c10::IValue total_size_s = "total_size";
  c10::IValue allocated_size_s = "allocated_size";
  c10::IValue active_size_s = "active_size";
  c10::IValue requested_size_s = "requested_size";
  c10::IValue stream_s = "stream";
  c10::IValue segment_type_s = "segment_type";
  c10::IValue segment_pool_id = "segment_pool_id";
  c10::IValue large_s = "large";
  c10::IValue small_s = "small";
  c10::IValue size_s = "size";
  c10::IValue state_s = "state";
  c10::IValue active_allocated_s = "active_allocated";
  c10::IValue active_pending_free_s = "active_pending_free";
  c10::IValue inactive_s = "inactive";
  c10::IValue addr_s = "addr";
  c10::IValue filename_s = "filename";
  c10::IValue name_s = "name";
  c10::IValue line_s = "line";
  c10::IValue frames_s = "frames";
  c10::IValue blocks_s = "blocks";
  c10::IValue is_expandable_s = "is_expandable";
  c10::IValue registration_counter_s = "registration_counter";

  auto empty_frames = new_list();

  std::vector<torch::CapturedTraceback*> frame_tracebacks;
  std::vector<c10::Dict<c10::IValue, c10::IValue>> frame_dict;

  auto add_frame_key = [&](const c10::Dict<c10::IValue, c10::IValue>& d,
                           const std::shared_ptr<c10::GatheredContext>& ctx) {
    if (ctx) {
      frame_tracebacks.push_back(getFromContext(ctx));
      frame_dict.push_back(d);
    } else {
      d.insert(frames_s, empty_frames);
    }
  };

  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    auto segmentDict = new_dict();
    segmentDict.insert(device_s, segmentInfo.device);
    segmentDict.insert(address_s, segmentInfo.address);
    segmentDict.insert(total_size_s, segmentInfo.total_size);
    segmentDict.insert(allocated_size_s, segmentInfo.allocated_size);
    segmentDict.insert(active_size_s, segmentInfo.active_size);
    segmentDict.insert(requested_size_s, segmentInfo.requested_size);
    segmentDict.insert(stream_s, int64_t(segmentInfo.stream));
    segmentDict.insert(segment_type_s, (segmentInfo.is_large ? large_s : small_s));
    segmentDict.insert(segment_pool_id, std::tuple<int64_t, int64_t>(segmentInfo.owner_private_pool_id));
    segmentDict.insert(is_expandable_s, segmentInfo.is_expandable);
    segmentDict.insert(registration_counter_s, segmentInfo.registration_counter);

    add_frame_key(segmentDict, segmentInfo.context_when_allocated);

    auto address = segmentInfo.address;
    auto blocks = new_list();
    for (const auto& blockInfo : segmentInfo.blocks) {
      auto blockDict = new_dict();
      blockDict.insert(address_s, address);
      blockDict.insert(size_s, blockInfo.size);
      blockDict.insert(requested_size_s, blockInfo.requested_size);
      blockDict.insert(
          state_s,
          (blockInfo.allocated ? active_allocated_s : (blockInfo.active ? active_pending_free_s : inactive_s)));
      add_frame_key(blockDict, blockInfo.context_when_allocated);
      address += blockInfo.size;
      blocks.push_back(blockDict);
    }
    segmentDict.insert(blocks_s, blocks);

    return segmentDict;
  };

  auto snapshot = c10_npu::NPUCachingAllocator::snapshot();

  auto segments = new_list();
  for (const auto& segmentInfo : snapshot.segments) {
    segments.push_back(segmentInfoToDict(segmentInfo));
  }

  auto traces = new_list();
  c10::IValue action_s = "action";
  c10::IValue alloc_s = "alloc";
  c10::IValue free_requested_s = "free_requested";
  c10::IValue free_completed_s = "free_completed";
  c10::IValue segment_alloc_s = "segment_alloc";
  c10::IValue segment_free_s = "segment_free";
  c10::IValue segment_map_s = "segment_map";
  c10::IValue segment_unmap_s = "segment_unmap";
  c10::IValue snapshot_s = "snapshot";
  c10::IValue oom_s = "oom";
  c10::IValue device_free_s = "device_free";
  c10::IValue pool_id_s = "pool_id";
  c10::IValue compile_context_s = "compile_context";

  using namespace c10_npu::NPUCachingAllocator;

  auto action_to_str = [&](auto action) {
    using Action = decltype(action);
    switch (action) {
      case Action::ALLOC:
        return alloc_s;
      case Action::FREE_REQUESTED:
        return free_requested_s;
      case Action::FREE_COMPLETED:
        return free_completed_s;
      case Action::SEGMENT_ALLOC:
        return segment_alloc_s;
      case Action::SEGMENT_FREE:
        return segment_free_s;
      case Action::OOM:
        return oom_s;
      case Action::SNAPSHOT:
        return snapshot_s;
      case Action::SEGMENT_UNMAP:
        return segment_unmap_s;
      case Action::SEGMENT_MAP:
        return segment_map_s;
      default:
        AT_ERROR("invalid TraceEntry action");
    }
    throw std::runtime_error("unreachable");
  };

// COMPAT(>= 2.14): Host allocator history APIs.
// CAN REMOVE guard when MIN_SUPPORTED >= (2, 14).
#if TORCH_NPU_VERSION_GE(2, 14)
  auto traceEntryToDict = [&](const auto& te) {
    auto trace_entry = new_dict();
    trace_entry.insert(action_s, action_to_str(te.action_));
    trace_entry.insert(addr_s, static_cast<int64_t>(te.addr_));
    trace_entry.insert(size_s, static_cast<int64_t>(te.size_));
    trace_entry.insert(stream_s, int64_t(te.stream_));
    trace_entry.insert(pool_id_s, std::tuple<int64_t, int64_t>(te.mempool_));
    trace_entry.insert("time_us", te.time_.t_);
    trace_entry.insert("compile_context", te.compile_context_);
    trace_entry.insert("user_metadata", te.user_metadata_);
    if (te.context_) {
      frame_tracebacks.push_back(getFromContext(te.context_));
      frame_dict.push_back(trace_entry);
    }
    return trace_entry;
  };
#endif

  for (const auto& traceInfo : snapshot.device_traces) {
    auto trace = new_list();
    for (const auto& te : traceInfo) {
      auto trace_entry = new_dict();
      trace_entry.insert(action_s, action_to_str(te.action_));
      trace_entry.insert(te.action_ == TraceEntry::OOM ? device_free_s : addr_s, te.addr_);
      trace_entry.insert(size_s, (int64_t)te.size_);
      trace_entry.insert(stream_s, int64_t(te.stream_));
      trace_entry.insert(pool_id_s, std::tuple<int64_t, int64_t>(te.mempool_));
      trace_entry.insert(compile_context_s, te.compile_context_);
      if (te.context_) {
        auto sc = getFromContext(te.context_);
        frame_tracebacks.push_back(sc);
        frame_dict.push_back(trace_entry);
      }
      trace.push_back(trace_entry);
    }
    traces.push_back(trace);
  }

  auto host_segments = new_list();
  auto host_traces = new_list();
// COMPAT(>= 2.14): Host allocator history APIs.
// CAN REMOVE guard when MIN_SUPPORTED >= (2, 14).
#if TORCH_NPU_VERSION_GE(2, 14)
  auto* host_allocator = at::getHostAllocator(at::kPrivateUse1);
  if (host_allocator && host_allocator->is_history_enabled()) {
    for (const auto& seg : host_allocator->get_segments()) {
      auto segment = new_dict();
      segment.insert(device_s, -1);
      segment.insert(address_s, static_cast<int64_t>(seg.address));
      segment.insert(total_size_s, static_cast<int64_t>(seg.size));
      segment.insert(allocated_size_s, static_cast<int64_t>(seg.allocated ? seg.size : 0));
      segment.insert(active_size_s, static_cast<int64_t>(seg.active ? seg.size : 0));
      segment.insert(requested_size_s, static_cast<int64_t>(seg.size));
      segment.insert(stream_s, int64_t(0));
      segment.insert(segment_type_s, small_s);
      segment.insert(segment_pool_id, std::tuple<int64_t, int64_t>(seg.owner_private_pool_id));
      segment.insert(is_expandable_s, false);
      add_frame_key(segment, seg.context_when_allocated);

      auto block = new_dict();
      block.insert(address_s, static_cast<int64_t>(seg.address));
      block.insert(size_s, static_cast<int64_t>(seg.size));
      block.insert(requested_size_s, static_cast<int64_t>(seg.size));
      block.insert(state_s, seg.allocated ? active_allocated_s : (seg.active ? active_pending_free_s : inactive_s));
      add_frame_key(block, seg.context_when_allocated);
      auto blocks = new_list();
      blocks.push_back(block);
      segment.insert(blocks_s, blocks);
      host_segments.push_back(segment);
    }
    for (const auto& te : host_allocator->get_traces()) {
      host_traces.push_back(traceEntryToDict(te));
    }
  }
#endif

  auto result = new_dict();
  result.insert("segments", segments);
  result.insert("device_traces", traces);
  result.insert("host_segments", host_segments);
  result.insert("host_traces", host_traces);

  auto external_annotations = new_list();
  for (const auto& ae : snapshot.external_annotations) {
    auto annotation_entry = new_dict();
    for (const auto& md : ae.metadata_) {
      annotation_entry.insert((c10::IValue)md.first, md.second);
    }
    annotation_entry.insert(device_s, ae.device_);
    annotation_entry.insert("time_us", ae.time_.t_);
    external_annotations.push_back(annotation_entry);
  }
  result.insert("external_annotations", external_annotations);

  auto frames = ivalue_symbolize(frame_tracebacks);
  for (auto i : c10::irange(frames.size())) {
    frame_dict.at(i).insert(frames_s, frames.at(i));
  }

  return write_pickle(result);
}

} // namespace torch_npu
