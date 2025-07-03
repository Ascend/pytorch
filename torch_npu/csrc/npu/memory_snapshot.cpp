#include <ATen/Context.h>
#include <torch/csrc/profiler/combined_traceback.h>
#include <torch/csrc/jit/serialization/pickler.h>

#include "torch_npu/csrc/utils/LazyInit.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
#include "torch_npu/csrc/profiler/combined_traceback.h"
#include "torch_npu/csrc/npu/memory_snapshot.h"

using torch::jit::Pickler;
using c10_npu::NPUCachingAllocator::BlockInfo;
using c10_npu::NPUCachingAllocator::SegmentInfo;

namespace torch_npu {

std::shared_ptr<c10::GatheredContext> gather()
{
#if defined(__x86_64__)
    return torch::CapturedTraceback::gather(true, true, false);
#else
    return torch_npu::CapturedTraceback::gather(true, true, false);
#endif
}

std::shared_ptr<c10::GatheredContext> gather_with_cpp()
{
#if defined(__x86_64__)
    return torch::CapturedTraceback::gather(true, true, true);
#else
    return torch_npu::CapturedTraceback::gather(true, true, true);
#endif
}

static void checkOptionIn(const std::string& option,
                          std::initializer_list<std::string> valid,
                          const char* error)
{
    TORCH_CHECK(valid.end() != std::find(valid.begin(), valid.end(), option),
                error, PTA_ERROR(ErrCode::NOT_FOUND));
}

void _record_memory_history(c10::optional<std::string> enabled,
                            c10::optional<std::string> context,
                            std::string stacks, size_t max_entries)
{
    if (enabled) {
        checkOptionIn(*enabled, {"state", "all"},
                      "expected state to be 'state', 'all', or None");
    }
    if (context) {
        checkOptionIn(
            *context, {"state", "alloc", "all"},
            "expected context to be 'state', 'alloc', 'all', or None");
    }
    checkOptionIn(stacks, {"python", "all"},
                  "expected stacks to be 'python', or 'all'");

    c10_npu::NPUCachingAllocator::CreateContextFn recorder = gather;
    if (enabled && stacks == "all") {
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
    c10_npu::NPUCachingAllocator::recordHistory(enabled.has_value(), recorder,
                                                max_entries, when);
    c10_npu::NPUWorkspaceAllocator::recordHistory(enabled.has_value(), recorder, when);
}

std::string write_pickle(const c10::IValue& v)
{
    std::vector<char> result;
    {
        auto writer = [&](const char* data, size_t size) {
            result.insert(result.end(), data, data + size);
        };
        Pickler pickler(writer, nullptr, nullptr, nullptr, nullptr, false);
        pickler.protocol();
        pickler.pushIValue(v);
        pickler.stop();
    }
    return std::string(result.begin(), result.end());
}

c10::Dict<c10::IValue, c10::IValue> new_dict()
{
    return c10::Dict<c10::IValue, c10::IValue>(c10::AnyType::get(),
                                               c10::AnyType::get());
}

c10::List<c10::IValue> new_list()
{
    return c10::List<c10::IValue>(c10::AnyType::get());
}

std::vector<c10::IValue> ivalue_symbolize(std::vector<torch::CapturedTraceback*>& to_symbolize)
{
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

torch::CapturedTraceback* getFromContext(const std::shared_ptr<c10::GatheredContext>& x)
{
    if (torch::CapturedTraceback* sc =
            dynamic_cast<torch::CapturedTraceback*>(x.get())) {
        return sc;
    }
    TORCH_CHECK(
        false,
        "attempting to gather stack context from the wrong StackContext type.", PTA_ERROR(ErrCode::NOT_FOUND));
}

std::string _memory_snapshot_pickled()
{
    c10::IValue device_s = "device";
    c10::IValue address_s = "address";
    c10::IValue total_size_s = "total_size";
    c10::IValue allocated_size_s = "allocated_size";
    c10::IValue active_size_s = "active_size";
    c10::IValue requested_size_s = "requested_size";
    c10::IValue stream_s = "stream";
    c10::IValue segment_type_s = "segment_type";
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

    auto empty_frames = new_list();

    std::vector<torch::CapturedTraceback*> frame_tracebacks;
    std::vector<c10::Dict<c10::IValue, c10::IValue> > frame_dict;

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
        segmentDict.insert(segment_type_s,
                           (segmentInfo.is_large ? large_s : small_s));
        segmentDict.insert(is_expandable_s, segmentInfo.is_expandable);

        add_frame_key(segmentDict, segmentInfo.context_when_allocated);

        auto address = segmentInfo.address;
        auto blocks = new_list();
        for (const auto& blockInfo : segmentInfo.blocks) {
            auto blockDict = new_dict();
            blockDict.insert(address_s, address);
            blockDict.insert(size_s, blockInfo.size);
            blockDict.insert(requested_size_s, blockInfo.requested_size);
            blockDict.insert(state_s,
                             (blockInfo.allocated
                                  ? active_allocated_s
                                  : (blockInfo.active ? active_pending_free_s
                                                      : inactive_s)));
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

    using namespace c10_npu::NPUCachingAllocator;

    auto action_to_str = [&](TraceEntry::Action action) {
        switch (action) {
            case TraceEntry::ALLOC:
                return alloc_s;
            case TraceEntry::FREE_REQUESTED:
                return free_requested_s;
            case TraceEntry::FREE_COMPLETED:
                return free_completed_s;
            case TraceEntry::SEGMENT_ALLOC:
                return segment_alloc_s;
            case TraceEntry::SEGMENT_FREE:
                return segment_free_s;
            case TraceEntry::OOM:
                return oom_s;
            case TraceEntry::SNAPSHOT:
                return snapshot_s;
            case TraceEntry::SEGMENT_UNMAP:
                return segment_unmap_s;
            case TraceEntry::SEGMENT_MAP:
                return segment_map_s;
            default:
                AT_ERROR("invalid TraceEntry action");
        }
        throw std::runtime_error("unreachable");
    };

    for (const auto& traceInfo : snapshot.device_traces) {
        auto trace = new_list();
        for (const auto& te : traceInfo) {
            auto trace_entry = new_dict();
            trace_entry.insert(action_s, action_to_str(te.action_));
            trace_entry.insert(te.action_ == TraceEntry::OOM ? device_free_s
                                                             : addr_s,
                               te.addr_);
            trace_entry.insert(size_s, (int64_t)te.size_);
            trace_entry.insert(stream_s, int64_t(te.stream_));
            if (te.context_) {
                auto sc = getFromContext(te.context_);
                frame_tracebacks.push_back(sc);
                frame_dict.push_back(trace_entry);
            }
            trace.push_back(trace_entry);
        }
        traces.push_back(trace);
    }

    auto result = new_dict();
    result.insert("segments", segments);
    result.insert("device_traces", traces);

    auto frames = ivalue_symbolize(frame_tracebacks);
    for (auto i : c10::irange(frames.size())) {
        frame_dict.at(i).insert(frames_s, frames.at(i));
    }

    return write_pickle(result);
}

} // namespace torch_npu
