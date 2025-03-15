#include <c10d/reducer_timer.hpp>
#include <c10d/debug.h>

#include "torch_npu/csrc/npu/Event.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"


namespace c10d {
namespace {

const int kMilliSecondToNanosSecond = 1000000;

class NpuTimer : public c10d::Timer {
public:
    explicit NpuTimer(c10::Device dev) : device(dev) {}

    void record(Event event) override
    {
        // Parent class sets the host-side time
        Timer::record(event);
        c10_npu::NPUGuard g(device);
        getEvent(event).record();
    }

    c10::optional<int64_t> measureDifference(Event start, Event end) override
    {
        // Currently elapsed_time does not support the return of negative values.
        // So measureDifference is only calculated when the debug level is detail.
        if (debug_level() != DebugLevel::Detail) {
            return c10::nullopt;
        }

        c10_npu::NPUGuard g(device);
        c10_npu::NPUEvent& start_event = getEvent(start);
        c10_npu::NPUEvent& end_event = getEvent(end);
        // It is possible users did not call backward or run codes in
        // no-sync mode, in this case, some npuEvents like "backward_compute_end",
        // "backward_comm_start" or "backward_comm_end" will not be recorded.
        // npuEvent is created when it is first time to be recorded.
        // If it is never recorded/created, skip synchronize and calculation.
        // Otherwise it will throw npu errors.
        if (!start_event.isCreated() || !end_event.isCreated()) {
            return c10::nullopt;
        }
        // set_runtime_stats_and_log is called at the beginning of forward call,
        // when it is cheap to synchronize the npu events of previous iteration,
        // as mostly all npu operations are finished in previous iteration.
        start_event.synchronize();
        end_event.synchronize();

        float milliseconds;
        try {
            milliseconds = start_event.elapsed_time(end_event);
        } catch (std::exception &e) {
            milliseconds = -1;
        }
        // If gpu_end is not recorded in this iteration,
        // milliseconds will have invalid value.
        // For some cases like DDP runs on non-sync mode,
        // gpu_end can not be recorded in this iteration and thus can not
        // calculate the valid avg_time.
        // In this case, skip calculating the avg_time and return.
        if (milliseconds < 0) {
            return c10::nullopt;
        }
        return int64_t(milliseconds * kMilliSecondToNanosSecond);
    }

private:
    c10::Device device;

    c10_npu::NPUEvent forward_start = c10_npu::NPUEvent(ACL_EVENT_TIME_LINE);
    c10_npu::NPUEvent backward_compute_start = c10_npu::NPUEvent(ACL_EVENT_TIME_LINE);
    c10_npu::NPUEvent backward_compute_end = c10_npu::NPUEvent(ACL_EVENT_TIME_LINE);
    c10_npu::NPUEvent backward_comm_start = c10_npu::NPUEvent(ACL_EVENT_TIME_LINE);
    c10_npu::NPUEvent backward_comm_end = c10_npu::NPUEvent(ACL_EVENT_TIME_LINE);

    c10_npu::NPUEvent& getEvent(Event event)
    {
        switch (event) {
            case Event::kForwardStart:
                return forward_start;
            case Event::kBackwardComputeStart:
                return backward_compute_start;
            case Event::kBackwardComputeEnd:
                return backward_compute_end;
            case Event::kBackwardCommStart:
                return backward_comm_start;
            case Event::kBackwardCommEnd:
                return backward_comm_end;
            default:
                TORCH_INTERNAL_ASSERT(false);
        }
    }
};

C10_REGISTER_TYPED_CLASS(TimerRegistry, c10::kPrivateUse1, NpuTimer);

} // namespace
} // namespace c10d
