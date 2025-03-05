from functools import partial
from typing import Optional, Callable, Any

from .scheduler import ProfilerAction
from .profiler_interface import _ProfInterface
from .analysis.prof_common_func._constant import print_warn_msg

__all__ = []


class ProfActionController:
    def __init__(
        self,
        prof,
        prof_inst: _ProfInterface,
        on_trace_ready: Optional[Callable[..., Any]] = None,        
    ) -> None:
        self.prof = prof
        self.prof_inst = prof_inst
        self.action_map = self._init_action_map()
        self.on_trace_ready = on_trace_ready

    def transit_action(self, prev_action, current_action):
        action_list = self.action_map.get((prev_action, current_action))
        if action_list:
            for action in action_list:
                action()

    def _trace_ready(self):
        if self.on_trace_ready:
            self.on_trace_ready(self.prof)

    def _init_action_map(self):
        action_map = {
            (ProfilerAction.NONE, ProfilerAction.NONE): [],
            (ProfilerAction.NONE, ProfilerAction.WARMUP): [self.prof_inst.init_trace, self.prof_inst.warmup_trace],
            (ProfilerAction.NONE, ProfilerAction.RECORD): [
                self.prof_inst.init_trace, self.prof_inst.start_trace],
            (ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE): [
                self.prof_inst.init_trace, self.prof_inst.start_trace],

            (ProfilerAction.WARMUP, ProfilerAction.NONE): [
                partial(print_warn_msg, "Incorrect schedule: WARMUP followed by NONE"),
                self.prof_inst.start_trace,
                self.prof_inst.stop_trace,
                self.prof_inst.finalize_trace],
            (ProfilerAction.WARMUP, ProfilerAction.WARMUP): [],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD): [self.prof_inst.start_trace],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD_AND_SAVE): [self.prof_inst.start_trace],

            (ProfilerAction.RECORD, ProfilerAction.NONE): [
                partial(print_warn_msg, "Incorrect schedule: RECORD followed by NONE"),
                self.prof_inst.stop_trace, self.prof_inst.finalize_trace],
            (ProfilerAction.RECORD, ProfilerAction.WARMUP): [
                partial(print_warn_msg, "Incorrect schedule: RECORD followed by WARMUP"),
                self.prof_inst.stop_trace, self.prof_inst.finalize_trace],
            (ProfilerAction.RECORD, ProfilerAction.RECORD): [],
            (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE): [],

            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE): [
                self.prof_inst.stop_trace,
                self.prof_inst.finalize_trace, self._trace_ready],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.WARMUP): [
                self.prof_inst.stop_trace, self.prof_inst.finalize_trace,
                self._trace_ready, self.prof_inst.init_trace, self.prof_inst.warmup_trace],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD): [
                self.prof_inst.stop_trace, self.prof_inst.finalize_trace,
                self._trace_ready, self.prof_inst.init_trace, self.prof_inst.start_trace],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD_AND_SAVE): [
                self.prof_inst.stop_trace, self.prof_inst.finalize_trace,
                self._trace_ready, self.prof_inst.init_trace, self.prof_inst.start_trace],
            # used for exit action
            (ProfilerAction.WARMUP, None): [
                partial(print_warn_msg,
                        "Incorrect schedule: Stop profiler while current state is WARMUP "
                        "which will result in empty parsed data."),
                self.prof_inst.finalize_trace, self.prof_inst.delete_prof_dir],
            (ProfilerAction.RECORD, None): [
                partial(print_warn_msg,
                        "Incorrect schedule: Stop profiler while current state is RECORD "
                        "which may result in incomplete parsed data."),
                self.prof_inst.stop_trace, self.prof_inst.finalize_trace, self._trace_ready],
            (ProfilerAction.RECORD_AND_SAVE, None): [
                partial(print_warn_msg,
                        "Stop profiler while current state is RECORD_AND_SAVE, "
                        "perhaps the scheduling cycle has not yet completed."),
                self.prof_inst.stop_trace, self.prof_inst.finalize_trace, self._trace_ready],
        }
        return action_map
