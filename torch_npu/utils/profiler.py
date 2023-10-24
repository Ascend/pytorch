import sys
import os
import time
import torch
import torch_npu


def Singleton(cls):
    _instances = {}

    def _singleton(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]
    return _singleton


@Singleton
class Profile(object):
    def __init__(self,
                 start_step: int = 10,
                 save_path: str = "./npu_profiling",
                 profile_type: str = None,
                 use_npu=True,
                 use_e2e_profiler=True,
                 **kwargs):
        r"""Dump TORCH/CANN/GE profiling data

        Args:
            start_step:    The step to dump profiling data
            save_path:      Save path of data
            profile_type:   The type of profile, Optional['TORCH', 'CANN', 'GE']
            use_npu:           Whether to output npu data, when torch profiling, Optional[True, False]
            use_e2e_profiler:  Whether to use e2e profile, when cann/ge profiling, Optional[True, False]

        note:
            When dumping GE profiling data, the configuration of 'GE_PROFILING_TO_STD_OUT'
            needs to be before model.to(device).

        Example:
            >>> model = Vgg16()
            >>> os.environ['GE_PROFILING_TO_STD_OUT'] = '1'
            >>> model.to(device)
        """
        self.step_count = 0
        self.save_path = save_path
        self.profile_type = profile_type
        self.enable = self.profile_type in ['TORCH', 'CANN', 'GE']
        self.start_step = start_step
        self.use_npu = use_npu
        self.use_e2e_profiler = use_e2e_profiler
        self.count = 0

        if self.enable:
            torch_args_set = set(["enabled", "use_cuda", "record_shapes", "with_flops", "profile_memory",
                                  "with_stack", "use_kineto", "use_cpu", "use_npu_simple"])
            cann_ge_args_set = set(["config"])
            if self.profile_type == "TORCH":
                if not set(kwargs.keys()).issubset(torch_args_set):
                    raise ValueError("Args '%s' invaild, expect args '%s' ." % (kwargs.keys(), torch_args_set))
                self.prof = torch.autograd.profiler.profile(use_npu=self.use_npu, **kwargs)
            elif self.profile_type == "CANN":
                if not set(kwargs.keys()).issubset(cann_ge_args_set):
                    raise ValueError("Args '%s' invaild, expect args '%s' ." % (kwargs.keys(), cann_ge_args_set))
                self.prof = torch.npu.profile(self.save_path, self.use_e2e_profiler, **kwargs)
            elif self.profile_type == "GE":
                if not set(kwargs.keys()).issubset(cann_ge_args_set):
                    raise ValueError("Args '%s' invaild, expect args '%s' ." % (kwargs.keys(), cann_ge_args_set))
                self.prof = torch.npu.profile(self.save_path, self.use_e2e_profiler, **kwargs)

            try:
                os.makedirs(self.save_path, exist_ok=True)
            except Exception as e:
                raise ValueError("the path of '%s' is invaild." % (self.save_path)) from e

    def __del__(self):
        if self.count != 0:
            self.end()

    def start(self):
        if self.count != 0:
            raise ValueError("start interface can only be called once")
        self.count = self.count + 1

        if self.enable:
            self.step_count = self.step_count + 1
            if self.step_count == self.start_step:
                self.prof.__enter__()

    def end(self):
        if self.count == 0:
            raise ValueError("start interface must be called after end")
        self.count = 0

        if self.enable and self.step_count == self.start_step:
            self.prof.__exit__(None, None, None)
            if self.profile_type == "TORCH":
                filename = "torch_prof_" + time.strftime("%Y%m%d%H%M%S") + ".json"
                self.prof.export_chrome_trace(os.path.join(self.save_path, filename))
            sys.exit()
