import torch


class CUDAProfiler:
    def __init__(self, enable=False, warmup=10, active=20, with_stack=False, with_memory=False, record_shapes=False, save_path="./profile"):
        self.enable = enable

        self.sp_with_stack = with_stack
        self.sp_with_memory = with_memory
        self.sp_record_shapes = record_shapes
        self.warmup = warmup
        self.active = active
        self.sp_save_path = save_path

        activites = [torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU]

        self.prof = torch.profiler.profile(
            with_stack=self.sp_with_stack,
            record_shapes=self.sp_record_shapes,
            profile_memory=self.sp_with_memory,
            activities=activites,
            schedule=torch.profiler.schedule(wait=0, warmup=self.warmup, active=self.active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.sp_save_path))

    def start(self):
        if self.enable:
            self.prof.start()

    def step(self):
        if self.enable:
            self.prof.step()

    def stop(self):
        if self.enable:
            self.prof.stop()
try:
    import torch_npu

    class NPUProfiler:
        def __init__(self, enable=False, warmup=10, active=20, level="level0", with_stack=False, with_memory=False, record_shapes=False, save_path="./profile"):
            self.enable = enable

            self.sp_level = level
            self.sp_with_stack = with_stack
            self.sp_with_memory = with_memory
            self.sp_record_shapes = record_shapes
            self.warmup = warmup
            self.active = active
            self.sp_save_path = save_path

            if self.sp_level == 'level0':
                profiler_level = torch_npu.profiler.ProfilerLevel.Level0
            elif self.sp_level == 'level1':
                profiler_level = torch_npu.profiler.ProfilerLevel.Level1
            elif self.sp_level == 'level2':
                profiler_level = torch_npu.profiler.ProfilerLevel.Level2
            else:
                raise ValueError(f"profiler_level only supports level0,"
                                f" 1, and 2, but gets {self.sp_level}")

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[
                    torch_npu.profiler.ExportType.Text,
                    torch_npu.profiler.ExportType.Db
                    ],
                profiler_level=profiler_level,
                msprof_tx=False,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                l2_cache=False,
                op_attr=False,
                data_simplification=False,
                record_op_args=False,
                gc_detect_threshold=None
            )

            activites = [torch_npu.profiler.ProfilerActivity.NPU,
                        torch_npu.profiler.ProfilerActivity.CPU]

            self.prof = torch_npu.profiler.profile(
                with_stack=self.sp_with_stack,
                record_shapes=self.sp_record_shapes,
                profile_memory=self.sp_with_memory,
                activities=activites,
                schedule=torch_npu.profiler.schedule(wait=0, warmup=self.warmup, active=self.active, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.sp_save_path),
                experimental_config=experimental_config)

        def start(self):
            if self.enable:
                self.prof.start()

        def step(self):
            if self.enable:
                self.prof.step()

        def stop(self):
            if self.enable:
                self.prof.stop()
except ImportError:
    # ignore the error if torch_npu is not installed
    pass