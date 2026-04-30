import torch
import time

from transformers import TrainerCallback

class TimingCallback(TrainerCallback):
    
    def __init__(self, profiler=None, mod='eager'):
        self.step_start_time = None
        self.step_times = []
        self.epoch_start_time = None
        self.profiler = profiler
        self.mod = mod
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {state.epoch + 1 if state.epoch else 1} begin")
        print(f"{'='*60}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        print(f"\n{'='*60}")
        print(f"Epoch {int(state.epoch)} ended, time taken:  {epoch_time:.2f}s")
        print(f"{'='*60}\n")
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        step_time = (time.time() - self.step_start_time) * 1000 
        self.step_times.append(step_time)

        if self.profiler is not None:
            try:
                self.profiler.step()
            except (AssertionError, StopIteration):
                self.profiler = None
                print("Profiler done.")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.npu.is_available():
            torch.npu.synchronize()

        print(f"[{self.mod}] step {state.global_step:4d}  "
              f"step_time: {step_time:.3f}ms  "
              f"loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        print("============= training begin =============")
        if self.profiler is not None:
            self.profiler.start()

    def on_train_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.stop()
        if self.step_times:
            valid_steps = self.step_times[100:] if len(self.step_times) > 100 else self.step_times
            avg_time = sum(valid_steps) / len(valid_steps)
            total_time = sum(valid_steps)
           
            print(f"\n{'='*60}")
            print("Step time consumption statistics (keeping only the last 100 steps)")
            print(f"[{self.mod}] total step:{len(valid_steps)}"
                  f" total steps time:{total_time:.2f}ms,"
                  f" avg step time:{avg_time:.3f}ms")
            print(f"{'='*60}")


def detect_device_type():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.npu.is_available():
                import os
                os.environ['TORCH_NPU_USE_COMPATIBLE_IMPL'] = '1'
                return "npu"
        except ImportError:
            pass
    except ImportError:
        pass
    return "cpu"

def get_profile(profiler_start_step: int, profiler_end_step: int, profiling_save_path: str):
    warm_step = profiler_start_step
    active_step = profiler_end_step - warm_step +1
    print(f"[Profile INFO]: warm_step: {warm_step}, active_step: {active_step}, profiling_save_path: {profiling_save_path}")
    device_type = detect_device_type()
    try:
        if device_type == "cuda":
            from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
            return profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=0, warmup=warm_step, active=active_step, repeat=1),
                on_trace_ready=tensorboard_trace_handler(profiling_save_path),
                record_shapes=False,
                profile_memory=False,
            )
        elif device_type == "npu":
            import torch_npu
            exp_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[
                    torch_npu.profiler.ExportType.Text,
                    torch_npu.profiler.ExportType.Db
                ],
                profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
                msprof_tx=False,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                l2_cache=False,
                op_attr=False,
                data_simplification=False,
                record_op_args=False,
                gc_detect_threshold=None
            )
            return torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU
                ],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=warm_step, active=active_step, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_save_path),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_modules=False,
                with_flops=False,
                experimental_config=exp_config
            )
        else:
            print(f"[Warning]: No supported acceleration device (CUDA/NPU) detected, profiler will be disabled")
            return None
    except ImportError as e:
        print(f"[Warning]: Failed to import modules required by profiler: {e}")
        return None