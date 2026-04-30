# -*- coding: utf-8 -*-
import os
import time
import argparse
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from CTR_Algorithm.pytorch.model import DIN
from CTR_Algorithm.data.AmazonDataPreprocess import AmazonBookPreprocess

MODEL_NAME = "DIN"

data = pd.read_csv('CTR_Algorithm/data/amazon-books-100k.txt')
data = AmazonBookPreprocess(data)
fields = data.max().max()

data_X = data.iloc[:,:-1]
data_y = data.label.values

tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size = 0.2, random_state=42, stratify=data_y)
train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size = 0.25, random_state=42, stratify=tmp_y)

train_X = torch.from_numpy(train_X.values).long()
val_X = torch.from_numpy(val_X.values).long()
test_X = torch.from_numpy(test_X.values).long()

train_y = torch.from_numpy(train_y).long()
val_y = torch.from_numpy(val_y).long()
test_y = torch.from_numpy(test_y).long()

train_set = Data.TensorDataset(train_X, train_y)
val_set = Data.TensorDataset(val_X, val_y)
train_loader = Data.DataLoader(dataset=train_set, batch_size=32, shuffle=True)
val_loader = Data.DataLoader(dataset=val_set, batch_size=32, shuffle=False)

def get_profile(profiler_start_step: int, profiler_end_step: int, profiling_save_path: str):
    warm_step = profiler_start_step
    active_step = profiler_end_step - warm_step +1
    print(f"[Profile INFO]: warm_step: {warm_step}, active_step: {active_step}, profiling_save_path: {profiling_save_path}")
    device = detect_device_type()
    if device == 'npu':
        import torch_npu
        g_prof_config = torch_npu.profiler._ExperimentalConfig(
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
        gc_detect_threshold=None)

        return torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=warm_step, active=active_step, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_save_path),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_modules=False,
                with_flops=False,
                experimental_config=g_prof_config)
    elif device == 'cuda':
        from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
        return profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=warm_step, active=active_step, repeat=1),
            on_trace_ready=tensorboard_trace_handler(profiling_save_path),
            record_shapes=False,
            profile_memory=False,
        )
    else:
        print(f"[Warning]: No supported acceleration device (CUDA/NPU) detected, profiler will be disabled")
        return None

def detect_device_type():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        try:
            import torch_npu
            import os
            os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'
            if torch.npu.is_available():
                return "npu"
        except ImportError:
            pass
    except ImportError:
        pass
    return "cpu"

def device_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.npu.is_available():
        torch.npu.synchronize()

def eval_op_prof(model: nn.Module, device, mod, args):
    dl = iter(val_loader)
    x, _ = next(dl)
    x = x.to(device)
    pt_path= MODEL_NAME +'_'+ mod + '_result.pt'

    model.eval()
    with torch.no_grad():
        res = model(x) 
        torch.save(res, pt_path)
        device_synchronize()

    prof = None
    if args.enable_profiler:
        profiling_save_path = args.profiler_save_path + '/' + MODEL_NAME + '/' + mod
        prof = get_profile(args.profiler_start_step, args.profiler_end_step, profiling_save_path)
        prof.start()

    execution_times = []
    with torch.no_grad():
        for i in range(args.max_steps):
            start_time = time.time()
            model(x)
            device_synchronize()
            end_time = time.time()
            step_time_ms = (end_time - start_time) * 1000
            print(f"[{mod}]: Step {i}: {step_time_ms:.4f} ms")
            
            if i >= 10:
                execution_times.append(step_time_ms)

            if args.enable_profiler:
                prof.step()
            
        if args.enable_profiler:
            prof.stop()
        if execution_times:
            avg_ms = sum(execution_times) / len(execution_times)
            print(f"[{mod}]:: Avg over {len(execution_times)} steps: {avg_ms:.4f} ms")

parser = argparse.ArgumentParser(description=MODEL_NAME + " infernece")
parser.add_argument("--max_steps", type=int, default=200,  
                    help="Total training steps")
parser.add_argument("--enable_compile", action="store_true",
                        help="Enable torch.compile and Inductor backend")
parser.add_argument("--enable_profiler", action="store_true",
                       help="Enable profiler for performance analysis")
parser.add_argument("--profiler_start_step", type=int, default=5,
                    help="Output directory for trained model")
parser.add_argument("--profiler_end_step", type=int, default=8,
                    help="Output directory for trained model")
parser.add_argument("--profiler_save_path", type=str, default="./profile",
                    help="Output directory for trained model")
args = parser.parse_args()

model = DIN.DeepInterestNet(feature_dim=fields, embed_dim=8, mlp_dims=[64,32], dropout=float(0.2))
device_type = detect_device_type()
device = torch.device(device_type)
model.to(device)
mod = 'eager'
if args.enable_compile:
    model = torch.compile(model, dynamic=False)
    mod = 'compile'
print(f"{MODEL_NAME} inference begin, mode is {mod}")
eval_op_prof(model, device, mod, args)