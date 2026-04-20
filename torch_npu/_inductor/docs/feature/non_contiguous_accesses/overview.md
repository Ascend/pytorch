# 离散访存特性介绍

## 特性简介
推荐模型的Embedding参数规模极其庞大，在推理及训练过程中需要频繁地对Embedding进行离散访问（embedding、gather等），模型中访存类算子对于inductor中自动融合性能有巨大影响。


| 算子类别               | pytorch算子                                               |
| ------------------ | ------------------------------------------------------- |
| load类(GM到UB为离散访存)  | aten.embedding、aten.index、aten.gather、aten.index_select |
| store类(UB到GM为离散访存) | aten.index_put、aten.scatter                             |
对于这些算子，在inductor中生成的算子将会存在离散访存类型的索引表达式；具体的以`aten.embedding`为例，先通过一次load将索引从gm(in_ptr0)搬运到ub中(tmp0)，再使用索引(tmp0)从gm(in_ptr1)搬运长度为128的向量组成新的ub上的tensor(tmp1)。
``` python
y0=tl.arange(0, Y0BLOCK_SUB) # y0为连续访存，假设Y0BLOCK_SUB为4
# in_ptr0 = [1, 0, 4, 5]
tmp0 = tl.load(in_ptr0 + (y0), y0_mask, other=0.0)
# tmp0 = [1, 0, 8, 10]
# in_ptr1 = [[row0], [row1], ..., [row10]]
tmp1 = tl.load(in_ptr1 + (x1 + 128*tmp0), y0_mask & x1_mask)
# tmp1 = [[row1], [row0], [row8], [row10]]
# tmp1 即为间接访存
```
## 原理
由于A2/A3上仅支持simd访存，对于离散访存场景仅能通过标量搬运，因此上述的离散访存算子将会fallback到eager模式运行。而在A5硬件中加入了simt访存能力，对于间接访存的场景使用simt能够加速离散数据的搬运，本特性将会支持上述的离散访存算子在A5硬件上的inductor融合。

如果在Inductor中生成了间接访存的IR，inductor则需要将该Kernel标记为需要使用间接访存相关算子。对于间接访存相关的算子，由于存在多种不同的Codegen以及Autotune逻辑，使用环境变量进行控制。
1. simd_simt_mix模式：
该情况算子在底层编译器存在3种处理方案：
+ SIMT方案：整体kernel为SIMT硬件实现。
+ SIMT模板方案：整体kernel为SIMD与SIMT混合的实现方案，对于算子中存在间接访存的load、store使用CCE模板算子通过SIMT实现，其余代码部分使用SIMD进行编译实现，可以同时利用SIMT离散访存和SIMD的计算能力。
+ SIMD甜点方案：整体kernel为SIMD的实现方案，对于存在间接访存的算子在ta层进行SIMD优化。
Inductor Autotune通过编译选项在生成不同的算子进行自动选择，将单测性能最优的算子作为整网中运行的算子。
2. fallback模式：
关闭离散访存，离散访存类的算子进行fallback处理。

![test](./arch.drawio.svg)

## 使用案例
以`embedding+sum`融合算子为例，介绍离散访存特性的使用方法。

``` python
import os
import csv
import shutil

import torch
import torch_npu
from torch._dynamo.testing import rand_strided

# 禁用cache
torch._inductor.config.force_disable_caches = True
torch._dynamo.config.recompile_limit = 4096

def delete_file_base(base_path):
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

def profiling_fn(fn, profiling_path, save_profiling=False):
    delete_file_base(profiling_path)
    WAIT=5
    WARMUP=0
    ACTIVE=10
    REPEAT=0
    SKIP_FIRST=0
    TOTAL_STEP = (WAIT + WARMUP + ACTIVE + SKIP_FIRST) * (REPEAT + 1)
    stream = torch.npu.current_stream()
    stream.synchronize()
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_path),
        **{
            'with_stack': False,
            'record_shapes': False,
            'profile_memory': False,
            'schedule': torch_npu.profiler.schedule(
                wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=REPEAT, skip_first=SKIP_FIRST
            )
        }
    ) as prof:
        stream.synchronize()
        for _ in range(TOTAL_STEP):
            fn()
            prof.step()
        stream.synchronize()

    for root, _, files in os.walk(profiling_path):
        for file in files:
            if file != 'kernel_details.csv':
                continue
            target_file = os.path.join(root, file)
            with open(target_file, newline='') as csvfile:
                durations = []
                reader = csv.DictReader(csvfile)
                for row_read in reader:
                    durations.append(float(row_read['Duration(us)']))

    if not durations:
        raise RuntimeError(f"Could not find kernel_details.csv from dir {profiling_path}")
    if not save_profiling:
        delete_file_base(profiling_path)
    return sum(durations) / ACTIVE

def embedding_sum_eager(arg0_1, arg2_1):
    embedding = torch.ops.aten.embedding.default(arg2_1, arg0_1)
    sum_1 = torch.ops.aten.sum.dim_IntList(embedding, [1])
    return sum_1

if __name__ == "__main__":
    arg0_1 = torch.randint(0, 9000, size=(128, 4000), dtype=torch.int32).npu()
    arg2_1 = rand_strided((9000, 128), (128, 1), device='npu', dtype=torch.float32)
    r1 = embedding_sum_eager(arg0_1, arg2_1)
    compiled_embedding_sum = torch.compile(embedding_sum_eager, backend="inductor")
    rc = compiled_embedding_sum(arg0_1, arg2_1)
    torch.testing.assert_close(r1, rc, atol=0.001, rtol=0.001)
    print(f"complied embedding_sum validation passed")

    eager_fn = lambda: embedding_sum_eager(arg0_1, arg2_1)
    inductor_fn = lambda: compiled_embedding_sum(arg0_1, arg2_1)
    eager_time_cost = profiling_fn(eager_fn, profiling_path='./result/eager', save_profiling=False)
    inductor_time_cost = profiling_fn(inductor_fn, profiling_path='./result/inductor', save_profiling=False)
    print(f"profiling eager_fn, fn_path: ./result/eager")
    print(f"profiling inductor, fn_path: ./result/inductor")
    print(f"eager time cost 10 step: {eager_time_cost}")
    print(f"inductor time cost 10 step: {inductor_time_cost}")
```

如果不打开离散访存特性(`export INDUCTOR_INDIRECT_MEMORY_MODE=fallback`),直接执行上述脚本：得到运行结果运行日志，通过profiling不打开离散访存特性，将会从eager模式(440.55us)劣化到inductor模式(1209.80us)
``` bash
[Warning]: tiling struct [ExpandIntoJaggedPermuteTilingData] is conflict with one in file expand_into_jagged_permute_tiling.h, line 21
/usr/local/python3.11.13/lib/python3.11/site-packages/torch/_dynamo/pgo.py:465: UserWarning: dynamo_pgo force disabled by torch._inductor.config.force_disable_caches
  warn_once(
W0226 11:27:21.957000 119301 site-packages/torch/_inductor/debug.py:454] [0/0] model__1_inference_3 debug trace: /data/test_dir/triton/indirect_mem/embedding_sum/torch_compile_debug/run_2026_02_26_11_27_08_101286-pid_119301/torchinductor/model__1_inference_3.0
[2026-02-26 11:27:22] [WARNING] [119301] profiler.py: Invalid parameter export_type: None, reset it to text.
[2026-02-26 11:27:23] [INFO] [119301] profiler.py: Start parsing profiling data: /data/test_dir/triton/indirect_mem/embedding_sum/profile_result/triton_d0894ab32616d540621169a362912f67/localhost.localdomain_119301_20260226112722262_ascend_pt
[2026-02-26 11:27:23] [INFO] [119301] profiler.py: SimpleProfilingAnalyzer parsed in a total time of 0:00:00.495121
complied embedding_sum validation passed
[2026-02-26 11:27:24] [WARNING] [119301] profiler.py: Invalid parameter export_type: None, reset it to text.
[2026-02-26 11:27:24] [WARNING] [119301] profiler.py: Profiler won't be using warmup, this can skew profiler results
[2026-02-26 11:27:24] [INFO] [121654] profiler.py: Start parsing profiling data in sync mode at: /data/test_dir/triton/indirect_mem/embedding_sum/result/eager/localhost.localdomain_119301_20260226112724121_ascend_pt
[2026-02-26 11:27:26] [INFO] [121663] profiler.py: CANN profiling data parsed in a total time of 0:00:02.012654
[2026-02-26 11:27:27] [INFO] [121654] profiler.py: All profiling data parsed in a total time of 0:00:02.915153
[2026-02-26 11:27:27] [WARNING] [119301] profiler.py: Invalid parameter export_type: None, reset it to text.
[2026-02-26 11:27:27] [WARNING] [119301] profiler.py: Profiler won't be using warmup, this can skew profiler results
[2026-02-26 11:27:27] [INFO] [121836] profiler.py: Start parsing profiling data in sync mode at: /data/test_dir/triton/indirect_mem/embedding_sum/result/inductor/localhost.localdomain_119301_20260226112727377_ascend_pt
[2026-02-26 11:27:29] [INFO] [121845] profiler.py: CANN profiling data parsed in a total time of 0:00:02.014242
[2026-02-26 11:27:30] [INFO] [121836] profiler.py: All profiling data parsed in a total time of 0:00:02.928056
profiling eager_fn, fn_path: ./result/eager
profiling inductor, fn_path: ./result/inductor
eager time cost 10 step: 440.55279999999993
inductor time cost 10 step: 1209.8092
```

其中`/data/test_dir/triton/indirect_mem/embedding_sum/torch_compile_debug/run_2026_02_26_11_27_08_101286-pid_119301/torchinductor/model__1_inference_3.0`目录为dump融合算子的临时文件
``` bash
ls -alh torch_compile_debug/run_2026_02_26_11_27_08_101286-pid_119301/torchinductor/model__1_inference_3.0
total 44K
drwxr-xr-x 2 root root 4.0K Feb 26 11:27 .
drwxr-xr-x 3 root root 4.0K Feb 26 11:27 ..
-rw-r--r-- 1 root root  733 Feb 26 11:27 fx_graph_readable.py -> eager算子
-rw-r--r-- 1 root root 3.0K Feb 26 11:27 fx_graph_runnable.py -> eager算子，可直接运行
-rw-r--r-- 1 root root  733 Feb 26 11:27 fx_graph_transformed.py
-rw-r--r-- 1 root root  168 Feb 26 11:27 inductor_provenance_tracking_node_mappings.json
-rw-r--r-- 1 root root    2 Feb 26 11:27 inductor_triton_kernel_to_post_grad_nodes.json
-rw-r--r-- 1 root root 2.1K Feb 26 11:27 ir_post_fusion.txt -> scheduler计算融合后的ir
-rw-r--r-- 1 root root 2.1K Feb 26 11:27 ir_pre_fusion.txt -> scheduler计算融合前的ir
-rw-r--r-- 1 root root 6.3K Feb 26 11:27 output_code.py -> 融合生成的算子临时文件
```

不打开该特性时，aten.embedding算子将会fallback到eager执行，观察该文件embedding算子将会被单独调用，不会融合生成triton kernel。
``` bash
def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        # Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
        buf0 = torch.ops.aten.embedding.default(arg1_1, arg0_1)
        del arg0_1
        del arg1_1
        buf1 = buf0
        assert_size_stride(buf1, (128, 4000, 128), (512000, 128, 1))
        del buf0
        buf2 = empty_strided((128, 128), (128, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [sum_1], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_unk_fused_sum_0.run(buf1, buf2, 128, 128, 4000, stream=stream0)
        del buf1
    return (buf2, )
```
`export INDUCTOR_INDIRECT_MEMORY_MODE=simd_simt_mix`打开离散访存融合后，inductor算子侧时间降低到260us。
``` bash
W0226 11:54:57.336000 142389 site-packages/torch/_inductor/debug.py:454] [0/0] model__1_inference_3 debug trace: /data/test_dir/triton/indirect_mem/embedding_sum/torch_compile_debug/run_2026_02_26_11_54_34_998561-pid_142389/torchinductor/model__1_inference_3.0
[2026-02-26 11:54:57] [WARNING] [142389] profiler.py: Invalid parameter export_type: None, reset it to text.
[2026-02-26 11:54:57] [INFO] [142389] profiler.py: Start parsing profiling data: /data/test_dir/triton/indirect_mem/embedding_sum/profile_result/triton_77c6b26b8a3caac6ec07cf8db01ce1da/localhost.localdomain_142389_20260226115457371_ascend_pt
[2026-02-26 11:54:57] [INFO] [142389] profiler.py: SimpleProfilingAnalyzer parsed in a total time of 0:00:00.288046
complied embedding_sum validation passed
[2026-02-26 11:54:58] [WARNING] [142389] profiler.py: Invalid parameter export_type: None, reset it to text.
[2026-02-26 11:54:58] [WARNING] [142389] profiler.py: Profiler won't be using warmup, this can skew profiler results
[2026-02-26 11:54:58] [INFO] [143471] profiler.py: Start parsing profiling data in sync mode at: /data/test_dir/triton/indirect_mem/embedding_sum/result/eager/localhost.localdomain_142389_20260226115458044_ascend_pt
[2026-02-26 11:55:00] [INFO] [143480] profiler.py: CANN profiling data parsed in a total time of 0:00:02.012710
[2026-02-26 11:55:01] [INFO] [143471] profiler.py: All profiling data parsed in a total time of 0:00:02.887635
[2026-02-26 11:55:01] [WARNING] [142389] profiler.py: Invalid parameter export_type: None, reset it to text.
[2026-02-26 11:55:01] [WARNING] [142389] profiler.py: Profiler won't be using warmup, this can skew profiler results
[2026-02-26 11:55:01] [INFO] [143653] profiler.py: Start parsing profiling data in sync mode at: /data/test_dir/triton/indirect_mem/embedding_sum/result/inductor/localhost.localdomain_142389_20260226115501306_ascend_pt
[2026-02-26 11:55:03] [INFO] [143662] profiler.py: CANN profiling data parsed in a total time of 0:00:02.016730
[2026-02-26 11:55:04] [INFO] [143653] profiler.py: All profiling data parsed in a total time of 0:00:02.880842
profiling eager_fn, fn_path: ./result/eager
profiling inductor, fn_path: ./result/inductor
eager time cost 10 step: 441.92249999999996
inductor time cost 10 step: 260.61800000000005
```
观察生成的dump文件，triton kernel将embedding和sum算子融合后减少算子间的内存搬运，提高了算子执行效率。
``` python
@triton.jit
def triton_unk_fused_embedding_sum_0(in_ptr0, in_ptr1, out_ptr0, y0_numel, x1_numel, r2_numel, Y0BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, R2BLOCK_SUB : tl.constexpr):
    x1_numel = 128
    X1BLOCK_SUB: tl.constexpr = 128
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,None,:]
        x1_mask = x1 < x1_numel
        _tmp8 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
        for loop_r2 in range(loops_r2):
            r2 = (loop_r2 * R2BLOCK_SUB) + base_r2[None,:,None]
            r2_mask = r2 < r2_numel
            tmp0 = tl.load(in_ptr0 + (r2 + 4000*y0), r2_mask & y0_mask, other=0.0)
            tmp1 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 9000, tl.int32)
            tmp2 = tmp0 + tmp1
            tmp3 = tmp0 < 0
            tmp4 = tl.where(tmp3, tmp2, tmp0)
            tl.device_assert(((0 <= tmp4) & (tmp4 < 9000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp4 < 9000")
            tmp6 = tl.load(in_ptr1 + (x1 + 128*tmp4), r2_mask & y0_mask & x1_mask)
            tmp7 = tl.reshape(tmp6, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
            tmp9 = _tmp8 + tmp7
            _tmp8 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp9, _tmp8)
        tmp8 = tl.sum(_tmp8, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
        tl.store(out_ptr0 + (x1 + 128*y0 ), tmp8, y0_mask & x1_mask)
```