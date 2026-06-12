import os
import argparse
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

import torch_npu

from config import CONFIG
from datasets.dummy_dataset import DummySparseDriveDataset, simple_collate
from models import SimpleV1SparseDrive

def patch_torch_npu_dvm_decomp_excludes(verbose: bool = True):
    """
    Runtime patch for:
      .../site-packages/torch_npu/_inductor/dvm/decomp.py

    It appends:
      aten._native_batch_norm_legit_functional
      aten.native_batch_norm_backward

    into decomps_to_exclude_npu (idempotent), then triggers patch_decomp().
    """
    import importlib
    import torch

    mod = importlib.import_module("torch_npu._inductor.dvm.decomp")

    aten = torch.ops.aten
    to_add = [
        aten._native_batch_norm_legit_functional,
        aten.native_batch_norm_backward,
    ]

    if not hasattr(mod, "decomps_to_exclude_npu"):
        raise RuntimeError("torch_npu._inductor.dvm.decomp has no decomps_to_exclude_npu")

    lst = mod.decomps_to_exclude_npu
    added = 0
    for op in to_add:
        if op not in lst:
            lst.append(op)
            added += 1

    if verbose:
        print(
            f"[PATCH] torch_npu dvm decomp: added {added}/{len(to_add)} excludes. "
            f"Now len={len(lst)}"
        )
    if hasattr(mod, "patch_decomp"):
        mod.patch_decomp()
        if verbose:
            print("[PATCH] torch_npu dvm decomp: patch_decomp() called.")
    else:
        if verbose:
            print("[PATCH] torch_npu dvm decomp: no patch_decomp() found; only list updated.")

    return mod

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (default: eager).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch_npu profiler (default: off).",
    )
    parser.add_argument(
        "--prof-dir",
        type=str,
        default=os.path.join(".", "prof", "example.prof"),
        help="Profiler output dir (only used when --profile).",
    )
    parser.add_argument(
        "--npu-backend",
        type=str,
        default="dvm")
    return parser.parse_args()


def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def get_profiler_config():
    return torch_npu.profiler._ExperimentalConfig(
        export_type=[
            torch_npu.profiler.ExportType.Text,
            torch_npu.profiler.ExportType.Db,
        ],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
    )


class NullProfiler:
    """兼容 profile 的空实现：start/step/stop 都是 no-op。"""
    def start(self): pass
    def step(self): pass
    def stop(self): pass


def maybe_enable_compile(model, enabled: bool):
    patch_torch_npu_dvm_decomp_excludes(verbose=True)
    if not enabled:
        print("[INFO] torch.compile disabled (eager mode).")
        return model
    try:
        model.img_backbone = torch.compile(
            model.img_backbone, dynamic=False, mode="reduce-overhead"
        )
        print("[INFO] torch.compile enabled successfully.")
    except Exception:
        print("[WARN] torch.compile failed, fallback to eager mode.")

    return model


def build_profiler(enabled: bool, prof_output_dir: str, warmup_step_num=5, exec_step_num=10):
    if not enabled:
        print("[INFO] profiler disabled.")
        return NullProfiler()

    os.makedirs(os.path.dirname(prof_output_dir), exist_ok=True)
    g_prof_config = get_profiler_config()

    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=warmup_step_num, active=exec_step_num, repeat=1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_output_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=g_prof_config,
    )
    print(f"[INFO] profiler enabled. output_dir = {prof_output_dir}")
    return prof


def main():
    args = parse_args()
    os.environ['TORCHINDUCTOR_NPU_BACKEND']=args.npu_backend
    torch.use_deterministic_algorithms(True)
    cfg = CONFIG
    set_seed(cfg["seed"])

    device = (
        "npu" if hasattr(torch, "npu") and torch.npu.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"[INFO] device = {device}")
    print(f"[INFO] flags: compile={args.compile}, profile={args.profile}")

    dataset = DummySparseDriveDataset(**cfg["dummy_dataset"])
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=simple_collate,
        drop_last=True,
    )

    model = SimpleV1SparseDrive(**cfg["model"]).to(device)
    print(model)

    # optional compile
    model = maybe_enable_compile(model, enabled=args.compile)

    # optional profiler
    warmup_step_num = 5
    exec_step_num = 10
    prof = build_profiler(
        enabled=args.profile,
        prof_output_dir=args.prof_dir,
        warmup_step_num=warmup_step_num,
        exec_step_num=exec_step_num,
    )

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    prof.start()
    print("[INFO] start training debug...")

    for step, batch in enumerate(dataloader):
        if step >= cfg["train_steps"]:
            break

        step_start_time = time.time()

        inputs = batch["inputs"].to(device)
        data_samples = batch["data_samples"]

        optimizer.zero_grad()
        loss_dict = model(inputs, data_samples=data_samples, mode="loss")
        total_loss = sum(v for v in loss_dict.values())
        total_loss.backward()
        optimizer.step()

        prof.step()

        step_time = time.time() - step_start_time
        log_str = f"[step {step}] total_loss={total_loss.item():.6f}, time={step_time:.4f}s"
        for k, v in loss_dict.items():
            log_str += f", {k}={v.item():.6f}"
        print(log_str)

    print("[INFO] train debug finished.")
    prof.stop()

    # quick sanity forward
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        inputs = batch["inputs"].to(device)
        data_samples = batch["data_samples"]
        outputs = model(inputs, data_samples=data_samples, mode="predict")

    print("=" * 120)
    print("[det]")
    print("det query:", outputs["head"]["det"]["query"].shape)
    print("det cls layers:", len(outputs["head"]["det"]["all_cls_scores"]))
    print("det reg layers:", len(outputs["head"]["det"]["all_reg_preds"]))
    print("det last cls:", outputs["head"]["det"]["all_cls_scores"][-1].shape)
    print("det last reg:", outputs["head"]["det"]["all_reg_preds"][-1].shape)

    print("\n[map]")
    print("map query:", outputs["head"]["map"]["query"].shape)
    print("map cls layers:", len(outputs["head"]["map"]["all_cls_scores"]))
    print("map reg layers:", len(outputs["head"]["map"]["all_reg_preds"]))
    print("map last cls:", outputs["head"]["map"]["all_cls_scores"][-1].shape)
    print("map last reg:", outputs["head"]["map"]["all_reg_preds"][-1].shape)

    print("\n[motion_plan]")
    mp = outputs["head"]["motion_plan"]
    for k, v in mp.items():
        print(k, v.shape if v is not None else None)

    print("\n[depth_branch]")
    for i, d in enumerate(outputs["depth_branch"]):
        print(f"depth[{i}] shape: {d.shape}")

    print("\n[fpn_feats]")
    for i, f in enumerate(outputs["fpn_feats"]):
        print(f"fpn_feats[{i}] shape: {f.shape}")


if __name__ == "__main__":
    main()