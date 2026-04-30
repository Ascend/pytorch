import argparse
import glob
import json
import os
import random
import stat
import time
from datetime import datetime

import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score

torch.manual_seed(2024)
random.seed(2024)

MODEL_NAME = "ETA"
STD_DEV = (2 / 512) ** 0.5 
EMBEDDING_FEATURE_NUM = 27 
TARGET_FIELDS = ["206", "207", "216", "210"] 

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

def json_file_load(json_name: str, json_path: str) -> dict:
    """
    Load a JSON file from the specified path.
    """
    flags = os.O_RDONLY
    modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    try:
        with os.fdopen(os.open(json_path, flags, modes), "r") as fp:
            json_re = json.load(fp)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{json_name} file not found: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading {json_name} file: {e}") from e

    return json_re

def collate_fn(batch):
    input_dicts = [item[0] for item in batch]
    target_dicts = [item[1] for item in batch]
    input_tensors = {}
    for key in input_dicts[0].keys():
        tensors = [d[key] for d in input_dicts if key in d and d[key] is not None]
        if not tensors:
            continue
        if tensors[0].dim() == 0:
            tensors = [t.unsqueeze(0) for t in tensors]
        input_tensors[key] = pad_sequence(tensors, batch_first=True)

    target_tensors = {}
    for key in target_dicts[0].keys():
        tensors = [d[key] for d in target_dicts if key in d and d[key] is not None]
        if not tensors:
            continue
        target_tensors[key] = torch.stack(tensors)

    return input_tensors, target_tensors

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_size", type=int, default=16, help="Embedding size")
    parser.add_argument("--attention_dim", type=int, default=64, help="")
    parser.add_argument("--enable_compile", action="store_true", help="")
    parser.add_argument("--num_heads", type=int, default=4, help="")
    parser.add_argument("--short_output_dim", type=int, default=16, help="short attention output dimension")
    parser.add_argument("--max_seq_len", type=int, default=50, help="")
    parser.add_argument("--topk", type=int, default=16, help="")
    parser.add_argument("--deep_layers", type=str, default="512,256,128,64", help="deep layers")
    parser.add_argument("--hash_bits", type=int, default=32, help="")
    parser.add_argument("--reuse_hash", type=bool, default=True, help="")
    parser.add_argument("--batch_size", type=int, default=4096, help="Number of batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        choices=["Adam", "Adagrad", "GD", "Momentum"], help="")
    parser.add_argument('--early_stop_patience', type=int, default=5, help="")
    parser.add_argument("--data_dir", type=str, default="./aliccp/aliccp_out/", help="data dir")
    parser.add_argument("--dt_dir", type=str, default='', help="data dt partition")
    parser.add_argument("--model_dir", type=str, default=f"./", help="code check point dir")
    parser.add_argument("--clear_existing_model", action="store_true", help="")
    parser.add_argument("--task_type", type=str, default="train",
                        choices=["train", "eval", "predict"], help="task type")
    parser.add_argument("--max_steps", type=int, default=100,  
                       help="Total training steps (优先级高于num_epochs，-1表示使用num_epochs控制)")
    parser.add_argument('--epoch_num', type=int, default=1, help="Number of epochs")
    parser.add_argument('--train_batch_num', type=int, default=2000, help="Number of train batchs")
    parser.add_argument('--test_batch_num', type=int, default=10, help="Number of test batchs")
    parser.add_argument("--enable_profiler", action="store_true",
                       help="Enable profiler for performance analysis")
    parser.add_argument("--profiler_start_step", type=int, default=5,
                       help="Output directory for trained model")
    parser.add_argument("--profiler_end_step", type=int, default=8,
                    help="Output directory for trained model")
    parser.add_argument("--profiler_save_path", type=str, default="./profile",
                    help="Output directory for trained model")
    return parser.parse_args()


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self._load_hdf5()

    def _load_hdf5(self):
        with h5py.File(self.hdf5_path, 'r') as f:
            self.input_sample = {}
            self.target_sample = {}
            y = np.array(f["y"])
            z = np.array(f["z"])
            fields = [key for key in f.keys() if key not in ["y", "z"]]
            self.target_sample.update({"y": torch.tensor(y, dtype=torch.float32)})
            self.target_sample.update({"z": torch.tensor(z, dtype=torch.float32)})
            for multi_field in fields:
                self.input_sample.update({multi_field: torch.tensor(np.array(f[multi_field]), dtype=torch.int64)})
        self._length = len(y)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        input_dict = {k: v[idx] for k, v in self.input_sample.items()}
        target_dict = {k: v[idx] for k, v in self.target_sample.items()}
        return input_dict, target_dict


class TorchDataSet(ConcatDataset):
    def __init__(self, files):
        datasets = [HDF5Dataset(fp) for fp in files]
        super().__init__(datasets)


class ETA(nn.Module):
    def __init__(self, spec, params):
        super(ETA, self).__init__()
        self.spec = spec
        self.params = params

        # embedding layers
        self.emb_weights = nn.ModuleDict()
        for key, vocab_len in spec["vocab_length"].items():
            self.emb_weights[key] = nn.Embedding(vocab_len + 1, params.embedding_size)
            nn.init.normal_(self.emb_weights[key].weight, std=STD_DEV)

        self.hash_weights = nn.Parameter(
            torch.randn(params.embedding_size, params.hash_bits),
            requires_grad=False
        )

        # attention paramters
        self.short_attentions = nn.ModuleList()
        self.long_attentions = nn.ModuleList()
        for _ in range(len(TARGET_FIELDS)): # 4 target fields
            self.short_attentions.append(ShortAttention(params))
            self.long_attentions.append(LongAttention(self.hash_weights, params))

        # mlp layer
        self.mlp = nn.Sequential()
        deep_layers = list(map(int, params.deep_layers.strip().split(",")))
        input_dim = EMBEDDING_FEATURE_NUM * params.embedding_size
        for i, dim in enumerate(deep_layers):
            self.mlp.add_module(f'mlp{i}', nn.Linear(input_dim, dim))
            self.mlp.add_module(f'relu{i}', nn.ReLU())
            input_dim = dim

        # output layer
        self.output = nn.Linear(input_dim, 1)

    def embedding_lookup_sparse(self, params: nn.Embedding, ids: torch.Tensor, combiner: str):
        mask = (ids >= 0).float().unsqueeze(-1)
        ids = ids.clone()
        ids[ids == -1] = 0
        embedding = params(ids) * mask
        if combiner == "sum":
            return embedding.sum(dim=1)
        elif combiner == "mean":
            return embedding.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-7)
        else:
            raise ValueError("combiner only supoort 'sum', 'mean'")

    def forward(self, features):
        # embedding
        embeddings = {}
        masks = {}
        for key in self.spec["one_hot_fields"]:
            embeddings[key] = self.emb_weights[key](features[key]).unsqueeze(1)
        for key in self.spec["multi_hot_fields"]:
            feat = features[key]
            masks[key] = (feat >= 0).bool().unsqueeze(1)
            feat = feat.clone()
            feat[feat == -1] = 0
            embeddings[key] = self.emb_weights[key](feat)
        for key in self.spec["special_fields"]:
            embeddings[key] = self.embedding_lookup_sparse(
                self.emb_weights[key], features[key], combiner="sum"
            ).unsqueeze(1)

        def long_emb_cat(field_name):
            dense_embedding = embeddings.get(field_name)
            dense_mask = masks.get(field_name)
            if dense_embedding is None or dense_mask is None:
                raise ValueError(f"Field {field_name} not found in embeddings or masks")
            padded_embedding = F.pad(dense_embedding, (0, 0, 0, self.params.max_seq_len), "constant", 0)
            padded_mask = F.pad(dense_mask, (0, self.params.max_seq_len), "constant", False)
            return (
                padded_embedding[:, :self.params.topk, :],
                padded_embedding[:, :self.params.max_seq_len, :],
                padded_mask[:, :, :self.params.topk],
                padded_mask[:, :, :self.params.max_seq_len]
            )
        emb_cats = [long_emb_cat(field) for field in self.spec["multi_hot_fields"]]
        target_fields = TARGET_FIELDS

        short_attns = []
        long_attns = []

        for i, (emb_cat, target) in enumerate(zip(emb_cats, target_fields)):
            emb_target = embeddings[target]

            emb_short = emb_cat[0]
            mask_short = emb_cat[2]
            short_attns.append(self.short_attentions[i](emb_target, emb_short, mask_short))

            emb_long = emb_cat[1]
            mask_long = emb_cat[3]
            long_attns.append(self.long_attentions[i](emb_target, emb_long, mask_long))

        # concat all embeddings
        all_embs = []
        for field in self.spec["one_hot_fields"] + self.spec["special_fields"]:
            if embeddings[field].dim() > 3:
                all_embs.append(embeddings[field].squeeze(1))
            else:
                all_embs.append(embeddings[field])
        all_embs += short_attns + long_attns
        all_embs = torch.cat(all_embs, dim=1)
        if all_embs.dim() >= 2:
            concat_emb = all_embs.squeeze(1)
        else:
            concat_emb = all_embs

        concat_emb = concat_emb.reshape(concat_emb.size(0), -1)
        # mlp and outputs
        mlp_out = self.mlp(concat_emb)
        logits = self.output(mlp_out).squeeze()
        pred = torch.sigmoid(logits)
        return pred, logits

    def build_loss(self, pred, labels, click_weight=0.14, epsilon=1e-7):
        if pred.shape != labels.shape:
            raise ValueError(f"pred and labels must be the same shape. "
                             f"pred shape: {pred.shape}, labels shape: {labels.shape}")
        pred = torch.clamp(pred, min=epsilon, max=1 - epsilon)
        loss = - (1 - click_weight) / click_weight * labels * torch.log(pred) - (1 - labels) * torch.log(1 - pred)
        return loss.mean()

    def build_optimizer(self):
        if self.params.optimizer == "Adam":
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.params.learning_rate,
                betas=[0.9, 0.999], eps=1e-8
            )
        elif self.params.optimizer == "Adagrad":
            optimizer = optim.Adagrad(
                params=self.parameters(),
                lr=self.params.learning_rate,
                initial_accumulator_value=1e-6
            )
        elif self.params.optimizer == "Momentum":
            optimizer = optim.SGD(
                params=self.parameters(),
                lr=self.params.learning_rate,
                momentum=0.95
            )
        elif self.params.optimizer == "SGD":
            optimizer = optim.SGD(
                params=self.parameters(),
                lr=self.params.learning_rate,
            )
        else:
            raise ValueError("Unsupported optimizer type: {}".format(args.optimizer))
        return optimizer


class ShortAttention(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.attention_dim = params.attention_dim
        self.num_heads = params.num_heads
        self.key_dim = self.attention_dim // self.num_heads

        self.q_fc = nn.Linear(params.embedding_size, self.attention_dim)
        self.k_fc = nn.Linear(params.embedding_size, self.attention_dim)
        self.v_fc = nn.Linear(params.embedding_size, self.attention_dim)
        self.o_fc = nn.Linear(self.attention_dim, params.short_output_dim)

    def forward(self, target_input, seq_input, mask):
        # target_input:[B, 1, E], seq_input:[B, S, E], mask:[B, 1, S] B = target_input.size(0)
        # logger.info(f"tgt shape: {target_input.shape}, seq shape: {seq_input.shape}, mask shape: {mask.shape}")
        b, tgt_len = target_input.shape[:2]
        b, seq_len = seq_input.shape[:2]
        query = self.q_fc(target_input) # [B, 1, A]
        key = self.k_fc(seq_input) # [B, S, A]
        value = self.v_fc(seq_input) # [B, S, A]

        # split heads
        query = query.view(b, tgt_len, self.num_heads, self.key_dim).permute(0, 2, 1, 3) # [B, Heads, 1, key_dim]
        key = key.view(b, seq_len, self.num_heads, self.key_dim).permute(0, 2, 3, 1) # [B, Heads, key_dim, S]
        value = value.view(b, seq_len, self.num_heads, self.key_dim).permute(0, 2, 1, 3) # [B, Heads, S, key_dim]

        # scaled dot-product attention
        scores = torch.matmul(query, key) / (self.key_dim ** 0.5) # [B, Heads, 1, S]
        scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(scores, dim=-1)

        output = torch.matmul(attn, value) # [B, Heads, 1, key_dim]
        output = output.permute(0, 2, 1, 3)
        b, lens, heads, dims = output.shape
        output = output.reshape(b * lens, 1, heads * dims)
        output = self.o_fc(output)
        return output


class LongAttention(nn.Module):
    def __init__(self, hash_weights, params):
        super().__init__()
        self.hash_weights = hash_weights
        self.reuse_hash = params.reuse_hash
        self.topk = params.topk
        self.short_attn = ShortAttention(params)

    def lsh_hash(self, vecs, hash_weights):
        rotated_vecs = torch.matmul(vecs, hash_weights)
        return (rotated_vecs > 0).float()

    def forward(self, target_input, seq_input, mask):
        # target_input:[B, 1, E], seq_input:[B, S, E], mask:[B, 1, S]
        b, s, e = seq_input.shape
        hash_weights = self.hash_weights if self.reuse_hash else nn.Parameter(
            torch.randn(e, self.hash_weights.size(-1)),
            requires_grad=False
        ) # [E, hash_bits]

        target_hash = self.lsh_hash(target_input.squeeze(), hash_weights) # [B, hash_bits]
        seq_hash = self.lsh_hash(seq_input, hash_weights) # [B, S, hash_bits]
        # calculate similarity(hamming distance)
        hash_sim = - torch.sum(torch.abs(seq_hash - target_hash.unsqueeze(1)), dim=-1) # [B, S]
        hash_sim = hash_sim.masked_fill(~mask.squeeze(1), float('-inf'))

        # select topk
        _, topk_idx = torch.topk(hash_sim, self.topk, dim=-1)  # [B, topk]

        topk_seq = torch.gather(
            seq_input,
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, e)
        )

        topk_idx = topk_idx.unsqueeze(1)
        topk_mask = torch.gather(mask, 2, topk_idx)

        # apply short attention to topk
        output = self.short_attn(target_input, topk_seq, topk_mask)
        return output

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


def device_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.npu.is_available():
        torch.npu.synchronize()


def evaluate_op(model: ETA, dataloader, device, args):
    mod = 'compile' if args.enable_compile else 'eager'
    dl = iter(dataloader)
    features, _ = next(dl)
    features = {k: v.to(device) for k, v in features.items()}

    for key, value in features.items():
        if isinstance(value, torch.Tensor) and value.dtype == torch.int64:
            features[key] = value.to(torch.int32)

    if mod == 'compile':
        model = torch.compile(model, dynamic=False)
    model.eval()
    with torch.no_grad():
        model(features) 
        device_synchronize()

    prof = None
    if args.enable_profiler:
        profiling_save_path = args.profiler_save_path + '/' + MODEL_NAME + '/' + mod
        prof = get_profile(args.profiler_start_step, args.profiler_end_step, profiling_save_path)
        prof.start()

    exec_times = []
    with torch.no_grad():
        for i in range(args.max_steps):
            e2etime1 = time.perf_counter()
            model(features)
            device_synchronize()
            e2etime2 = time.perf_counter()
            elapsed_ms = (e2etime2 - e2etime1) * 1000
            print(f"iterations {i} / {args.max_steps}: [{mod}] e2e time: {elapsed_ms} ms.")

            if i >= 10:
                exec_times.append(elapsed_ms)

            if args.enable_profiler:
                prof.step()
    if args.enable_profiler:
        prof.stop()   
    if exec_times:
        avg_time = sum(exec_times) / len(exec_times)
        print("Step time consumption statistics (excluding the first 10 steps)")
        print(f"\n>>> [{mod}] avg e2e: {avg_time:.3f} ms <<<")

def main(args):
    test_files = glob.glob("%stest/data_test.csv.hd5.*" % args.data_dir)
    spec = json_file_load("spec", os.path.join(args.data_dir, "spec.json"))
    model = ETA(spec, args)
    device_type = detect_device_type()
    print(f"Using device type {device_type}")
    device = torch.device(device_type)
    model.to(device)
    test_dataset = TorchDataSet(test_files[1:2])
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                collate_fn=collate_fn,
                                prefetch_factor=100,
                                num_workers=10)
    evaluate_op(model, test_loader, device, args)


if __name__ == "__main__":
    print(f"{MODEL_NAME} inference begin !!!")
    args = parse_arguments()
    main(args)