import os
import stat
import glob
import json
import random
import shutil
import logging
import time
from datetime import datetime
import argparse

import pytz
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.metrics import roc_auc_score

torch.manual_seed(2024)
random.seed(2024)

MODEL_NAME = "MMOE"
EMBEDDING_FEATURE_NUM = 23 
STD_DEV = (2 / 512) ** 0.5 

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Example with Command Line Arguments')
    parser.add_argument('--embedding_size', type=int, default=16, help="Embedding size")
    parser.add_argument('--batch_size', type=int, default=4096, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "Adagrad", "GD", "Momentum"],
                        help="Optimizer type")
    parser.add_argument('--expert_layers', type=str, default="512,256", help="Expert layers")
    parser.add_argument('--tower_layers', type=str, default="128,64", help="tower layers")
    parser.add_argument('--ctr_task_wgt', type=float, default=0.5, help="loss weight of ctr task")
    parser.add_argument('--data_dir', type=str, default="./aliccp/aliccp_out/", help="Data directory")
    parser.add_argument('--dt_dir', type=str, default="", help="Data dt partition")
    parser.add_argument('--model_dir', type=str, default=f"./",
                        help="Model checkpoint directory")
    parser.add_argument('--servable_model_dir', type=str, default=f"./",
                        help="Export servable model for pytorch Serving")
    parser.add_argument('--clear_existing_model', action="store_true", help="Clear existing model or not")
    parser.add_argument('--max_seq_len', type=int, default=50, help="Max length of sequence")
    parser.add_argument('--task_num', type=int, default=2, help="Task number")
    parser.add_argument('--experts_num', type=int, default=8, help="Number of experts")
    parser.add_argument('--log_level', type=str, default="DEBUG",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Log level")
    parser.add_argument('--epoch_num', type=int, default=10, help="Number of epochs")
    parser.add_argument('--train_batch_num', type=int, default=2000, help="Number of train batchs")
    parser.add_argument('--eval_batch_num', type=int, default=20, help="Number of eval batchs")
    parser.add_argument('--test_batch_num', type=int, default=20, help="Number of test batchs")
    parser.add_argument("--enable_compile", action="store_true", help="")
    parser.add_argument("--max_steps", type=int, default=100, help="Total training steps")
    parser.add_argument("--enable_profiler", action="store_true",
                       help="Enable profiler for performance analysis")
    parser.add_argument("--profiler_start_step", type=int, default=5,
                       help="Output directory for trained model")
    parser.add_argument("--profiler_end_step", type=int, default=8,
                    help="Output directory for trained model")
    parser.add_argument("--profiler_save_path", type=str, default="./profile",
                    help="Output directory for trained model")
    return parser.parse_args()


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


def pre_deal_dataset(dataset):
    num_pos = 0
    num_neg = 0
    num_pos_y = 0
    num_neg_y = 0
    dataset_index = 0
    for sub_ds in dataset.datasets:
        z = sub_ds.target_sample["z"]
        num_pos += (z == 1).sum().item()
        num_neg += (z == 0).sum().item()

        y = sub_ds.target_sample["y"]
        num_pos_y += (y == 1).sum().item()
        num_neg_y += (y == 0).sum().item()

        # logger.info("find pos index %s", dataset_index)
        dataset_index += 1
    return np.log(num_neg_y / num_pos_y), np.log(num_neg / num_pos)


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
        self.positive_indices = np.where(y == 1)[0]
        self.negative_indices = np.where(y == 0)[0]
        # logger.info(f"load file {self.hdf5_path} finished")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        input_dict = {k: v[idx] for k, v in self.input_sample.items()}
        target_dict = {k: v[idx] for k, v in self.target_sample.items()}
        return input_dict, target_dict


class TorchDataSet(ConcatDataset):
    def __init__(self, files):
        print(files)
        datasets = [HDF5Dataset(fp) for fp in files]
        super().__init__(datasets)


class TorchMmoeModel(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.spec = json_file_load("spec", os.path.join(self.params.data_dir, "spec.json"))
        self.embedding_layers = self.build_embedding_layers()
        self.experts = self.build_experts()
        self.gates = self.build_gate_networks()
        self.task_output_layers = self.build_task_output_layers()

        self.tower_names = ['ctr', 'cvr']
        self.towers = nn.ModuleDict()
        self.towers_output_layers = nn.ModuleDict()
        tower_units = list(map(int, self.params.tower_layers.strip().split(',')))
        input_dim = self.params.experts_num * list(map(int, self.params.expert_layers.strip().split(',')))[-1]
        for name in self.tower_names:
            tower_layers = []
            in_dim = input_dim
            for out_dim in tower_units:
                tower_layers.append(nn.Linear(in_dim, out_dim))
                tower_layers.append(nn.BatchNorm1d(out_dim))
                tower_layers.append(nn.ReLU())
                tower_layers.append(nn.Dropout(p=0.2))
                in_dim = out_dim
            self.towers[name] = nn.Sequential(*tower_layers)
            self.towers_output_layers[name] = nn.Linear(in_dim, 1)

    def forward(self, features: dict):
        # Build the embedding layer
        x_deep = self.get_embedding(features)
        experts_out = [expert(x_deep) for expert in self.experts]
        experts_out = torch.stack(experts_out, dim=1)

        gate_outs = [gate(x_deep) for gate in self.gates]

        task_outputs = []
        for gate_network in gate_outs:
            gate_network = gate_network.unsqueeze(-1)
            task_out = torch.multiply(experts_out, gate_network)
            task_out_shape = list(task_out.shape)
            task_outputs.append(torch.reshape(task_out, shape=[-1, task_out_shape[1] * task_out_shape[2]]))
        return task_outputs

    def build_embedding_layers(self):
        embeddings = nn.ModuleDict()
        for key, vocab_len in self.spec["vocab_length"].items():
            embeddings[key] = nn.Embedding(vocab_len + 1, self.params.embedding_size)
            nn.init.normal_(embeddings[key].weight, std=STD_DEV)
        return embeddings

    def build_experts(self):

        experts = nn.ModuleList()
        expert_units = list(map(int, self.params.expert_layers.strip().split(',')))
        input_dim = self.params.embedding_size * EMBEDDING_FEATURE_NUM

        for _ in range(self.params.experts_num):
            expert_layers = []
            in_features = input_dim

            for out_features in expert_units:
                expert_layers.append(nn.Linear(in_features, out_features))
                expert_layers.append(nn.BatchNorm1d(out_features))
                expert_layers.append(nn.ReLU())
                expert_layers.append(nn.Dropout(0.2))
                in_features = out_features
            experts.append(nn.Sequential(*expert_layers))
        return experts

    def build_gate_networks(self):
        gates = nn.ModuleList()
        input_dim = self.params.embedding_size * EMBEDDING_FEATURE_NUM

        for _ in range(self.params.task_num):
            gate = nn.Sequential(
                nn.Linear(input_dim, self.params.experts_num),
                nn.Softmax(dim=1)
            )
            gates.append(gate)
        return gates

    def build_task_output_layers(self):
        task_output_layers = nn.ModuleList()
        expert_units = list(map(int, self.params.expert_layers.strip().split(',')))
        input_dim = self.params.experts_num * expert_units[-1]
        tower_units = list(map(int, self.params.tower_layers.strip().split(',')))
        for _ in range(self.params.task_num):
            tower = []
            in_features = input_dim
            for out_features in tower_units:
                tower.append(nn.Linear(in_features, out_features))
                tower.append(nn.ReLU())
                in_features = out_features
            task_output_layers.append(nn.Sequential(*tower))
        return task_output_layers

    def embedding_lookup_sparse_fake(self, key,
                                     ids: torch.Tensor,
                                     combiner: str = None,
                                     name: str = None) -> torch.Tensor:
        dense_mask = torch.unsqueeze(torch.where(ids >= 0,
                                                 torch.ones_like(ids, dtype=torch.float32),
                                                 torch.zeros_like(ids)
                                                 ),
                                     dim=-1
                                     )

        # Replace invalid IDs (-1) with zeros
        ids = torch.where(ids == -1, torch.zeros_like(ids), ids)
        embedding_layer = self.embedding_layers[key]
        embedding_output = embedding_layer(ids)
        embedding = embedding_output * dense_mask
        summed_embedding = torch.sum(embedding, axis=1)
        if combiner == "sum":
            return summed_embedding
        elif combiner == "mean":
            return summed_embedding / torch.sum(dense_mask, axis=1)
        else:
            raise ValueError("combiner only supoort 'sum', 'mean'")

    def get_embedding(self, features: dict) -> torch.Tensor:
        """
        Build the embedding layer for the model.

        Args:
            features (dict): The input features.
        Returns:
            torch.Tensor: The concatenated and reshaped embedding tensor.
        """
        on_hot_field_lst = self.spec.get("one_hot_fields")
        other_field_lst = self.spec.get("multi_hot_fields") + self.spec.get("special_fields")

        embeddings = {}
        for key in on_hot_field_lst:
            embedding_layer = self.embedding_layers[key]
            embeddings[key] = embedding_layer(features[key])
            embeddings[key] = torch.reshape(embeddings[key], [-1, 1, self.params.embedding_size])
        for key in other_field_lst:
            embeddings[key] = torch.unsqueeze(
                self.embedding_lookup_sparse_fake(key=key, ids=features[key], combiner="sum"),
                dim=1
            )
        embedding = torch.concat(
            [embeddings.get(field_name) for field_name in self.spec.get("one_hot_fields")] +
            [embeddings.get(field_name) for field_name in self.spec.get("multi_hot_fields")] +
            [embeddings.get(field_name) for field_name in self.spec.get("special_fields")],
            dim=2,
        )

        return torch.reshape(embedding, [-1, EMBEDDING_FEATURE_NUM * self.params.embedding_size])

    def build_tower(self, tower_input: torch.Tensor, name: str) -> torch.Tensor:
        """
        Build the tower network for a specific task.

        Args:
            tower_input (torch.Tensor): The input tensor for the tower.
            name (str): The name of the tower.
        Returns:
            torch.Tensor: The output tensor of the tower network.
        """
        tower_units = list(map(int, self.params.tower_layers.strip().split(',')))
        y_tower = tower_input
        for tower_i, _ in enumerate(tower_units):
            tower_linear = nn.Linear(in_features=y_tower.shape[-1], out_features=tower_units[tower_i])
            tower_linear_out = tower_linear(y_tower)
            relu = nn.ReLU()
            y_tower = relu(tower_linear_out)
        return y_tower

    def build_predictions(self, task_outputs: list) -> dict:
        """
        Build the predictions for the model.

        Args:
            task_outputs (list): A list of task output tensors.

        Returns:
            dict: A dictionary containing the predictions for ctr, cvr, and ctcvr.
        """
        preds = {}
        tower_outputs = {}
        for i, name in enumerate(self.tower_names):
            y = self.towers[name](task_outputs[i])
            y = self.towers_output_layers[name](y)
            y = torch.reshape(y, [-1, ])
            preds[name + '_logit'] = y
            preds[name] = torch.sigmoid(y)
            tower_outputs[name] = y

        ctr_pred = preds.get('ctr')
        cvr_pred = preds.get('cvr')
        if ctr_pred is not None and cvr_pred is not None:
            if 'ctr_logit' not in preds or 'cvr_logit' not in preds:
                raise ValueError("Missing required keys in preds dictionary")
            else:
                preds['ctcvr'] = ctr_pred * cvr_pred
                preds['ctcvr_logit'] = preds['ctr_logit'] + preds['cvr_logit']
        return preds

    def build_loss(self,
                   labels: dict,
                   y_ctr_logit: torch.Tensor,
                   y_ctcvr_logit: torch.Tensor,
                   ctr_weight,
                   ctcvr_weight) -> torch.Tensor:
        """
        Build the loss function for the model.

        Args:
            labels (dict): A dictionary containing the true labels for ctr and ctcvr.
            y_ctr_logit (torch.Tensor): The predicted ctr values.
            y_ctcvr_logit (torch.Tensor): The predicted ctcvr values.
        Returns:
            torch.Tensor: The combined loss tensor.
        """
        y_ctr_logit = y_ctr_logit.view(-1)
        y_ctcvr_logit = y_ctcvr_logit.view(-1)
        y = labels['y'].view(-1)
        z = labels['z'].view(-1)

        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ctr_weight],
                                                                dtype=torch.float32,
                                                                device=y.device)
                                        )
        ctcvr_bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ctcvr_weight],
                                                                      dtype=torch.float32,
                                                                      device=z.device)
                                              )

        ctr_loss = bce_loss(y_ctr_logit, y)
        ctcvr_loss = ctcvr_bce_loss(y_ctcvr_logit, z)
        ctr_task_wgt = self.params.ctr_task_wgt

        return ctr_task_wgt * ctr_loss + (1 - ctr_task_wgt) * ctcvr_loss

    def build_optimizer(self):
        """
        Build the optimizer for training.

        Args:
            loss (torch.Tensor): The loss tensor to minimize.

        Returns:
           optim: The operation for applying gradients.

        Raises:
            ValueError: If the optimizer type is not supported.
        """
        if self.params.optimizer == "Adam":
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.params.learning_rate,
                betas=[0.9, 0.99], eps=1e-8
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


def evaluate_op(model: TorchMmoeModel, te_files, device, args):
    te_dataset = TorchDataSet(te_files)
    te_ctr, te_ctcvr = pre_deal_dataset(te_dataset)
    test_dataloader = DataLoader(dataset=te_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               collate_fn=collate_fn,
                               prefetch_factor=100,
                               num_workers=10)
    dl = iter(test_dataloader)
    input_sample, target_sample = next(dl)
    input_sample = {k: v.to(device) for k, v in input_sample.items()}

    mod = 'compile' if args.enable_compile else 'eager'
    if mod == 'compile':
        model = torch.compile(model, mode='reduce-overhead', dynamic=False)

    model.eval()
    with torch.no_grad():
        model(input_sample) 
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
            model(input_sample)
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
    model = TorchMmoeModel(args)
    te_files = glob.glob("%stest/data_test.csv.hd5.*" % args.data_dir)
    device_type = detect_device_type()
    print(f"Using device type {device_type}")
    device = torch.device(device_type)
    model.to(device)
    evaluate_op(model, te_files, device, args)


if __name__ == "__main__":
    print(f"{MODEL_NAME} inference begin !!!")
    args = parse_arguments()
    main(args)