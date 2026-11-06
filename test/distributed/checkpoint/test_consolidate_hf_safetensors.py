# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NPU validation cases for torch.distributed.checkpoint._consolidate_hf_safetensors APIs.
Covers: consolidate_safetensors_files_on_every_rank, consolidate_safetensors_files,
_calculate_max_contiguous_elements, _write_sub_tensor_to_file_optimized.
"""
import json
import os

import safetensors
import torch
import torch.distributed.checkpoint as dist_cp
import torch_npu
from torch import distributed as dist
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    _calculate_max_contiguous_elements,
    _write_sub_tensor_to_file_optimized,
    consolidate_safetensors_files,
    consolidate_safetensors_files_on_every_rank,
)
from torch.distributed.checkpoint._hf_utils import _metadata_fn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU, with_comms

_GLOBAL_SHAPE = (6, 8)
_TOTAL_BYTES = 6 * 8 * 4 * 2

# (indices, sub_tensor_shape, tensor_shape) -> expected contiguous element count
_CONTIGUOUS_CASES = [
    (([2], [5], [10]), 3),
    (([1, 0], [3, 4], [6, 4]), 8),
    (([1, 2], [3, 4], [6, 8]), 2),
    (([1, 0, 0], [3, 2, 4], [5, 2, 4]), 16),
    (([2, 3], [3, 4], [6, 8]), 1),
    (([0, 0], [2, 5], [4, 5]), 10),
    (([1, 0], [3, 2], [4, 8]), 2),
    (([0, 0], [2, 3], [6, 10]), 3),
]

# (indices, sub_tensor_shape, tensor_shape) -> expected error message pattern
_CONTIGUOUS_ERROR_CASES = [
    (([], [3], [5]), "Input lists cannot be empty"),
    (([1], [3, 4], [6, 5]), "All input lists must have the same length"),
    (([7, 0], [3, 4], [6, 5]), "Index .* at dimension .* is out of bounds"),
    (([0, 0], [3, 9], [6, 5]), "Sub-tensor dimension .* at position .* exceeds"),
]


class TestNPUConsolidateHFSafeTensors(DTensorTestBase):
    """NPU cases for the HF safetensors consolidation helpers."""

    device_type = "npu"
    backend = "hccl"

    @property
    def world_size(self) -> int:
        # 固定 2 进程：全局 shape (6, 8) 按 2 整除分片；CI 默认 4 进程时会越界
        return 2

    def init_pg(self, *args, **kwargs) -> None:
        self.device_type = type(self).device_type
        torch_npu.npu.set_device(self.rank)
        super().init_pg(*args, **kwargs)

    def _spawn_shards(self, ckpt_dir, row_name, col_name):
        """Each rank saves its row/column shard of one shared tensor via the HF writer."""
        mesh = init_device_mesh("npu", (dist.get_world_size(),))
        local_rank = mesh.get_local_rank()
        torch.manual_seed(2026)
        full = torch.randn(_GLOBAL_SHAPE, dtype=torch.float32).npu()
        row_local = full.chunk(dist.get_world_size(), dim=0)[local_rank].contiguous()
        col_local = full.chunk(dist.get_world_size(), dim=1)[local_rank].contiguous()
        row_dt = DTensor.from_local(
            row_local, device_mesh=mesh, placements=[Shard(0)],
            shape=_GLOBAL_SHAPE, stride=full.stride(),
        )
        col_dt = DTensor.from_local(
            col_local, device_mesh=mesh, placements=[Shard(1)],
            shape=_GLOBAL_SHAPE, stride=full.stride(),
        )
        dist_cp.save(
            state_dict={row_name: row_dt, col_name: col_dt},
            storage_writer=dist_cp.HuggingFaceStorageWriter(
                path=ckpt_dir, save_distributed=True
            ),
        )
        dist.barrier()
        os.sync()
        return full

    def _load_part(self, out_dir, name):
        return safetensors.torch.load_file(os.path.join(out_dir, name))

    def _assert_full_tensor(self, loaded, key, full):
        self.assertEqual(list(loaded.keys()), [key])
        self.assertTrue(torch.equal(loaded[key].npu(), full))

    @with_comms
    @with_temp_dir
    @skipIfUnsupportMultiNPU(2)
    def test_single_file_merge(self):
        out = os.path.join(self.temp_dir, "merged")
        os.makedirs(out, exist_ok=True)
        full = self._spawn_shards(self.temp_dir, "w_row", "w_col")

        if dist.get_rank() == 0:
            consolidate_safetensors_files(
                self.temp_dir, out, fqn_to_index_mapping={"w_row": 1, "w_col": 1}
            )
            merged = self._load_part(out, "model-00001-of-00001.safetensors")
            self.assertEqual(sorted(merged.keys()), ["w_col", "w_row"])
            self.assertTrue(torch.equal(merged["w_row"].npu(), full))
            self.assertTrue(torch.equal(merged["w_col"].npu(), full))
            meta = self._read_meta(out)
            self.assertEqual(meta["metadata"]["total_size"], _TOTAL_BYTES)
            self.assertEqual(
                set(meta["weight_map"].values()), {"model-00001-of-00001.safetensors"}
            )
        dist.barrier()

    @with_comms
    @with_temp_dir
    @skipIfUnsupportMultiNPU(2)
    def test_multi_file_merge(self):
        out = os.path.join(self.temp_dir, "merged")
        os.makedirs(out, exist_ok=True)
        full = self._spawn_shards(self.temp_dir, "alpha", "beta")

        if dist.get_rank() == 0:
            consolidate_safetensors_files(
                self.temp_dir, out, fqn_to_index_mapping={"alpha": 1, "beta": 2}
            )
            part1 = self._load_part(out, "model-00001-of-00002.safetensors")
            part2 = self._load_part(out, "model-00002-of-00002.safetensors")
            self._assert_full_tensor(part1, "alpha", full)
            self._assert_full_tensor(part2, "beta", full)
            meta = self._read_meta(out)
            self.assertEqual(meta["metadata"]["total_size"], _TOTAL_BYTES)
            self.assertEqual(meta["weight_map"]["alpha"], "model-00001-of-00002.safetensors")
            self.assertEqual(meta["weight_map"]["beta"], "model-00002-of-00002.safetensors")
        dist.barrier()

    @with_comms
    @with_temp_dir
    @skipIfUnsupportMultiNPU(2)
    def test_every_rank_merge_two_files(self):
        out = os.path.join(self.temp_dir, "merged")
        os.makedirs(out, exist_ok=True)
        full = self._spawn_shards(self.temp_dir, "p", "q")

        consolidate_safetensors_files_on_every_rank(
            self.temp_dir, out, fqn_to_index_mapping={"p": 1, "q": 2}
        )
        part1 = self._load_part(out, "model-00001-of-00002.safetensors")
        part2 = self._load_part(out, "model-00002-of-00002.safetensors")
        self._assert_full_tensor(part1, "p", full)
        self._assert_full_tensor(part2, "q", full)
        self.assertTrue(os.path.exists(os.path.join(out, _metadata_fn)))
        dist.barrier()

    @with_comms
    @with_temp_dir
    @skipIfUnsupportMultiNPU(2)
    def test_every_rank_merge_single_file(self):
        out = os.path.join(self.temp_dir, "merged")
        os.makedirs(out, exist_ok=True)
        full = self._spawn_shards(self.temp_dir, "x", "y")

        consolidate_safetensors_files_on_every_rank(
            self.temp_dir, out, fqn_to_index_mapping={"x": 1, "y": 1}
        )
        merged = self._load_part(out, "model-00001-of-00001.safetensors")
        self.assertEqual(sorted(merged.keys()), ["x", "y"])
        self.assertTrue(torch.equal(merged["x"].npu(), full))
        self.assertTrue(torch.equal(merged["y"].npu(), full))
        dist.barrier()

    def _read_meta(self, out_dir):
        with open(os.path.join(out_dir, _metadata_fn)) as fh:
            return json.load(fh)

    def test_contiguous_elements_errors(self):
        fn = _calculate_max_contiguous_elements
        for (indices, sub_shape, tensor_shape), pattern in _CONTIGUOUS_ERROR_CASES:
            with self.assertRaisesRegex(ValueError, pattern):
                fn(indices, sub_shape, tensor_shape)

    def test_contiguous_elements_values(self):
        fn = _calculate_max_contiguous_elements
        for (indices, sub_shape, tensor_shape), want in _CONTIGUOUS_CASES:
            self.assertEqual(fn(indices, sub_shape, tensor_shape), want)

    def _run_writer_case(self, full_shape, offsets, sub_shape):
        elem = 4
        buf = bytearray(full_shape[0] * full_shape[1] * elem)
        data = torch.randn(sub_shape[0] * sub_shape[1], dtype=torch.float32)
        _write_sub_tensor_to_file_optimized(
            memoryview(buf), data.numpy().tobytes(), elem,
            full_shape, offsets, sub_shape,
        )
        got = torch.frombuffer(bytearray(buf), dtype=torch.float32).view(full_shape)
        return got, data

    def test_write_sub_tensor_row_region(self):
        got, data = self._run_writer_case([2, 5], [1, 0], [1, 5])
        want = torch.zeros(2, 5, dtype=torch.float32)
        want[1:2, :] = data.view(1, 5)
        self.assertTrue(torch.equal(got, want))

    def test_write_sub_tensor_col_region(self):
        got, data = self._run_writer_case([3, 4], [0, 1], [3, 2])
        want = torch.zeros(3, 4, dtype=torch.float32)
        want[:, 1:3] = data.view(3, 2)
        self.assertTrue(torch.equal(got, want))


if __name__ == "__main__":
    run_tests()
