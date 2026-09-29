# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause

"""Compatibility check for PyTorch's custom process-group delegation (#188489)."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist


class TestNewGroupBackend(unittest.TestCase):
    @unittest.skipIf(
        tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) < (2, 14),
        "Custom PG subgroup delegation requires PyTorch 2.14 or newer",
    )
    def test_custom_pg_receives_default_and_explicit_backend(self):
        backend_name = "pta_new_group_backend_test"
        received_backends = []

        class DelegatingProcessGroup(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return backend_name

            def _get_backend(self, device):
                return self

            def new_group(
                self,
                ranks,
                timeout=None,
                backend=None,
                pg_options=None,
                group_name=None,
                group_desc=None,
            ):
                received_backends.append(str(backend))
                return DelegatingProcessGroup(ranks.index(0), len(ranks))

        dist.Backend.register_backend(
            backend_name,
            lambda opts, pg_options: DelegatingProcessGroup(
                opts.group_rank, opts.group_size
            ),
            extended_api=True,
            devices=["cpu"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            store = dist.FileStore(str(Path(temp_dir) / "store"), 1)
            dist.init_process_group(
                backend_name, store=store, rank=0, world_size=1
            )
            try:
                dist.new_group(ranks=[0])
                dist.new_group(ranks=[0], backend=f"cpu:{backend_name}")
                self.assertEqual(
                    received_backends,
                    [backend_name, f"cpu:{backend_name}"],
                )
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
