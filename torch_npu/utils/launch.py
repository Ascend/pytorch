# Copyright (c) 2023 Huawei Technologies Co., Ltd
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

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed.launcher.api
from torch.distributed.elastic.multiprocessing import Std

import torch_npu


@dataclass
class LaunchConfig:
    min_nodes: int
    max_nodes: int
    nproc_per_node: int
    run_id: str = ""
    role: str = "default_role"
    rdzv_endpoint: str = ""
    rdzv_backend: str = "etcd"
    rdzv_configs: Dict[str, Any] = field(default_factory=dict)
    rdzv_timeout: int = -1
    max_restarts: int = 3
    monitor_interval: float = 30.0
    start_method: str = "spawn"
    log_dir: Optional[str] = None
    redirects: Union[Std, Dict[int, Std]] = Std.NONE
    tee: Union[Std, Dict[int, Std]] = Std.NONE
    metrics_cfg: Dict[str, str] = field(default_factory=dict)
    local_addr: Optional[str] = None

    def __post_init__(self):
        default_timeout = 900
        if self.rdzv_timeout != -1:
            self.rdzv_configs["timeout"] = self.rdzv_timeout
        elif "timeout" not in self.rdzv_configs:
            self.rdzv_configs["timeout"] = default_timeout
        if self.rdzv_backend == 'static':
            self.rdzv_configs["rank"] = int(os.environ.get('NODE_RANK', self.rdzv_configs["rank"]))


def add_launch_methods():
    torch.distributed.launcher.api.LaunchConfig = LaunchConfig
