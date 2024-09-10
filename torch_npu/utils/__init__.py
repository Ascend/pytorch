# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

from torch.serialization import register_package

import torch_npu

from .module import apply_module_patch
from .tensor_methods import add_tensor_methods
from .torch_funcs import add_torch_funcs
from .serialization import save, load, _npu_tag, _npu_deserialize, save_async
from ._tensor_str import add_str_methods
from .dataloader import add_dataloader_method
from .utils import manual_seed, seed, print_error_log, print_warn_log, print_info_log
from .storage import add_storage_methods
from .fx import add_fx_methods
from .checkpoint import add_checkpoint_methods
from .combine_tensors import npu_combine_tensors, get_part_combined_tensor, is_combined_tensor_valid
from .launch import add_launch_methods
from .collect_env import add_collect_env_methods, get_cann_version
from .asd_detector import set_asd_loss_scale, register_asd_hook
from ._step import add_asd_patch

__all__ = ["save_async"]


serialization_patches = [
    ["save", save],
    ["load", load],
    ["random.seed", seed],
    ["random.manual_seed", manual_seed],
    ["seed", seed],
    ["manual_seed", manual_seed]
]

register_package(30, _npu_tag, _npu_deserialize)


# init flopcount
if not torch_npu._C._flops_count_init():
    raise RuntimeError("flopcount initialization failed" + prof_error(ErrCode.UNAVAIL))
