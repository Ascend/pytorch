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

from .module import apply_module_patch
from .tensor_methods import add_tensor_methods
from .torch_funcs import add_torch_funcs
from .serialization import save, load
from ._tensor_str import add_str_methods
from .dataloader import add_dataloader_method
from .utils import manual_seed, seed
from .storage import add_storage_methods

serialization_patches = [
    ["save", save],
    ["load", load],
    ["random.seed", seed],
    ["random.manual_seed", manual_seed],
    ["seed", seed],
    ["manual_seed", manual_seed]
]
