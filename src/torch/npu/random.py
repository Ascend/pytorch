# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
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

import torch
from . import _lazy_init, _lazy_call, device_count, current_device

__all__ = ['manual_seed', 'manual_seed_all',
           'seed', 'seed_all', 'initial_seed']


def manual_seed(seed):
    r"""Sets the seed for generating random numbers for the current NPU.
    It's safe to call this function if NPU is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-NPU model, this function is insufficient
        to get determinism.  To seed all NPUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)

    def cb():
        idx = current_device()
        default_generator = torch.npu.default_generators[idx]
        default_generator.manual_seed(seed)

    _lazy_call(cb)


def manual_seed_all(seed):
    r"""Sets the seed for generating random numbers on all NPUs.
    It's safe to call this function if NPU is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb():
        for i in range(device_count()):
            default_generator = torch.npu.default_generators[i]
            default_generator.manual_seed(seed)

    _lazy_call(cb)


def seed():
    r"""Sets the seed for generating random numbers to a random number for the current NPU.
    It's safe to call this function if NPU is not available; in that
    case, it is silently ignored.

    .. warning::
        If you are working with a multi-NPU model, this function will only initialize
        the seed on one NPU.  To initialize all NPUs, use :func:`seed_all`.
    """
    def cb():
        idx = current_device()
        default_generator = torch.npu.default_generators[idx]
        default_generator.seed()

    _lazy_call(cb)


def seed_all():
    r"""Sets the seed for generating random numbers to a random number on all NPUs.
    It's safe to call this function if NPU is not available; in that
    case, it is silently ignored.
    """
    def cb():
        random_seed = 0
        seeded = False
        for i in range(device_count()):
            default_generator = torch.npu.default_generators[i]
            if not seeded:
                default_generator.seed()
                random_seed = default_generator.initial_seed()
                seeded = True
            else:
                default_generator.manual_seed(random_seed)

    _lazy_call(cb)


def initial_seed():
    r"""Returns the current random seed of the current NPU.

    .. warning::
        This function eagerly initializes NPU.
    """
    _lazy_init()
    idx = current_device()
    default_generator = torch.npu.default_generators[idx]
    return default_generator.initial_seed()
