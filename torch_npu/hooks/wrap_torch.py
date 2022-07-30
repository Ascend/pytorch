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


import torch


def wrap_add(*args, **kwargs):
    torch._C._VariableFunctions.add(*args, *kwargs)


def wrap_sub(*args, **kwargs):
    torch._C._VariableFunctions.sub(*args, *kwargs)


def wrap_div(*args, **kwargs):
    torch._C._VariableFunctions.div(*args, *kwargs)


def wrap_mul(*args, **kwargs):
    torch._C._VariableFunctions.mul(*args, *kwargs)
