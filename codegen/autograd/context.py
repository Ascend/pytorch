# Copyright (c) 2023 Huawei Technologies Co., Ltd
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

import functools
from typing import Callable

from codegen.api.autograd import NativeFunctionWithDifferentiabilityInfo as NFWDI
from codegen.context import native_function_manager
from codegen.utils import T


# Like tools.api.context.with_native_function, but for
# NativeFunctionWithDifferentiabilityInfo.
def with_native_function_with_differentiability_info(func: Callable[[NFWDI], T]) -> Callable[[NFWDI], T]:
    @functools.wraps(func)
    def wrapper(f: NFWDI) -> T:
        with native_function_manager(f.func):
            return func(f)
    return wrapper
