# Copyright (c) 2023, Huawei Technologies.
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

from enum import Enum


class FileTag(Enum):
    # pytorch file tag
    TORCH_OP = 1
    OP_MARK = 2
    MEMORY = 3
    GC_RECORD = 6
    PYTHON_TRACER_FUNC = 7
    PYTHON_TRACER_HASH = 8
