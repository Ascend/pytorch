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

from ..prof_common_func.file_tag import FileTag
from ..prof_bean.memory_use_bean import MemoryUseBean
from ..prof_bean.op_mark_bean import OpMarkBean
from ..prof_bean.torch_op_bean import TorchOpBean
from ..prof_bean._gc_record_bean import GCRecordBean
from ..prof_bean._python_tracer_hash_bean import PythonTracerHashBean
from ..prof_bean._python_tracer_func_bean import PythonTracerFuncBean


class FwkFileParserConfig:
    FILE_DISPATCH_MAP = {
        FileTag.TORCH_OP: r"^torch\.op_range",
        FileTag.OP_MARK: r"^torch\.op_mark",
        FileTag.MEMORY: r"^torch\.memory_usage",
        FileTag.GC_RECORD: r"torch\.gc_record",
        FileTag.PYTHON_TRACER_FUNC: r"torch\.python_tracer_func",
        FileTag.PYTHON_TRACER_HASH: r"torch\.python_tracer_hash",
    }

    FILE_BEAN_MAP = {
        FileTag.TORCH_OP: {"bean": TorchOpBean, "is_tlv": True, "struct_size": 57},
        FileTag.OP_MARK: {"bean": OpMarkBean, "is_tlv": True, "struct_size": 40},
        FileTag.MEMORY: {"bean": MemoryUseBean, "is_tlv": True, "struct_size": 75},
        FileTag.GC_RECORD: {"bean": GCRecordBean, "is_tlv": False, "struct_size": 24},
        FileTag.PYTHON_TRACER_FUNC: {"bean": PythonTracerFuncBean, "is_tlv": False, "struct_size": 33},
        FileTag.PYTHON_TRACER_HASH: {"bean": PythonTracerHashBean, "is_tlv": True, "struct_size": 8}
    }
