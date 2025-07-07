import os
import contextlib
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    List,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import torch
from torch._inductor import config
from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT
from torch._inductor.graph import GraphLowering

from torch_npu.utils._error_code import ErrCode, pta_error

empty_json = "{}"


@contextlib.contextmanager
def lock_context(key):
    from filelock import FileLock
    lock_dir = get_lock_dir()
    lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
    with lock:
        yield


def patch_aot_code_compiler_compile():
    # In v2.6.0, aoti has bug when init oss_proxy_executor with default op_json,
    # which could not be skipped, so here we try to create a new npu op_json,
    # and clear the content of default op_json.
    from torch._inductor.codecache import AotCodeCompiler
    AotCodeCompiler.src_compile = AotCodeCompiler.compile

    @classmethod
    def compile_npu(
        cls,
        graph: GraphLowering,
        source_code: str,
        serialized_extern_kernel_nodes: Optional[str],
        device_type: str,
        additional_files: List[str],
    ) -> Union[List[str], str]:
        result = cls.src_compile(
            graph, source_code, serialized_extern_kernel_nodes,
            device_type, additional_files
        )
        generated_files = additional_files
        if not config.aot_inductor.package:
            return result
        
        output_so = [r for r in result if r.endswith(".so")]
        if len(output_so) > 1:
            raise RuntimeError(f"Could not generate npu op json, because there are"
                               f"more than one so in generated files: {result}" + pta_error(ErrCode.INTERNAL))
        output_so = output_so[0]
        key = os.path.basename(output_so)[0].replace(".", "_")
        dir_basename = os.path.splitext(output_so)[0]
        with lock_context(key):
            if serialized_extern_kernel_nodes:
                extern_kernel_nodes_json = dir_basename + "_npu.json"
                with open(extern_kernel_nodes_json, "w") as f:
                    f.write(serialized_extern_kernel_nodes)
                generated_files.append(extern_kernel_nodes_json)
            
            if serialized_extern_kernel_nodes:
                source_json_file = dir_basename + ".json"
                with open(source_json_file, "w") as f:
                    f.write(empty_json)
        return generated_files
    AotCodeCompiler.compile = compile_npu
        