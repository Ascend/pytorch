import dataclasses
import os
import contextlib
import hashlib
import json
import logging
import subprocess
import sys
import sysconfig
from time import time, time_ns
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
from torch._inductor.exc import CppCompileError
from torch._inductor.codecache import (
    CacheBase,
    get_lock_dir,
    write,
    LOCK_TIMEOUT,
    DLLWrapper,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import (
    clear_on_fresh_inductor_cache,
    is_linux,
    is_windows,
)
import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error

from .cpp_builder import library_paths
from . import config as npu_config
from .codegen.catlass.catlass_utils import get_npu_arch, _normalize_npu_arch

empty_json = "{}"

log = logging.getLogger("torch._inductor")


@contextlib.contextmanager
def lock_context(key):
    from filelock import FileLock
    lock_dir = get_lock_dir()
    lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
    with lock:
        yield



def patch_cache_base_get_system():
    # patch function CacheBase.get_system with get_system_npu, add logic to support CANN
    @staticmethod
    def get_system():
        try:
            from triton.compiler.compiler import triton_key

            # Use triton_key instead of triton.__version__ as the version
            # is not updated with each code change
            triton_version = triton_key()
        except ModuleNotFoundError:
            triton_version = None

        try:
            system: Dict[str, Any] = {
                "device": {"name": None},
                "version": {
                    "triton": triton_version,
                },
            }
            device_properties = torch_npu.npu.get_device_properties(
                torch_npu.npu.current_device()
            )
            if torch.version.cann is not None:
                system["device"]["name"] = device_properties.name
                system["version"]["cann"] = torch.version.cann
            elif torch.version.cuda is not None:
                system["device"]["name"] = device_properties.name
                system["version"]["cuda"] = torch.version.cuda
            else:
                system["device"]["name"] = device_properties.gcnArchName
                system["version"]["hip"] = torch.version.hip
        except (AssertionError, RuntimeError):
            # If deivce is not installed, none of the above config is relevant.
            system = {}

        system["hash"] = hashlib.sha256(
            json.dumps(system, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return system

    CacheBase.get_system = get_system


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
        wrapper_code: str,
        kernel_code: str,
        serialized_extern_kernel_nodes: Optional[str],
        *,
        device_type: str,
        additional_files: list[str],
    ) -> Union[List[str], str]:
        result = cls.src_compile(
            graph,
            wrapper_code,
            kernel_code,
            serialized_extern_kernel_nodes,
            device_type=device_type,
            additional_files=additional_files,
        )
        generated_files = additional_files
        if not config.aot_inductor.package:
            return result

        output_so = [r for r in result if r.endswith(".so")]
        if len(output_so) > 1:
            raise RuntimeError(
                f"Could not generate npu op json, because there are"
                f"more than one so in generated files: {result}"
                + pta_error(ErrCode.INTERNAL)
            )
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


def _catlass_include_paths() -> List[str]:
    from .cpp_builder import get_ascend_home

    ASCEND_HOME = get_ascend_home()
    catlass_path = npu_config.catlass.catlass_dir
    return [
        # Use realpath to get canonical absolute paths, in order not to mess up cache keys
        os.path.realpath(os.path.join(ASCEND_HOME, "compiler/tikcpp")),
        os.path.realpath(os.path.join(ASCEND_HOME, "compiler/tikcpp/tikcfw")),
        os.path.realpath(os.path.join(ASCEND_HOME, "compiler/tikcpp/tikcfw/impl")),
        os.path.realpath(os.path.join(ASCEND_HOME, "compiler/tikcpp/tikcfw/interface")),
        os.path.realpath(os.path.join(ASCEND_HOME, "include")),
        os.path.realpath(os.path.join(ASCEND_HOME, "include/experiment/runtime")),
        os.path.realpath(os.path.join(ASCEND_HOME, "include/experiment/msprof")),
        os.path.realpath(os.path.join(catlass_path, "include")),
    ]


def _ascend_lib_options() -> List[str]:
    lpaths = library_paths(npu=True) + [sysconfig.get_config_var("LIBDIR")]
    extra_ldflags: List[str] = []
    if is_linux():
        for path in lpaths:
            # -rpath ensures the DLL can find its dependencies when loaded, even
            # if the library path is non-standard.
            extra_ldflags.extend([f"-L{path}", "-Xlinker", f"-rpath={path}"])

        extra_ldflags.append("-lruntime")
        extra_ldflags.append("-lstdc++")
        extra_ldflags.append("-lascendcl")
        extra_ldflags.append("-lm")
        extra_ldflags.append("-ltiling_api")
        extra_ldflags.append("-lplatform")
        extra_ldflags.append("-lc_sec")
        extra_ldflags.append("-ldl")
        extra_ldflags.append("-lnnopbase")
    else:
        raise NotImplementedError(
            "Unsupported env, failed to find ascend libs! Currently only Linux is supported."
        )
    return extra_ldflags


def _bisheng_host_compiler_options() -> List[str]:
    return [
        "-fPIC",
        "-fno-strict-aliasing",
        "-fvisibility=hidden",
        "-Wconversion",
    ]


def _bisheng_compiler_options(is_mix: bool = False) -> List[str]:
    npu_arch = _normalize_npu_arch(get_npu_arch())
    if npu_arch == "910B":
        arch = "dav-c220"
    elif npu_arch == "910D":
        arch = "dav-c310"
    else:
        raise ValueError(f"Unrecognized NPU arch: {npu_arch}")

    if not is_mix:
        # pure cube kernel
        arch += '-cube'

    options = [
        f"--cce-aicore-arch={arch}",
        "-O2",
        "-std=c++17",
        "-xcce",
        "-DL2_CACHE_HINT",
    ]
    if npu_arch == "910D":
        options.append("-DCATLASS_ARCH_A5_ENABLED")
    elif npu_arch == "910B":
        options.append("-DCATLASS_ARCH_A2_ENABLED")

    if npu_config.catlass.enable_debug_info:
        options.extend(["--lineinfo", "-g"])

    return options


def _bisheng_compiler() -> Optional[str]:
    if os.path.exists(os.getenv("ASCEND_HOME_PATH")):
        return os.path.realpath(
            os.path.join(
                os.getenv("ASCEND_HOME_PATH", ""), "tools/ccec_compiler/bin/bisheng"
            )
        )
    return "bisheng"


def catlass_compile_command(
    src_files: List[str],
    dst_file: str,
    dst_file_ext: str,
    extra_args: Optional[List[str]] = None,
    is_mix: bool = False,
) -> str:
    if extra_args is None:
        extra_args = []
    include_paths = _catlass_include_paths()
    ascend_lib_options = _ascend_lib_options()
    bisheng_host_compiler_options = _bisheng_host_compiler_options()
    bisheng_compiler_options = _bisheng_compiler_options(is_mix)
    options = (
        bisheng_compiler_options
        + extra_args
        + [
            f"-Xcompiler {opt}" if "=" in opt else f"-Xcompiler={opt}"
            for opt in bisheng_host_compiler_options
        ]
        + ["-I" + path for path in include_paths]
        + ascend_lib_options
    )
    src_file = " ".join(src_files)
    res = ""
    if dst_file_ext == "o":
        res = f"{_bisheng_compiler()} {' '.join(options)} -c -o {dst_file} {src_file}"
    elif dst_file_ext == "so":
        options.append("-shared")
        res = f"{_bisheng_compiler()} {' '.join(options)} -o {dst_file} {src_file}"
    elif dst_file_ext == "exe":
        res = f"{_bisheng_compiler()} {' '.join(options)} -o {dst_file} {src_file}"
    else:
        raise NotImplementedError(f"Unsupported output file suffix {dst_file_ext}!")
    log.debug("Bisheng command: %s", res)
    return res


class NPUCompileError(CppCompileError):
    pass


@clear_on_fresh_inductor_cache
class CATLASSCodeCache:
    @dataclasses.dataclass
    class CacheEntry:
        input_path: str
        output_path: str

    cache: Dict[str, CacheEntry] = {}
    cache_clear = staticmethod(cache.clear)
    _SOURCE_CODE_SUFFIX = "cpp"

    @classmethod
    def write(cls, source_code: str, dst_file_ext: str, is_mix: bool) -> Tuple[str, str]:
        """
        Writes source code into a file with dst_file_ext as the file extension.
        Returns the hash key of source code, and the path to the file.
        """

        catlass_command = repr(
            catlass_compile_command(["dummy_input"], "dummy_output", dst_file_ext, is_mix=is_mix)
        )
        key, input_path = write(source_code, cls._SOURCE_CODE_SUFFIX, extra=catlass_command)
        return key, input_path

    @classmethod
    def compile(
        cls, source_code: str, dst_file_ext: str, extra_args: Optional[List[str]] = None, is_mix: bool = False
    ) -> Tuple[str, str, str]:
        """
        Compiles CATLASS source_code into a file with dst_file_ext extension.
        Returns a tuple of dst_file_path, hash_key, source_code_path
        """
        key, input_path = cls.write(source_code, dst_file_ext, is_mix)
        if key not in cls.cache:
            from filelock import FileLock

            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
            with lock:
                output_path = input_path[: -len(cls._SOURCE_CODE_SUFFIX)] + dst_file_ext
                if not os.path.exists(output_path):
                    cmd = catlass_compile_command(
                        [input_path], output_path, dst_file_ext, extra_args, is_mix
                    )
                    start_time = time()
                    log.debug("CATLASS Compilation: %s", cmd)
                    cmd_parts = cmd.split(" ")
                    try:
                        subprocess.check_output(
                            cmd_parts, stderr=subprocess.STDOUT, env=os.environ
                        )
                    except subprocess.CalledProcessError as error:
                        raise NPUCompileError(cmd_parts, error.output) from error
                    end_time = time()
                    log_duration_msg = f"CATLASS Compilation took {end_time - start_time} seconds. Compile command: {cmd}"
                    log.info(log_duration_msg)
                else:
                    log.debug(
                        "CATLASS Compilation skipped: %s since output already exists",
                        input_path,
                    )
                cls.cache[key] = CATLASSCodeCache.CacheEntry(input_path, output_path)

        return (cls.cache[key].output_path, key, input_path)

    @classmethod
    def load(cls, source_code: str, dst_file_ext: str, is_mix: bool = False) -> Tuple[DLLWrapper, str, str]:
        """
        Compiles source code and loads the generated .so file.
        Returns a tuple of DLLWrapper, hash_key, source_code_path
        """

        if dst_file_ext != "so":
            raise RuntimeError(
                f"Only support loading a .so file for now. "
                f"Requested file extension: {dst_file_ext}. Source code: {source_code}"
            )
        dst_file_path, hash_key, source_code_path = cls.compile(
            source_code, dst_file_ext, is_mix=is_mix
        )
        return (DLLWrapper(dst_file_path), hash_key, source_code_path)