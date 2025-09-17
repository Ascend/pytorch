import functools
import re
import dataclasses
import os
import sys
from itertools import chain, count, zip_longest
from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING, Union
import sympy
import torch
from torch import dtype as torch_dtype
from torch._inductor import config
from torch._inductor.codecache import CudaKernelParamCache
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch._inductor.codegen.aoti_hipify_utils import maybe_hipify_code_wrapper
from torch._inductor.codegen.common import get_device_op_overrides
from torch._inductor.codegen.cpp_utils import cexpr, DTYPE_TO_CPP, DEVICE_TO_ATEN
from torch._inductor.codegen.triton_utils import should_unwrap_unspec_arg
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.codegen.wrapper import PythonWrapperCodegen, SymbolicCallArg
from torch._inductor.ir import IRNode, TensorBox, GraphPartitionSignature
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch._inductor.utils import DeferredLineBase, IndentedBuffer
from torch._inductor.virtualized import V
from torch._inductor.utils import _align, ALIGN_BYTES

from .. import config as npu_config
from ..config import npu_block as NPU_ALIGN_BYTES

if TYPE_CHECKING:
    from torch._inductor.graph import GraphLowering


def checkIfTrue(value, msg):
    if not value:
        raise RuntimeError(msg)
    return True


_cpp_string_literal_escapes = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\t": "\\t",
    "\r": "\\r",
}
_cpp_string_literal_pattern = re.compile(r'["\\\n\t\r]')


def cpp_string_literal(s: str) -> str:
    escaped = _cpp_string_literal_pattern.sub(
        lambda match: _cpp_string_literal_escapes[match.group(0)], s
    )
    return f'"{escaped}"'


@dataclasses.dataclass
class UnwrapUnspecArg:
    """Marker that we need to call .item() on the tensor"""

    dtype: torch_dtype


@dataclasses.dataclass
class DeferredNpuTritonCallWrapper:
    """
    When using cpp wrapper, GPU kernel load and launch needs to wait for Triton kernels
    to be tuned and stored as cubin files, so use a deferred generating the final wrapper around
    the triton kernel until right before the prefix is written.
    """

    wrapper_name: str
    kernel_name: str
    arg_types: list[Any]
    kernel_id: int

    def generate(self, wrapper):
        prefix = wrapper.prefix
        if self.kernel_name.startswith("multi_kernel_"):
            # MultiKernel will select one kernel after running the autotune block
            self.kernel_name = MultiKernelCall.lookup_choice(self.kernel_name)
        params = CudaKernelParamCache.get(self.kernel_name)
        def_args = params["def_args"]
        arg_types = self.arg_types
        inductor_meta = params["inductor_meta"]

        if "extra_launcher_args" in inductor_meta and len(def_args) > len(arg_types):
            # extra_launcher_args should already be in def_args
            arg_types = arg_types + [SymbolicCallArg] * len(
                inductor_meta["extra_launcher_args"]
            )

        if not V.graph.aot_mode:
            prefix.writeline(
                maybe_hipify_code_wrapper(
                    f"static {wrapper.device_codegen.cpp_kernel_type()} {self.kernel_name} = nullptr;"
                )
            )
            kernel_var_name = self.kernel_name
        else:
            kernel_var_name = f"kernels_.{self.kernel_name}"

        # tensors can be RAIIAtenTensorHandle or ConstantHandle, so make them template types
        template_types = [
            f"typename {name}_type_"
            for name, arg_type in zip(def_args, arg_types)
            if isinstance(arg_type, (torch_dtype, UnwrapUnspecArg))
        ]
        if V.graph.aot_mode:
            template_types.append("typename kernels_type_")
        if template_types:
            prefix.writeline(f"template <{', '.join(template_types)}>")
        prefix.writeline(f"static inline void {self.wrapper_name}(")
        with prefix.indent():
            for name, arg_type in zip(def_args, arg_types):
                if isinstance(arg_type, (torch_dtype, UnwrapUnspecArg)):
                    prefix.writeline(f"const {name}_type_& {name},")
                elif issubclass(arg_type, (SymbolicCallArg, sympy.Expr, int)):
                    prefix.writeline(f"int64_t {name},")
                elif arg_type is float:
                    prefix.writeline(f"float {name},")
                elif arg_type is bool:
                    prefix.writeline(f"bool {name},")
                else:
                    raise ValueError(f"Unexpected arg type {arg_type}")
            prefix.writeline(f"{wrapper.device_codegen.cpp_stream_type()} stream_,")
            if V.graph.aot_mode:
                prefix.writeline("kernels_type_& kernels_,")
            prefix.writeline(
                "const std::optional<std::string>& cubin_dir_ = std::nullopt"
            )
        prefix.writeline("){")
        with prefix.indent():
            self.generate_grid(prefix, inductor_meta, params)
            self.generate_load_kernel(prefix, kernel_var_name, params)
            self.generate_launch_kernel(prefix, wrapper, kernel_var_name, params)
        prefix.writeline("}")
        # Ensure the cubin file is included in the package
        V.graph.wrapper_code.additional_files.append(
            params[get_cpp_wrapper_cubin_path_name()]
        )

    def generate_grid(
        self,
        prefix: IndentedBuffer,
        inductor_meta: dict[str, Any],
        params: dict[str, Any],
    ):
        from ..npu_triton_heuristics import GridExprNpu

        numels = [arg for arg in params["call_args"] if "_numel" in arg]
        grid = GridExprNpu.from_meta_and_set_numel(
            inductor_meta, params["config"], numels, "cpp"
        )
        for line in grid.prefix:
            prefix.writeline(line)
        prefix.splice(
            f"""\
            uint32_t grid_0 = {grid.x_grid};
            uint32_t grid_1 = {grid.y_grid};
            uint32_t grid_2 = {grid.z_grid};
            """
        )
        prefix.writeline("if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;")

    def generate_load_kernel(self, prefix, kernel_var_name, params):
        prefix.writeline(f"if ({kernel_var_name} == nullptr) {{")
        with prefix.indent():
            load_kernel_args = [
                cpp_string_literal(params[get_cpp_wrapper_cubin_path_name()]),
                cpp_string_literal(params["mangled_name"]),
                str(params["shared_mem"]),
                "cubin_dir_",
            ]
            prefix.writeline(
                f"{kernel_var_name} = loadKernel({', '.join(load_kernel_args)}); "
            )
        prefix.writeline("}")

    def generate_launch_kernel(self, prefix, wrapper, kernel_var_name, params):
        triton_meta = params["triton_meta"]
        arg_type_lookup = dict(zip(params["def_args"], self.arg_types))
        # difference between Python and C++ wrapper: C++ wrapper strips out equal_to_1 constants
        call_args = [name for name in params["call_args"] if name not in triton_meta["constants"]]
        arg_types = [arg_type_lookup[name] for name in call_args]
        arg_signatures = [triton_meta["signature"][name] for name in call_args]
        call, call_args_str = wrapper.generate_args_decl(
            prefix,
            call_args,
            arg_types,
            arg_signatures,
            kernel_var_name,
            self.kernel_id,
        )
        prefix.writeline(f"{call_args_str}")

        prefix.writeline(r"launchKernel({}, {});".format(call, f'"{kernel_var_name}"'))


class CppWrapperNpu(CppWrapperCpu):
    """
    Generates cpp wrapper for running on NPU and calls CUDA kernels
    """

    def __init__(self) -> None:
        self.device = "npu"
        self.device_codegen = get_device_op_overrides(self.device)
        super().__init__()
        self.grid_id = count()
        self.visited_raii_handle = set()
        self.visited_handle_for_kernel_id = dict()
        self._triton_call_wrappers: dict[str, DeferredNpuTritonCallWrapper] = {}

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        # comment at CppWrapperCpu `codegen_subgraph` function.
        return CppWrapperNpu()

    def super_write_header_rewrite(self):
        """Copied from CppWrapperCpu to:
        (1) change __file__ path for cpython, so that we can use aoti_runtime in current path.
        (2) rewrite include path of aoti header file.
        """
        if V.graph.is_const_graph:
            # We do not write header for constant graph, it will be written by main module.
            return

        if V.graph.aot_mode:
            self.header.splice(
                """
                #include <torch_npu/csrc/inductor/aoti_runtime/interface.h>
                #include <torch_npu/csrc/inductor/aoti_runtime/model.h>
                """
            )
            with open(
                os.path.join(os.path.dirname(__file__), "aoti_runtime", "interface.cpp")
            ) as f:
                self.header.splice(f.read())
        else:
            self.header.splice(
                """
                import torch
                from torch._inductor.codecache import CppWrapperCodeCache

                cpp_wrapper_src = (
                '''
                #include <pybind11/pybind11.h>
                namespace py = pybind11;

                class RAIIPyObject {
                public:
                    RAIIPyObject() : obj_(nullptr) {}
                    RAIIPyObject(PyObject* obj) : obj_(obj) {}
                    ~RAIIPyObject() {
                        Py_XDECREF(obj_);
                    }
                    RAIIPyObject& operator=(const RAIIPyObject& other) {
                        if (this != &other) {
                            Py_XDECREF(obj_);
                            obj_ = other.obj_;
                            Py_XINCREF(obj_);
                        }
                        return *this;
                    }
                    operator PyObject*() {
                        return obj_;
                    }
                    PyObject* get() {
                        return obj_;
                    }
                private:
                    PyObject* obj_;
                };

                #include <torch_npu/csrc/inductor/aoti_runtime/device_utils.h>
                #include <torch_npu/csrc/inductor/aoti_runtime/utils.h>
                using namespace torch::aot_inductor;
                """
            )

        self.header.splice(
            f"""
            #include <torch_npu/csrc/inductor/aoti_runtime/arrayref_tensor.h>
            #include <torch_npu/csrc/inductor/aoti_runtime/thread_local.h>
            #include <torch_npu/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
            // Here comment c_shim_npu.h because npu doesn't implement it.
            // #include <torch_npu/csrc/inductor/aoti_torch/generated/c_shim_{self.device}.h>

            #include <c10/util/generic_math.h>
            typedef at::Half half;
            typedef at::BFloat16 bfloat16;

            // Round up to the nearest multiple of {ALIGN_BYTES}
            [[maybe_unused]] static int64_t align(int64_t nbytes) {{
              return (nbytes + {ALIGN_BYTES} - 1) & -{ALIGN_BYTES};
            }}
            """
        )
        extend_aoti_c_shim_include = (
            f"torch/csrc/inductor/aoti_torch/generated/extend/c_shim_{self.device}.h"
        )
        extend_aoti_c_shim_path = os.path.join(
            os.path.dirname(torch.__file__),
            "include",
            extend_aoti_c_shim_include,
        )
        if os.path.exists(extend_aoti_c_shim_path):
            self.header.splice(f"#include <{extend_aoti_c_shim_include}>")

        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        if config.profiler_mark_wrapper_call or enable_kernel_profile:
            # No C shim for profiling APIs, assuming profiling is a debugging feature which
            # does not provide any ABI compatibility promise.
            self.header.splice("#include <ATen/record_function.h>")

    def write_header(self):
        if V.graph.is_const_graph:
            # We do not write header for constant graph, it will be written by main module.
            return

        self.super_write_header_rewrite()
        self.header.splice("#include <unistd.h>")
        self.header.splice("#include <filesystem>")
        self.header.splice(self.device_codegen.abi_compatible_header())
        self.header.splice(
            maybe_hipify_code_wrapper(self.device_codegen.kernel_driver())
        )
        self.header.splice("#include <torch_npu/csrc/framework/OpCommand.h>")
        self.header.splice("#include <experiment/runtime/runtime/rt.h>")
        if npu_config.aot_inductor.debug_kernel:
            self.header.splice("#include <torch/torch.h>")

    def write_get_raw_stream(self, device_idx: int, graph=None) -> str:
        name = f"stream{device_idx}"
        self.writeline(
            maybe_hipify_code_wrapper(
                f"{self.device_codegen.cpp_stream_type()} {name};"
            )
        )
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK({self.device_codegen.aoti_get_stream()}({device_idx}, (void**)&{name}));"
        )
        return name

    def codegen_inputs(self):
        # See Note: [Input Alignment handling in Inductor]
        #
        # JIT Inductor does not guard on input alignment. It relies on copy_misaligned_inputs to
        # copy misaligned inputs to aligned buffers. For AOTInductor, we expect users to use it
        # as non-Python deployment for its best performance, so implicitly copying misaligned inputs
        # to aligned buffers is going to bring a surprising performance hit. Instead, we check input
        # alignment and throw an error if any input is misaligned.
        if V.graph.aot_mode and V.graph.inputs_to_check:
            for idx in V.graph.inputs_to_check:
                input_name = V.graph.graph_input_names[idx]
                value = V.graph.graph_inputs[input_name]

                self.prefix.splice(
                    f"""
                    if ((long({input_name}.data_ptr()) & ({NPU_ALIGN_BYTES} -1)) != 0) {{
                        throw std::runtime_error("{input_name} is not aligned to {NPU_ALIGN_BYTES} bytes");
                    }}
                    """
                )

        super().codegen_inputs()

    def define_kernel(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: Optional[str] = None,
        gpu=True,
    ):
        if gpu:
            if config.triton.autotune_at_compile_time:
                # Call PythonWrapperCodegen to create the autotune code block
                PythonWrapperCodegen.define_kernel(
                    self, kernel_name, kernel_body, metadata, gpu
                )
        else:
            return CppWrapperCpu.define_kernel(
                self, kernel_name, kernel_body, metadata, gpu
            )

    def generate(self, is_inference):
        with dynamo_timed("CppWrapperNpu.generate", log_pt2_compile_event=True):
            self.prefix.writeline("\n")
            if not V.graph.aot_mode:
                for kernel in chain(
                    sorted(self.src_to_kernel.values()),
                    sorted(
                        [entry[0] for entry in self.user_defined_kernel_cache.values()]
                    ),
                ):
                    self.prefix.writeline(
                        maybe_hipify_code_wrapper(
                            f"static {self.device_codegen.cpp_kernel_type()} {kernel} = nullptr;"
                        )
                    )
                self.prefix.writeline("\n")
            return super().generate(is_inference)

    def generate_user_defined_triton_kernel(
        self,
        kernel_name: str,
        raw_args: List[Any],
        grid: List[Any],
        configs,
        triton_meta,
        constexprs,
    ):
        if (
            config.triton.autotune_at_compile_time
            and kernel_name not in self.kernel_autotune_names
        ):
            # Call PythonWrapperCodegen to create the autotune code block
            PythonWrapperCodegen.generate_user_defined_triton_kernel(
                self,
                kernel_name,
                raw_args,
                grid,
                configs,
                triton_meta,
                constexprs,
            )

        # in C++ wrapper, we don't pass constexpr args, as they don't
        # get added as parameters to the PTX code compiled from the
        # user-defined Triton kernel (only non-constexpr args do)
        raw_args = [raw_arg for i, raw_arg in enumerate(raw_args) if i not in constexprs]
        args = [self.val_to_arg_str(v) for v in raw_args]
        arg_types = [
            arg.get_dtype() if isinstance(arg, IRNode) else type(arg)
            for arg in raw_args
        ]

        # Call self.generate_kernel_call to generate the real kernel call in cpp
        self.generate_kernel_call(
            kernel_name,
            args,
            arg_types=arg_types,
            raw_args=raw_args,
            grid=grid,
            gpu=True,
            triton=True,
            triton_meta=triton_meta,
            autotune_configs=configs,
        )

    def codegen_tensor_item_npu(
        self, dtype: torch.dtype, tensor: str, scalar: str, indented_buffer=None
    ):
        dtype_str = str(dtype).split(".")[-1]
        writer = indented_buffer or self

        if dtype == torch.float16 or dtype == torch.bfloat16:
            scalar_tmp = f"{scalar}_tmp"
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar_tmp};")
            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar_tmp}));"
            )
            writer.writeline(f"float {scalar} = float({scalar_tmp});")
            struct_data = f"float {scalar} __attribute__((aligned(4)));"
            arg_data = f"static_cast<float>({scalar})"
        else:
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar};")
            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar}));"
            )
            struct_data = f"{DTYPE_TO_CPP[dtype]} {scalar} __attribute__((aligned(sizeof({DTYPE_TO_CPP[dtype]} ))));"
            arg_data = f"static_cast<{DTYPE_TO_CPP[dtype]}>({scalar})"

        return struct_data, arg_data

    def codegen_device(self, device):
        if device.type not in DEVICE_TO_ATEN:
            raise RuntimeError(device.type + "not found in DEVICE_TO_ATEN")
        device_str = DEVICE_TO_ATEN[device.type][5:].lower()  # remove "at::k"
        if device_str == "privateuse1":
            device_str = "npu"
        self.used_cached_devices.add(device_str)
        return f"cached_torch_device_type_{device_str}, {device.index if device.index else 0}"

    def write_wrapper_decl(self):
        super().write_wrapper_decl()
        with self.prefix.indent():
            if not V.graph.aot_mode:
                return
            dump_path = npu_config.aot_inductor.dump_path_cpp
            if npu_config.aot_inductor.debug_kernel:
                self.prefix.splice(
                    f"""
                    auto dump_path = std::filesystem::current_path() / "{dump_path}";
                    if (!std::filesystem::exists(dump_path)) {{
                        std::filesystem::create_directory(dump_path);
                    }}
                    """
                )

                self.prefix.splice(
                    """
                    auto  tensor_handle_to_tensor_pointer = [](AtenTensorHandle handle) {
                        return reinterpret_cast<at::Tensor*>(handle);
                    };
                    """
                )

    def generate_debug_str(self, args, kernel_name, kernel_id, mark):
        if not npu_config.aot_inductor.debug_kernel:
            return ""
        if kernel_id not in self.visited_handle_for_kernel_id:
            self.visited_handle_for_kernel_id[kernel_id] = set()

        def get_tensor_from_handle(h, t):
            if h in self.visited_handle_for_kernel_id[kernel_id]:
                return ""
            self.visited_handle_for_kernel_id[kernel_id].add(h)
            return f"        auto {t} = *tensor_handle_to_tensor_pointer({h});\n"

        # Only dump tensor args, e.g, ['buf2', '8L', '4L'] => ['buf2']
        tensor_args = [arg for arg in args if not arg[0].isdigit()]

        tensor_args_h = [f"{arg}_h" for arg in tensor_args]
        tensor_args_t = [f"{arg}_t" for arg in tensor_args]
        handle_tensor_str = "".join(
            [get_tensor_from_handle(h, t) for h, t in zip(tensor_args_h, tensor_args_t)]
        )

        dump_path = npu_config.aot_inductor.dump_path_cpp
        return f"""
        c10_npu::npuSynchronizeDevice();
        \n{handle_tensor_str}
        std::vector<at::Tensor> arg_{mark}{{{", ".join(tensor_args_t)}}};
        torch::save(arg_{mark}, "{dump_path}/{kernel_id}_{kernel_name}_{mark}.pt");
        """

    def generate_args_decl(
        self,
        code,
        call_args,
        arg_types,
        arg_signatures,
        kernel_name,
        kernel_id,
        is_triton_kernel=True,
    ):
        """
        Generates any declarations of args to pass into a kernel call, and then returns the arg names.

        In more detail:
        * declarations: e.g. this function has a side effect of generating lines like `auto var_0 = ...;`
        * returns: a string with the list of args, e.g. "var_0, var_1"

        call_args: list of call arguments
        arg_types: list of argument types
        arg_signatures: list with signatures of all the args
        is_triton_kernel: whether these are passed into a triton kernel or not. In particular,
                          calls to triton kernels will have an additional global scratch space
                          arg injected at the front of the arg list.
        """
        new_args: list[str] = []

        # Add more cases for other types as needed
        signature2dtype = {
            "i1": "int32_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }

        struct_def_body = ""
        struct_arg_body = ""

        def process_args(arg, arg_type, arg_signature=None):
            var_name = f"var_{next(self.arg_var_id)}"
            # ignore nvTmaDesc, as host-side TMA descriptors need
            # to be passed to the compiled Triton kernel by value
            if isinstance(arg_type, UnwrapUnspecArg) and arg_signature != "nvTmaDesc":
                self.codegen_tensor_item_npu(
                    arg_type.dtype,
                    arg,
                    var_name,
                    indented_buffer=code,
                )
            elif isinstance(arg_type, torch_dtype) and arg_signature != "nvTmaDesc":
                device_ptr_type = self.device_codegen.cpp_device_ptr()
                code.writeline(
                    maybe_hipify_code_wrapper(
                        f"{device_ptr_type} {var_name} = reinterpret_cast<{device_ptr_type}>({arg}.data_ptr());"
                    )
                )
                if npu_config.aot_inductor.debug_kernel:
                    if arg not in self.visited_raii_handle:
                        self.writeline(f"AtenTensorHandle {arg}_h = {arg}.get();")
                        self.visited_raii_handle.add(arg)
                struct_data = f"void* {var_name} __attribute__((aligned(8)));"
                arg_data = f"static_cast<void*>({var_name})"
            elif arg_type in (sympy.Integer, int):
                code.writeline(f"int {var_name} = {cexpr(arg)};")
                struct_data = f"int {var_name} __attribute__((aligned(4)));"
                arg_data = f"static_cast<int>({var_name})"
            elif arg_type in (sympy.Float, float):
                code.writeline(f"float {var_name} = {cexpr(arg)};")
                struct_data = f"float {var_name} __attribute__((aligned(4)));"
                arg_data = f"static_cast<float>({var_name})"
            # For symbolic call arguments, examine the arg signatures from triton meta
            # to explicitly cast to the right type
            # Reason: `auto` can infer unexpected type against kernel input signature.
            elif (
                isinstance(arg_type, type(SymbolicCallArg))
                and arg_signature is not None
                and arg_signature in signature2dtype.keys()
            ):
                code.writeline(
                    f"{signature2dtype[arg_signature]} {var_name} = {cexpr(arg)};"
                )
                struct_data = f"{signature2dtype[arg_signature]} {var_name} __attribute__((aligned(sizeof({signature2dtype[arg_signature]}))));"
                arg_data = f"static_cast<{signature2dtype[arg_signature]}>({var_name})"
            else:
                raise TypeError("Infer arg_type to cpp failed!")
            nonlocal struct_def_body
            nonlocal struct_arg_body
            struct_def_body += struct_data + " "
            struct_arg_body += arg_data + ", "

        for arg, arg_type, arg_signature in zip_longest(
            call_args, arg_types, arg_signatures
        ):
            process_args(arg, arg_type, arg_signature)

        debug_str_before_kernel = self.generate_debug_str(
            call_args, kernel_name, kernel_id, "before"
        )
        debug_str_after_kernel = self.generate_debug_str(
            call_args, kernel_name, kernel_id, "after"
        )
        launch_str = f"""
    auto launch_call_{kernel_id} = [=]() {{
        rtError_t ret;
        void* ffts_addr = NULL;
        uint32_t ffts_len;
        ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
        if (ret != RT_ERROR_NONE) return ret;
        void* workspace_addr = NULL;
        void* sync_block_lock = NULL;

        struct __attribute__((packed)) {{
            void* ffts_addr __attribute__((aligned(8)));
            void* sync_block_lock __attribute__((aligned(8)));
            void* workspace_addr __attribute__((aligned(8)));
            {struct_def_body}
            int32_t grid_0 __attribute__((aligned(4)));
            int32_t grid_1 __attribute__((aligned(4)));
            int32_t grid_2 __attribute__((aligned(4)));
        }} kernel_args = {{
            static_cast<void*>(ffts_addr),
            static_cast<void*>(sync_block_lock),
            static_cast<void*>(workspace_addr),
            {struct_arg_body}
            static_cast<int32_t>(grid_0),
            static_cast<int32_t>(grid_1),
            static_cast<int32_t>(grid_2)
        }};
        
        uint32_t block_num = grid_0 * grid_1 * grid_2;
        auto arg_ptr = static_cast<void*>(&kernel_args);
        auto arg_size = sizeof(kernel_args);
        {debug_str_before_kernel}
        ret = rtKernelLaunch({kernel_name}, block_num, arg_ptr, arg_size, NULL, stream_);
        {debug_str_after_kernel}
        if (ret != RT_ERROR_NONE) return ret;
        return ret;
    }};
        """
        return f"launch_call_{kernel_id}", launch_str

    def generate_kernel_call_npu(
        self,
        kernel_name: str,
        call_args,
        grid=None,
        device_index=None,
        npu=True,
        triton=True,
        arg_types=None,
        raw_args=None,
        grid_fn: str = "grid",
        triton_meta=None,
        autotune_configs=None,
        grid_extra_kwargs="",
    ):
        if (
            config.triton.autotune_at_compile_time
            and kernel_name not in self.kernel_autotune_names
        ):
            # Call PythonWrapperCodegen to create the autotune code block
            PythonWrapperCodegen.generate_kernel_call(
                self,
                kernel_name,
                call_args,
                grid,
                device_index,
                npu,
                triton,
                arg_types,
                raw_args,
                grid_fn,
                triton_meta,
                autotune_configs,
                grid_extra_kwargs,
            )

        if device_index is None:
            current_device = V.graph.get_current_device_or_throw()
            device_index = current_device.index

        stream = (
            "stream"
            if V.graph.aot_mode
            else self.write_get_raw_stream(device_index, V.graph)
        )

        if triton:
            call_args, arg_types = self.prepare_triton_wrapper_args(
                call_args, arg_types
            )
            wrapper_name = f"call_{kernel_name}"
            current_kernel_id = next(self.kernel_callsite_id)
            if wrapper_name not in self._triton_call_wrappers:
                self._triton_call_wrappers[wrapper_name] = DeferredNpuTritonCallWrapper(
                    wrapper_name, kernel_name, arg_types, current_kernel_id
                )
            call_args.append(stream)
            if V.graph.aot_mode:
                call_args.append("kernels")
                call_args.append("this->cubin_dir_")
            debug_printer_manager = V.graph.wrapper_code.debug_printer
            debug_printer_manager.set_printer_args(
                call_args[: len(arg_types)], kernel_name, arg_types, None
            )
            with debug_printer_manager:
                self.writeline(f"{wrapper_name}({', '.join(call_args)});")
        else:
            casted = []
            for arg_type, arg in zip(arg_types, call_args):
                new_arg = arg
                if arg_type.endswith("*") and arg != "nullptr":
                    new_arg = f"{arg}.data_ptr()"
                casted.append(f"({arg_type}){new_arg}")
            call_args_str = ", ".join(casted)
            self.writeline(f"kernels.{kernel_name}({call_args_str}, {stream});")

    def generate_kernel_call(
        self,
        kernel_name: str,
        call_args,
        grid=None,
        device_index=None,
        gpu=True,
        triton=True,
        arg_types=None,
        raw_args=None,
        grid_fn: str = "grid",
        triton_meta=None,
        autotune_configs=None,
        grid_extra_kwargs="",
    ):
        """
        Override the default value of argument 'gpu' to True here.
        generate_kernel_call can still be called with gpu=False because of
        a mix of cpu kernels and gpu kernels.
        """

        """
        To fit with NPU: we write a new function 'generate_kernel_call_npu
        and make a new parameter called 'npu', which always equals to 'gpu',
        because 'gpu' parameter means 'not cpu' in upper logic
        """

        if not gpu:
            # Even in CppWrapperNpu, we may see cpp kernels
            return CppWrapperCpu.generate_kernel_call(
                self,
                kernel_name,
                call_args,
                grid,
                device_index,
                gpu,
                triton,
                arg_types,
                raw_args,
                grid_fn,
                triton_meta,
                autotune_configs,
                grid_extra_kwargs,
            )

        self.generate_kernel_call_npu(
            kernel_name,
            call_args,
            grid,
            device_index,
            gpu,
            triton,
            arg_types,
            raw_args,
            grid_fn,
            triton_meta,
            autotune_configs,
            grid_extra_kwargs,
        )

    def finalize_prefix(self):
        """Define the triton kernels now that autotuning is finished"""
        old_prefix = self.prefix  # new content should go at start of prefix
        self.prefix = IndentedBuffer()
        super().finalize_prefix()
        for kernel in self._triton_call_wrappers.values():
            self.prefix.writeline("\n")
            kernel.generate(self)
        self.prefix.writeline("\n")
        self.prefix.splice(old_prefix)

    @staticmethod
    def prepare_triton_wrapper_args(
        call_args: list[Any], arg_types: list[Any]
    ) -> tuple[list[Any], list[Any]]:
        new_args = []
        new_args_types = []
        for arg, arg_type in zip(call_args, arg_types):
            if isinstance(arg, str):
                if isinstance(arg_type, torch_dtype) and should_unwrap_unspec_arg(arg):
                    # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
                    arg_type = UnwrapUnspecArg(dtype=arg_type)
                new_args.append(arg)
            elif isinstance(arg, bool):
                new_args.append(str(arg).lower())
            elif isinstance(arg, (int, float, SymbolicCallArg)):
                new_args.append(str(arg))
            else:
                new_args.append(cexpr(V.graph.sizevars.simplify(arg)))
            new_args_types.append(arg_type)
        return new_args, new_args_types

    def make_zero_buffer(self, name):
        return f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_zero_({name}.get()));"
