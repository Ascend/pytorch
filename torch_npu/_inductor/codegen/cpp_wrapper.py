import dataclasses
import os
import sys
from itertools import chain, count, zip_longest
from typing import Any, Optional, TYPE_CHECKING, Union

from typing_extensions import Self
import sympy
import torch
from torch import dtype as torch_dtype
from torch._inductor import config
from torch._inductor.codegen.aoti_hipify_utils import maybe_hipify_code_wrapper
from torch._inductor.codegen.cpp_utils import cexpr, DTYPE_TO_CPP, DEVICE_TO_ATEN
from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu, DeferredTritonCallWrapper
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.codegen.wrapper import PythonWrapperCodegen, SymbolicCallArg, pexpr
from torch._inductor.ir import IRNode, TensorBox, GraphPartitionSignature
from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu, DeferredTritonCallWrapper, UnwrapUnspecArg
from torch._inductor.codegen.wrapper import PythonWrapperCodegen, SymbolicCallArg
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch._inductor.utils import ALIGN_BYTES

from .common import get_device_op_overrides
from .. import config as npu_config
from ..runtime.triton_heuristics import GridExprNpu
from ..utils import triton_support_ffts, NPU_ALIGN_BYTES

if TYPE_CHECKING:
    from torch._inductor.graph import GraphLowering

# follow triton-ascend implement
DTYPE_TO_CPP[torch.bool] = "int32_t"
DTYPE_TO_CPP[torch.float16] = "float"
DTYPE_TO_CPP[torch.bfloat16] = "float"

@dataclasses.dataclass
class DeferredNpuTritonCallWrapper(DeferredTritonCallWrapper):
    """
    When using cpp wrapper, GPU kernel load and launch needs to wait for Triton kernels
    to be tuned and stored as cubin files, so use a deferred generating the final wrapper around
    the triton kernel until right before the prefix is written.
    """

    wrapper_name: str
    kernel_name: str
    arg_types: list[Any]
    kernel_id: int

    def generate_grid(
        self,
        prefix: IndentedBuffer,
        inductor_meta: dict[str, Any],
        params: dict[str, Any],
    ):

        numels = [arg for arg in params["def_args"] if "_numel" in arg]
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


class CppWrapperNpu(CppWrapperGpu):
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
                #include <torch_npu/csrc/inductor/aoti_runtime/utils_npu.h>
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
                #include <torch_npu/csrc/inductor/aoti_runtime/utils_npu.h>
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
        self.header.splice("#include <runtime/runtime/rt.h>")

    def generate_node_numel_expr(self, kernel_name: str, node, numel_expr):
        expr = f"{kernel_name}_{node.name}_numel"

        if (expr, V.graph) not in self.kernel_numel_expr:
            # declare expr once in each graph (scope)
            self.kernel_numel_expr.add((expr, V.graph))
            self.writeline(f"int64_t {expr} = {cexpr(numel_expr)};")
        else:
            self.writeline(f"{expr} = {cexpr(numel_expr)};")
        return SymbolicCallArg(expr, numel_expr)


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

    def generate_kernel_call(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_args=None,
        triton_meta=None,
    ):
        super().generate_kernel_call(
            kernel_name,
            call_args,
            device=device,
            triton=triton,
            arg_types=arg_types,
            raw_args=raw_args,
            triton_meta=triton_meta
        )

        wrapper_name = f"call_{kernel_name}"
        if wrapper_name in self._triton_call_wrappers:
            # trans DeferredTritonCallWrapper to DeferredNpuTritonCallWrapper
            wrapper = self._triton_call_wrappers[wrapper_name]
            current_kernel_id = next(self.kernel_callsite_id)
            npu_wrapper = DeferredNpuTritonCallWrapper(
                wrapper.wrapper_name,
                wrapper.kernel_name,
                wrapper.arg_types,
                current_kernel_id,
            )
            self._triton_call_wrappers[wrapper_name] = npu_wrapper

    def add_device_include(self, device: str) -> None:
        if device in self.included_devices:
            return

        self.included_devices.add(device)

        # todo: add aoti_include, cpp_wrapper, aoti_torch/c implement in csrc/inductor to support extern kernel

    def generate(self, is_inference):
        with dynamo_timed("CppWrapperNpu.generate", log_pt2_compile_event=True):
            return super().generate(is_inference)

    def prepare_triton_kernel_call(self, call_args):
        new_call_args = call_args
        if npu_config.inductor_static_mode:
            # in inductor_static_mode, numel arg is constexpr, remove all Integer constant args from call_args
            new_call_args = [
                call_arg
                for call_arg in call_args
                if not isinstance(call_arg, sympy.Integer)
            ]

        return super().prepare_triton_kernel_call(new_call_args)

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

    def generate_args_decl(
        self,
        code: Union[IndentedBuffer, Self],
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
            "u1": "uint32_t",
            "u8": "uint8_t",
            "u16": "uint16_t",
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

        target_support_ffts = triton_support_ffts()

        def process_args(arg, arg_type, arg_signature=None):
            var_name = f"var_{next(self.arg_var_id)}"
            # ignore nvTmaDesc, as host-side TMA descriptors need
            # to be passed to the compiled Triton kernel by value
            if isinstance(arg_type, UnwrapUnspecArg) and arg_signature != "nvTmaDesc":
                struct_data, arg_data = self.codegen_tensor_item_npu(
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
                struct_data = f"void* {var_name} __attribute__((aligned(8)));"
                arg_data = f"static_cast<void*>({var_name})"
            elif arg_type in (sympy.Integer, int):
                code.writeline(f"int32_t {var_name} = {cexpr(arg)};")
                struct_data = f"int32_t {var_name} __attribute__((aligned(4)));"
                arg_data = f"static_cast<int32_t>({var_name})"
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

        ffts_str = """
        void* ffts_addr = NULL;
        uint32_t ffts_len;
        ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
        if (ret != RT_ERROR_NONE) return ret;
        """
        launch_str = f"""
    auto launch_call_{kernel_id} = [=]() {{
        rtError_t ret;
        {ffts_str if target_support_ffts else ''}
        void* workspace_addr = NULL;
        void* sync_block_lock = NULL;

        struct __attribute__((packed)) {{
            {"void* ffts_addr __attribute__((aligned(8)));" if target_support_ffts else ''}
            void* sync_block_lock __attribute__((aligned(8)));
            void* workspace_addr __attribute__((aligned(8)));
            {struct_def_body}
            int32_t grid_0 __attribute__((aligned(4)));
            int32_t grid_1 __attribute__((aligned(4)));
            int32_t grid_2 __attribute__((aligned(4)));
        }} kernel_args = {{
            {"static_cast<void*>(ffts_addr)," if target_support_ffts else ''}
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
        ret = rtKernelLaunch({kernel_name}, block_num, arg_ptr, arg_size, NULL, stream_);
        if (ret != RT_ERROR_NONE) return ret;
        return ret;
    }};
        """
        return f"launch_call_{kernel_id}", launch_str
