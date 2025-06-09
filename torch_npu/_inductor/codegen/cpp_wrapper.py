import functools
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
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.codegen.wrapper import PythonWrapperCodegen, SymbolicCallArg
from torch._inductor.ir import IRNode, TensorBox
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch._inductor.runtime.triton_heuristics import grid as default_grid_fn
from torch._inductor.utils import DeferredLineBase
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


class DeferredNpuKernelLine(DeferredLineBase):
    """
    When using cpp wrapper, NPU kernel load and launch needs to wait for Triton kernels
    to be tuned and stored as cubin files, so use a deferred line to backfill those information
    """

    def __init__(
            self,
            kernel_name: str,
            line_template: str,
            keys: Tuple[str, ...],
            additional_files: List[str],
    ):
        super().__init__(line_template)
        checkIfTrue(not isinstance(line_template, DeferredLineBase), "line template can not be DeferredLineBase")
        self.additional_files = additional_files
        self.kernel_name = kernel_name
        self.line_template = line_template
        self.keys = keys

    def __call__(self):
        if self.kernel_name.startswith("multi_kernel_"):
            # MultiKernel will select one kernel after running the autotune block
            self.kernel_name = MultiKernelCall.lookup_choice(self.kernel_name)
        params = CudaKernelParamCache.get(self.kernel_name)
        checkIfTrue(params is not None, f"{self.kernel_name} not found in CudaKernelParamCache")

        for key in self.keys:
            checkIfTrue(key in params, f"{key} not found in CudaKernelParamCache[{self.kernel_name}]")

            if key == get_cpp_wrapper_cubin_path_name():
                checkIfTrue(os.path.exists(params[key]), f"{params[key]} does not exist")
                self.additional_files.append(params[key])

        return self.line_template % tuple(params[key] for key in self.keys)

    def _new_line(self, line):
        return DeferredNpuKernelLine(
            self.kernel_name, line, self.keys, self.additional_files
        )


class DeferredNpuDefaultGrid:
    """
    A container for the default grid, which may be used by DeferredNpuGridLine
    """

    def __init__(
            self,
            kernel_name: str,
            grid,
            grid_callable: Optional[Callable[..., Any]] = None,
            **grid_extra_kwargs,
    ):
        self.kernel_name = kernel_name
        self.grid = grid
        self.grid_callable = grid_callable
        self.grid_extra_kwargs = grid_extra_kwargs

    def __iter__(self):
        # DeferredNpuDefaultGrid can be passed to the base class, PythonWrapperCodegen,
        # to generate the autotune code block, and thus we need this iterator
        return iter(self.grid)

    def _process_grid(self, grid: Union[List[Any], Tuple[Any, ...]]):
        if isinstance(grid, (list, tuple)):
            return [self._process_grid(e) for e in grid]
        else:
            return grid.inner_expr if isinstance(grid, SymbolicCallArg) else grid

    def __call__(self):
        if self.kernel_name.startswith("multi_kernel_"):
            # MultiKernel will select one kernel after running the autotune block
            self.kernel_name = MultiKernelCall.lookup_choice(self.kernel_name)

        grid = self.grid
        checkIfTrue(isinstance(grid, (list, tuple)), f"expected {grid=} to be a list")

        grid = self._process_grid(grid)

        checkIfTrue(self.grid_callable is not None, "grid_callable can't be None")

        if not self.grid_extra_kwargs:
            grid_fn = self.grid_callable(*grid)
        else:
            grid_fn = self.grid_callable(*grid, **self.grid_extra_kwargs)

        params = CudaKernelParamCache.get(self.kernel_name)
        checkIfTrue(params is not None, f"{self.kernel_name} not found in CudaKernelParamCache")

        return grid_fn(params["meta"])


class DeferredNpuGridLine(DeferredLineBase):
    """
    When using cpp wrapper, NPU kernel load and launch needs to wait for Triton kernels
    to be tuned and stored as cubin files, so use a deferred line to backfill those information
    """

    def __init__(
            self,
            kernel_name: str,
            grid_var: str,
            grid,
            autotune_configs,
    ):
        super().__init__("")
        self.kernel_name = kernel_name
        self.grid_var = grid_var
        self.grid = grid
        self.autotune_configs = autotune_configs

    def __call__(self):
        if self.kernel_name.startswith("multi_kernel_"):
            # MultiKernel will select one kernel after running the autotune block
            self.kernel_name = MultiKernelCall.lookup_choice(self.kernel_name)

        params = CudaKernelParamCache.get(self.kernel_name)

        checkIfTrue(params is not None, f"{self.kernel_name} not found in CudaKernelParamCache")

        if self.autotune_configs is not None:
            # This indicates the Triton kernel is a user-defined one.
            grid = None
            if len(self.grid) == 1:
                grid = self.grid[0]
            else:
                for i, c in enumerate(self.autotune_configs):
                    if all(arg == params["meta"][key] for key, arg in c.kwargs.items()):
                        grid = self.grid[i]
                        break
            checkIfTrue(grid is not None, "grid can not be None")
            grid_args_str = ", ".join(
                [cexpr(V.graph.sizevars.simplify(item)) for item in grid]
            )
        else:
            launch_grid = (params['grid_x'], params['grid_y'], params['grid_z'])
            grid_args_str = ", ".join(
                [cexpr(item) for item in launch_grid]
            )

        return f"\n    Grid {self.grid_var} = Grid({grid_args_str});\n"

    def _new_line(self, line):
        return DeferredNpuGridLine(
            self.kernel_name, self.grid_var, self.grid, self.autotune_configs
        )


class CppWrapperNpu(CppWrapperCpu):
    """
    Generates cpp wrapper for running on NPU and calls CUDA kernels
    """

    def __init__(self) -> None:
        self.device = 'npu'
        self.device_codegen = get_device_op_overrides(self.device)
        super().__init__()
        self.grid_id = count()
        self.visited_raii_handle = set()
        self.visited_handle_for_kernel_id = dict()

    @staticmethod
    def create(
            is_subgraph: bool, subgraph_name: str, parent_wrapper: PythonWrapperCodegen
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
                checkIfTrue(input_name in V.graph.graph_inputs, f"{input_name} not found in graph inputs")

                value = V.graph.graph_inputs[input_name]
                checkIfTrue(isinstance(value, TensorBox),
                            f"{input_name} is expected to be tensor but found as {type(value)}")

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

    @functools.lru_cache(None)  # noqa: B019
    def generate_load_kernel_once(
            self,
            kernel_name: str,
            device_index,
            graph: "GraphLowering",  # for per-graph caching
    ):
        keys = (get_cpp_wrapper_cubin_path_name(), "mangled_name", "shared_mem")
        kernel_var_name = f"kernels.{kernel_name}" if V.graph.aot_mode else kernel_name
        self.writeline(f"if ({kernel_var_name} == nullptr) {{")
        deferred_gpu_kernel_line = DeferredNpuKernelLine(
            kernel_name,
            "    " + kernel_var_name + r' = loadKernel("%s", "%s", %s, this->cubin_dir_);',
            keys,
            self.additional_files,
        )
        self.writeline(deferred_gpu_kernel_line)
        self.writeline("}")
        return kernel_var_name

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
            struct_data = f'float {scalar} __attribute__((aligned(4)));'
            arg_data = f'static_cast<float>({scalar})'
        else:
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar};")
            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar}));"
            )
            struct_data = f'{DTYPE_TO_CPP[dtype]} {scalar} __attribute__((aligned(sizeof({DTYPE_TO_CPP[dtype]} ))));'
            arg_data = f'static_cast<{DTYPE_TO_CPP[dtype]}>({scalar})'

        return struct_data, arg_data

    def codegen_device(self, device):
        if device.type not in DEVICE_TO_ATEN:
            raise RuntimeError(device.type + "not found in DEVICE_TO_ATEN")
        device_str = DEVICE_TO_ATEN[device.type][5:].lower() # remove "at::k"
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
        handle_tensor_str = "".join([
            get_tensor_from_handle(h, t) for h, t in zip(tensor_args_h, tensor_args_t)
        ])

        dump_path = npu_config.aot_inductor.dump_path_cpp
        return f"""
        c10_npu::npuSynchronizeDevice();
        \n{handle_tensor_str}
        std::vector<at::Tensor> arg_{mark}{{{", ".join(tensor_args_t)}}};
        torch::save(arg_{mark}, "{dump_path}/{kernel_id}_{kernel_name}_{mark}.pt");
        """

    def generate_launch_call(
        self,
        call_args,
        arg_types,
        arg_signatures,
        kernel_id,
        grid_var,
        kernel_name
    ):
        kernel_val_name = f"kernels.{kernel_name}" if V.graph.aot_mode else kernel_name
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


        struct_def_body = ''
        struct_arg_body = ''

        def process_args(arg, arg_type, arg_signature=None):
            var_name = f"var_{next(self.arg_var_id)}"
            # ignore nvTmaDesc, as host-side TMA descriptors need
            # to be passed to the compiled Triton kernel by value
            if isinstance(arg_type, torch_dtype) and arg_signature != "nvTmaDesc":
                if arg.endswith(".item()"):  # scalar
                    # Need to declare a scalar in this case
                    arg = arg[:-7]
                    struct_data, arg_data = self.codegen_tensor_item_npu(
                        arg_type,
                        arg,
                        var_name,
                    )
                else:
                    # void*
                    device_ptr_type = self.device_codegen.cpp_device_ptr()
                    self.writeline(
                        maybe_hipify_code_wrapper(
                            f"{device_ptr_type} {var_name} = reinterpret_cast<{device_ptr_type}>({arg}.data_ptr());"
                        )
                    )
                    if npu_config.aot_inductor.debug_kernel:
                        if arg not in self.visited_raii_handle:
                            self.writeline(
                                f"AtenTensorHandle {arg}_h = {arg}.get();"
                            )
                            self.visited_raii_handle.add(arg)
                    struct_data = f'void* {var_name} __attribute__((aligned(8)));'
                    arg_data = f'static_cast<void*>({var_name})'

            elif arg_type in (sympy.Integer, int):
                # int
                self.writeline(f"int {var_name} = {cexpr(arg)};")
                struct_data = f'int {var_name} __attribute__((aligned(4)));'
                arg_data = f'static_cast<int>({var_name})'

            elif arg_type in (sympy.Float, float):
                # float
                self.writeline(f"float {var_name} = {cexpr(arg)};")
                struct_data = f'float {var_name} __attribute__((aligned(4)));'
                arg_data = f'static_cast<float>({var_name})'

            # For symbolic call arguments, examine the arg signatures from triton meta
            # to explicitly cast to the right type
            # Reason: `auto` can infer unexpected type against kernel input signature.
            elif (
                    isinstance(arg_type, type(SymbolicCallArg))
                    and arg_signature is not None
                    and arg_signature in signature2dtype.keys()
            ):
                # or scalar symbolic type，currently only support scalar symbolic type
                self.writeline(
                    f"{signature2dtype[arg_signature]} {var_name} = {cexpr(arg)};"
                )
                struct_data = f'{signature2dtype[arg_signature]} {var_name} __attribute__((aligned(sizeof({signature2dtype[arg_signature]}))));'
                arg_data = f'static_cast<{signature2dtype[arg_signature]}>({var_name})'
            else:
                raise TypeError("Infer arg_type to cpp failed!")

            nonlocal struct_def_body
            nonlocal struct_arg_body
            struct_def_body += struct_data + ' '
            struct_arg_body += arg_data + ', '

        for arg, arg_type, arg_signature in zip_longest(
                call_args, arg_types, arg_signatures
        ):
            process_args(arg, arg_type, arg_signature)

        debug_str_before_kernel = self.generate_debug_str(call_args, kernel_name, kernel_id, "before")
        debug_str_after_kernel = self.generate_debug_str(call_args, kernel_name, kernel_id, "after")

        launch_str = f"""
    auto launch_call_{kernel_id} = [=]() {{
        int32_t grid_x = {grid_var}.grid_x;
        int32_t grid_y = {grid_var}.grid_y;
        int32_t grid_z = {grid_var}.grid_z;
        rtError_t ret;
        void* ffts_addr = NULL;
        uint32_t ffts_len;
        ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
        if (ret != RT_ERROR_NONE) return ret;
        void* workspace_addr = NULL;

        struct __attribute__((packed)) {{
            void* ffts_addr __attribute__((aligned(8)));
            void* workspace_addr __attribute__((aligned(8)));
            {struct_def_body}
            int32_t grid_x __attribute__((aligned(4)));
            int32_t grid_y __attribute__((aligned(4)));
            int32_t grid_z __attribute__((aligned(4)));
        }} kernel_args = {{
            static_cast<void*>(ffts_addr),
            static_cast<void*>(workspace_addr),
            {struct_arg_body}
            static_cast<int32_t>(grid_x),
            static_cast<int32_t>(grid_y),
            static_cast<int32_t>(grid_z)
        }};
        
        uint32_t block_num = grid_x * grid_y * grid_z;
        auto arg_ptr = static_cast<void*>(&kernel_args);
        auto arg_size = sizeof(kernel_args);
        {debug_str_before_kernel}
        ret = rtKernelLaunch({kernel_val_name}, block_num, arg_ptr, arg_size, NULL, stream);
        {debug_str_after_kernel}
        if (ret != RT_ERROR_NONE) return ret;
        return ret;
    }};
        """
        return f"launch_call_{kernel_id}", launch_str

    def generate_default_grid(
            self,
            kernel_name: str,
            grid_args: List[Any],
            gpu: bool = True,
            grid_callable: Optional[Callable[..., Any]] = default_grid_fn,
            **grid_extra_kwargs,
    ):
        """
        Generate grid configs for launching a CUDA kernel using the grid
        function from triton_heuristics. Because its computation needs
        to read kernel config after autotune, it is done in a deferred way
        using DeferredNpuDefaultGrid.
        """
        checkIfTrue(gpu, "CppWrapperNpu.generate_default_grid does not support non-NPU")
        return DeferredNpuDefaultGrid(
            kernel_name, grid_args, grid_callable, **grid_extra_kwargs
        )

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
            device_index, call_args = self.prepare_triton_kernel_call(
                device_index, call_args
            )
            _ = self.generate_load_kernel_once(kernel_name, device_index, V.graph)

            # args with value 1 are added into equal_to_1 and constants
            # in triton_meta (in the Python codegen) which makes them
            # inlined in the PTX and compiled CUBIN
            arg_signatures = []
            if (
                    triton_meta is not None
                    and triton_meta.get("configs")
                    and triton_meta.get("signature")
            ):
                equal_to_1 = triton_meta["configs"][0].equal_to_1
                call_args = [
                    arg
                    for i, arg in enumerate(call_args)
                    if i not in equal_to_1
                ]
                arg_types = [t for i, t in enumerate(arg_types) if i not in equal_to_1]
                # extract the arg signatures from triton_meta
                arg_signatures = triton_meta["signature"].values()
                arg_signatures = [
                    v
                    for i, v in enumerate(arg_signatures)
                    if i not in equal_to_1
                ]

            current_kernel_id = next(self.kernel_callsite_id)
            current_grid_id = next(self.grid_id)

            # gen grids
            grid_var = f"{kernel_name}_grid_{current_grid_id}"
            self.writeline(
                DeferredNpuGridLine(kernel_name, grid_var, grid, autotune_configs)
            )

            call, call_args_str = self.generate_launch_call(
                call_args, arg_types, arg_signatures, current_kernel_id, grid_var, kernel_name
            )
            self.writeline(f"{call_args_str}")

            # add debug printer code for all triton kernel related calls
            debug_printer_manager = V.graph.wrapper_code.debug_printer
            debug_printer_manager.set_printer_args(
                call_args, kernel_name, arg_types, None
            )
            with debug_printer_manager:
                self.writeline(f"if ({grid_var}.is_non_zero()) {{")
                self.writeline(
                    DeferredNpuKernelLine(
                        kernel_name,
                        r"    launchKernel({}, {});".format( \
                            call,
                           f'"{kernel_name}"',
                        ),
                        (),
                        self.additional_files,
                    ),
                )

                self.writeline("}\n")
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

    def make_zero_buffer(self, name):
        return f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_zero_({name}.get()));"
