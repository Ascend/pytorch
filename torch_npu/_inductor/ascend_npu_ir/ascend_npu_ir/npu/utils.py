import os
import sys
import re
import hashlib
import tempfile
import textwrap
import functools
import logging
import sysconfig
import shutil
import subprocess
import copy
import torch
import torch.nn as nn
from typing import Any, Tuple
from sympy import Expr
from pathlib import Path
from typing import List

from typing import (
    Optional,
    List,
    Dict,
    Union,
    Tuple
)

from torch.fx.graph_module import (
    _custom_builtins,
    _addindent,
    warnings
)
import torch_npu

try:
    from torch_mlir import ir
    import torch_mlir
    from torch_mlir.dialects import func as func_dialect
except ImportError:
    print("Can NOT find torch_mlir, INSTALL it first.")

from ..build_info import ABI_TAG

MLIR_DTYPE_MAPPING = {
    "f32": torch.float32,
    "i1" : torch.bool,
    "bf16" : torch.bfloat16,
    "f16" : torch.float16,
    "si64" : torch.int64
}

def run_once(f):
    """Runs a function (successfully) only once.
    The running can be reset by setting the `has_run` attribute to False
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            result = f(*args, **kwargs)
            wrapper.has_run = True
            return result
        return None
    wrapper.has_run = False
    return wrapper

def get_device_info(example_inputs) -> Union[Tuple[str, int], None]:
    for inp in example_inputs:
        if isinstance(inp, torch.Tensor):
            return inp.device, inp.device.index
        
@functools.lru_cache(None)
def _get_ascend_path() -> str:
    path = os.getenv("ASCEND_HOME_PATH", "")
    if path == "":
        raise Exception("ASCEND_HOME_PATH is not set, source <ascend-toolkit>/set_env.sh first")
    return Path(path)

def _build_npu_ext(obj_name: str, src_path, src_dir) -> str:
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so_path = os.path.join(src_dir, f"{obj_name}{suffix}")

    cxx = os.environ.get("CC")
    if cxx is None:
        clangxx = shutil.which("clang++")
        gxx = shutil.which("g++")
        cxx = gxx if gxx is not None else clangxx
        if cxx is None:
            raise RuntimeError("Failed to find C++ compiler")
    cc_cmd = [cxx, src_path]

    # find the python library
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

    cc_cmd += [f"-I{py_include_dir}"]
    torch_npu_root = Path(torch_npu.__file__).resolve().parent
    
    cpp_common_dir = (
        torch_npu_root / "include" / "torch_npu" / "csrc" / "inductor" / "mlir"
    )
    
    torch_npu_dir = torch_npu_root / "include"
    torch_npu_lib_dir = torch_npu_root / "lib"

    cc_cmd += [
        f"-I{torch_npu_dir}",
        f"-I{cpp_common_dir}",
        f"-L{torch_npu_lib_dir}",
        "-ltorch_npu",
        f"-Wl,-rpath",
        "-std=c++17",
        f"-D_GLIBCXX_USE_CXX11_ABI={ABI_TAG}",
        "-shared",
    ]
    cc_cmd += ["-fPIC", "-o", so_path]
    ret = subprocess.check_call(cc_cmd)

    if ret == 0:
        return so_path
    else:
        raise RuntimeError("Failed to compile " + src_path)


def parse_fx_example_inputs(gm: torch.fx.GraphModule):
    name_to_example_inputs = {}
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            name_to_example_inputs[node.name] = node.meta['val']
    return name_to_example_inputs


def generate_compiler_repro_string(gm: torch.fx.GraphModule):
    from torch._dynamo.debug_utils import NNModuleToString
    model_str = textwrap.dedent(
        f"""
import torch
import torch_npu
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

        """
    )

    model_str += NNModuleToString.convert(gm)
    model_str += "\n"
    model_str += "mod = Repro()\n"
    return model_str


def get_fx_graph_code(code, num_args, method=2, runnable=False, kernel_code='', kernel_name=None):      
    kernel_header = ''
    kernel_wrapper = ''
    kernel_runner_and_acc_comp = ''
    if len(kernel_code):
        kernel_header = """
from torch import empty_strided, empty, randn
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.codecache import CustomAsyncCompile
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.utils import (
    logger,
)
import logging
logger.setLevel(logging.INFO)
async_compile = CustomAsyncCompile()

"""
        kernel_wrapper = """
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.npu_inductor_plugin import get_current_raw_stream as get_raw_stream

async_compile.wait(globals())
del async_compile

stream0 = get_raw_stream(0)
"""
        kernel_runner_and_acc_comp = f"""
kernel_dump_path = os.path.join(dir_path, 'kernel_dump')
for file_name in os.listdir(kernel_dump_path):
    kernel_path = os.path.join(kernel_dump_path, file_name)
    {kernel_name}.replace_kernel_by_path(kernel_path)
    {kernel_name}.run(
        *args,
        stream=stream0)

    output1 = args[num_args:]

    if not os.environ.get("DISABLE_ACC_COMP", "0") == "1":
        for o1, o2 in zip(output1, output2):
            if o2.dtype != o1.dtype:
                o2 = o2.to(o1.dtype)
            acc_comp_tol = npu_config.acc_comp_tol.get(o1.dtype, npu_config.acc_comp_tol['default'])
            rtol = acc_comp_tol['rtol']
            atol = acc_comp_tol['atol']
            torch.testing.assert_close(o1, o2, rtol=rtol, atol=atol, equal_nan=False)
        print('accuracy success!')
"""
    code = textwrap.indent(code, '    ')
    transformed_code_template = f"""
def get_args():
    args = torch.load(os.path.join(dir_path, "data.pth"))
    args = [arg.npu() if isinstance(arg, torch.Tensor) else arg for arg in args]
    num_args = {num_args}

    return args
"""
    run_code_template = f"""

try: 
    args = torch.load(os.path.join(dir_path, "data.pth"))
except Exception as e:
    {{{{FAKE_ARGS_PLACEHOLDER}}}}
args = [arg.npu() if isinstance(arg, torch.Tensor) else arg for arg in args]
num_args = {num_args}

fx_inputs = [clone_preserve_strides(arg) for arg in args[:num_args]]
"""
    fx_runner = f"""
fx_inputs = [inp.float() if inp.dtype == torch.bfloat16 else inp for inp in fx_inputs]
with torch.no_grad():
    output2 = model(*fx_inputs)
"""
    code_template = f"""
import os    
import torch
from torch._inductor.compile_fx import clone_preserve_strides
from torch._dynamo.testing import rand_strided
from torch import device

import torch_npu
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir import config as npu_config 
{kernel_header}
file_path = os.path.abspath(__file__) 
dir_path = os.path.dirname(file_path)

{kernel_code}
{kernel_wrapper}

class GraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
{code}
model = GraphModule().npu()

{run_code_template if runnable else transformed_code_template}    
{fx_runner if runnable else ''}
{kernel_runner_and_acc_comp if runnable else ''}
"""
    return code_template

def codegen_python_shape_tuple(shape: Tuple[Expr, ...]) -> str:
    from torch._inductor.virtualized import V
    parts = list(map(V.graph.wrapper_code.codegen_python_sizevar, shape))
    if len(parts) == 0:
        return "()"
    if len(parts) == 1:
        return f"({parts[0]}, )"
    return f"({', '.join(parts)})"

def view_to_reshape(gm: torch.fx.GraphModule):
    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.view.default
    ):
        nd.target = torch.ops.aten.reshape.default
    
    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.div.Tensor
    ):
        if not (isinstance(nd.args[1], torch.fx.node.Node) and \
                isinstance(nd.args[1].meta['val'], torch.Tensor)):
            nd.target = torch.ops.aten.div.Scalar
    
    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.add.Tensor
    ):
        if not (isinstance(nd.args[1], torch.fx.node.Node) and \
                isinstance(nd.args[1].meta['val'], torch.Tensor)):
            nd.target = torch.ops.aten.add.Scalar
    
    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.sub.Tensor
    ):  
        if not (isinstance(nd.args[1], torch.fx.node.Node) and \
                isinstance(nd.args[1].meta['val'], torch.Tensor)):
            nd.target = torch.ops.aten.sub.Scalar
    
    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.mul.Tensor
    ):
        if not (isinstance(nd.args[1], torch.fx.node.Node) and \
                isinstance(nd.args[1].meta['val'], torch.Tensor)):
            nd.target = torch.ops.aten.mul.Scalar

    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.prims.convert_element_type.default
    ):
        nd.target = torch.ops.npu.npu_dtype_cast.default
    
def npu_cast_to_prim_cast(gm: torch.fx.GraphModule):
    """
    Replace npu.npu_dtype_cast ops in the GraphModule to prims.convert_element_type ops.
    """
    new_gm = copy.deepcopy(gm)
    for nd in new_gm.graph.nodes:
        if nd.target in [torch.ops.npu.npu_dtype_cast.default, torch.ops.npu.npu_dtype_cast_backward.default, torch.ops.npu._npu_dtype_cast.default, torch.ops.npu._npu_dtype_cast_backward.default]:
            nd.target = torch.ops.prims.convert_element_type.default
        if nd.target in [torch.ops.aten.index_put_.default]:
            nd.target = torch.ops.aten.index_put.default
    return new_gm

def modify_gm_for_acc_comp(gm: torch.fx.GraphModule):
    """
    In precision comparison mode, if the second argument of npu_dtype_cast is torch.bfloat16, change it to torch.float32.
    """
    for nd in gm.graph.nodes:
        if nd.target in [torch.ops.npu.npu_dtype_cast.default, torch.ops.npu.npu_dtype_cast_backward.default, torch.ops.npu._npu_dtype_cast.default, torch.ops.npu._npu_dtype_cast_backward.default]:
            if nd.args[1] == torch.bfloat16:
                new_args = list(nd.args)
                new_args[1] = torch.float32
                nd.args = tuple(new_args)

def replace_iota_int64_to_int32(nd: torch.fx.Node):
    """
    Replace iota dtype from int64 to int32.
    """
    if nd.target in [torch.ops.prims.iota.default] and nd.kwargs['dtype'] == torch.int64:
        new_args = dict(nd.kwargs)
        new_args['dtype'] = torch.int32
        nd.kwargs = new_args

def npu_optimize_fx_graph(gm: torch.fx.GraphModule):
    """
    optimize fx graph for npu
    """
    aten_empty_nodes = set()
    for nd in gm.graph.nodes:
        replace_iota_int64_to_int32(nd)
        # Replace npu type_as ops in the GraphModule to cast ops.
        if nd.target == torch.ops.aten.empty.memory_format and len(nd.users) == 1:
            aten_empty_nodes.add(nd)
        if nd.target == torch.ops.aten.copy.default:
            node0 = nd.args[0]
            if node0 in aten_empty_nodes:
                with gm.graph.inserting_after(nd):
                    dtype = node0.kwargs.get('dtype')
                    op_target = torch.ops.npu.npu_dtype_cast.default
                    args = (nd.args[1], dtype)
                    new_node = gm.graph.call_function(op_target, args=args)
                    new_node.name = nd.name
                nd.replace_all_uses_with(new_node)
                gm.graph.erase_node(nd)
                aten_empty_nodes.remove(node0)
                gm.graph.erase_node(node0)
                
    gm.recompile()

def get_last_node(gm: torch.fx.GraphModule):
    last_node = None
    for node in gm.graph.nodes:
        last_node = node
    return last_node

def fx_graph_op_types(gm: torch.fx.GraphModule) -> List[str]:
    op_types = []
    for nd in gm.graph.nodes:
        if nd.op not in ['call_function', 'call_method', 'call_module']:
            continue
        type_str = str(nd.target)
        if type_str.startswith(('aten', 'prims', 'npu')):
            op_types.append(type_str.split('.')[1])
    return op_types

def scalarize_tensor_ops_on_scalars(gm: torch.fx.GraphModule):
    # Modify gm.graph
    for node in gm.graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == "call_function":
            # The target attribute is the function
            # that call_function calls.
            # call_function[target=torch.ops.aten.add.Tensor](args = (%arg64_1, 1), kwargs = {})
            if node.target == torch.ops.aten.add.Tensor:
                if len(node.args) != 2 or node.kwargs != {}:
                    continue
                elif not isinstance(node.args[1], torch.fx.node.Node):
                    node.target = torch.ops.aten.add.Scalar
            if node.target == torch.ops.aten.mul.Tensor:
                if len(node.args) != 2 or node.kwargs != {}:
                    continue
                elif not isinstance(node.args[1], torch.fx.node.Node):
                    node.target = torch.ops.aten.mul.Scalar

    gm.graph.lint()  # Does some checks to make sure the

    # Recompile the forward() method of `gm` from its Graph
    gm.recompile()

def generate_fake_inputs(name_to_example_inputs):
    inputs_str = ""
    for name, example_input in name_to_example_inputs.items():
        input_str = (
            f"{name} = rand_strided("
            f"{codegen_python_shape_tuple(example_input.size())}, "
            f"{codegen_python_shape_tuple(example_input.stride())}, "
            f"device='{example_input.device}', dtype={example_input.dtype})"
        )
        inputs_str += f"        {input_str}\n"

    return inputs_str


def get_num_call_functions(graph):
    num_call_functions = 0
    for node in graph.graph.nodes:
        if node.op == "call_function" and node.target != torch.ops.aten.reshape.default:
            num_call_functions += 1
        if num_call_functions > 1:
            break
    return num_call_functions

class MLIRProcessor:
    def __init__(self, bisheng_install_path: str = None):
        """
        初始化MLIR处理器
        
        :param bisheng_install_path: Bisheng安装路径，默认从环境变量获取
        """
        self.bisheng_torch_mlir_path = f"bishengir-opt"
        
    def extract_function(self, module: ir.Module) -> func_dialect.FuncOp:
        """从MLIR模块中提取主函数并添加标记属性"""
        with module.context:
            for func in module.body.operations:
                if isinstance(func, func_dialect.FuncOp):
                    func.attributes["hacc.placeholder"] = ir.UnitAttr.get(func.context)
                    return func
        raise ValueError("No valid FuncOp found in module")
    
    def rebuild_mlir_module(self, module_str: str) -> ir.Module:
        """从字符串重新构建MLIR模块"""
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            torch_mlir.dialects.torch.register_dialect(ctx)
            return ir.Module.parse(module_str)
    
    def get_signature(self, func: func_dialect.FuncOp) -> tuple:
        """获取函数的签名信息：类型签名、输出数量和张量维度"""
        func_type = func.type
        signature = {}
        ranks = []
        
        # 处理输入+输出类型
        for i, tensor_type in enumerate(func_type.inputs + func_type.results):
            try:  # RankedTensorType
                signature[i] = '*' + str(tensor_type.element_type)
                ranks.append(len(tensor_type.shape))
            except AttributeError:  # ValueTensorType
                type_str = str(tensor_type)
                signature[i] = '*' + type_str.split(',')[-1].split('>')[0]
                # 从类型字符串中提取维度信息
                dim_start = type_str.find('[') + 1
                dim_end = type_str.find(']', dim_start)
                dim_str = type_str[dim_start:dim_end]
                ranks.append(dim_str.count(',') + 1 if dim_str else 1)
        
        num_outputs = len(func_type.results)
        return signature, num_outputs, ranks
    
    def process_mlir(self, 
                    module: Union[str, ir.Module], 
                    get_sig: bool = True, 
                    dynamic: bool = False) -> tuple:
        """
        处理MLIR模块的核心方法
        
        :param module: MLIR模块字符串或对象
        :param get_sig: 是否获取函数签名
        :param dynamic: 是否为动态执行模式
        :return: (函数字符串, 元数据字典)
        """
        if isinstance(module, str):
            module = self.rebuild_mlir_module(module)
        
        func = self.extract_function(module)
        kernel_info = None
        func_str = str(func)
        func_hash_str = func_str + "_host" if dynamic else func_str
        module_hash = hashlib.sha256(func_hash_str.encode()).hexdigest()
        logger.info(f"Generated kernel hash: {module_hash}")

        if get_sig:
            signature, num_outputs, ranks = self.get_signature(func)
            kernel_info = {
                "signature": signature,
                "ranks": ranks,
                'kernel_hash': module_hash,
            }
        
        return func_str, kernel_info
    
    def get_named_op_str(self,
                        module: Union[str, ir.Module],
                        kernel_name: str,
                        dynamic: bool = False) -> Dict[str, Any]:
        """
        获取命名操作格式的MLIR字符串
        
        :param module: MLIR模块字符串或对象
        :param kernel_name: 内核名称（用于临时文件）
        :param dynamic: 是否为动态执行模式
        :return: 包含处理结果和签名字典
        """
        func_str, sig_dict = self.process_mlir(module, get_sig=True, dynamic=dynamic)
        
        cleaned_func = func_str.replace(
            '"#hfusion.fusion_kind<PURE_ELEMWISE>"', 
            '#hfusion.fusion_kind<PURE_ELEMWISE>'
        )
        logger.debug(f"原始Linalg方言MLIR:\n{cleaned_func}")
        
        # 执行转换命令
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_mlir_path = os.path.join(tmpdir, f"{kernel_name}.mlir")
            with open(torch_mlir_path, 'w') as f:
                f.write(cleaned_func)

            cmd = (f"{self.bisheng_torch_mlir_path} "
                    "--torch-backend-to-named-op-backend-pipeline="
                    "\"ensure-no-implicit-broadcast=true\" "
                    f"{torch_mlir_path}")
        
            try:
                result = subprocess.check_output(
                    cmd, text=True, shell=True
                )
                # 过滤全局定义并更新函数属性
                processed_mlir = "\n".join(
                    line for line in result.splitlines() 
                    if "ml_program.global" not in line
                )
                
                # 根据模式设置函数属性
                func_attr = ("hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>" 
                            if dynamic else 
                            "hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>")
                processed_mlir = processed_mlir.replace("hacc.placeholder", func_attr)
                
                # 应用额外的数据类型处理（需实现mlir_match_and_replace_unsupported_dtypes）
                final_mlir = self._replace_unsupported_dtypes(processed_mlir)
                logger.debug(f"转换后的NamedOp方言MLIR:\n{final_mlir}")
                
                return final_mlir, sig_dict
            
            except subprocess.CalledProcessError as e:
                logger.error(f"命令执行失败: {cmd}\n错误: {e.output}")
                raise RuntimeError(f"MLIR转换失败: {e.stderr}") from e
    
    def _replace_unsupported_dtypes(self, mlir_text: str) -> str:
        """替换不支持的MLIR数据类型"""
        pattern1 = r"%(\d+) = arith\.truncf %(\w+) : f64 to bf16"
        matches1 = re.findall(pattern1, mlir_text)

        for var1, var2 in matches1:
            pattern2 = rf"%" + var2 + r" = arith\.constant (\d+(\.\d+)?) : f64"
            match2 = re.search(pattern2, mlir_text)
            if match2:
                mlir_text = re.sub(r': f64', ': f32', mlir_text)
        return mlir_text

def mlir_match_and_replace_unsupported_dtypes(mlir_text: str) -> str:
    pattern1 = r"%(\d+) = arith\.truncf %(\w+) : f64 to bf16"
    matches1 = re.findall(pattern1, mlir_text)

    for var1, var2 in matches1:
        pattern2 = rf"%" + var2 + r" = arith\.constant (\d+(\.\d+)?) : f64"
        match2 = re.search(pattern2, mlir_text)
        if match2:
            mlir_text = re.sub(r': f64', ': f32', mlir_text)
    return mlir_text


def to_folder(
        gm: torch.fx.GraphModule, 
        folder: Union[str, os.PathLike], 
        graph_hash: str,
        module_name: str = "FxModule"):
    """Dumps out module to ``folder`` with ``module_name`` so that it can be
    imported with ``from <folder> import <module_name>``

    Args:

        folder (Union[str, os.PathLike]): The folder to write the code out to

        module_name (str): Top-level name to use for the ``Module`` while
            writing out the code
    """
    folder = Path(folder)
    Path(folder).mkdir(exist_ok=True)
    tab = " " * 4
    custom_builtins = "\n".join([v.import_str for v in _custom_builtins.values()])
    model_str = f"""
import torch
{custom_builtins}

from torch.nn import *

class {module_name}(torch.nn.Module):
    def __init__(self):
        super().__init__()
"""

    def _gen_model_repr(module_name: str, module: torch.nn.Module) -> Optional[str]:
        safe_reprs = [
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
        ]
        if type(module) in safe_reprs:
            return f"{module.__repr__()}"
        else:
            return None

    blobified_modules = []
    for module_name, module in gm.named_children():
        module_str = _gen_model_repr(module_name, module)
        if module_str is None:
            module_file = folder / f"{module_name}.pt"
            torch.save(module, module_file)
            blobified_modules.append(module_name)
            module_repr = module.__repr__().replace("\r", " ").replace("\n", " ")
            # weights_only=False as this is legacy code that saves the model
            module_str = (
                f"torch.load(r'{module_file}', weights_only=False) # {module_repr}"
            )
        model_str += f"{tab * 2}self.{module_name} = {module_str}\n"

    for buffer_name, buffer in gm._buffers.items():
        if buffer is None:
            continue
        model_str += f"{tab * 2}self.register_buffer('{buffer_name}', torch.empty({list(buffer.shape)}, dtype={buffer.dtype}))\n"  # noqa: B950

    for param_name, param in gm._parameters.items():
        if param is None:
            continue
        model_str += f"{tab * 2}self.{param_name} = torch.nn.Parameter(torch.empty({list(param.shape)}, dtype={param.dtype}))\n"  # noqa: B950

    model_str += f"{_addindent(gm.code, 4)}\n"

    module_file = folder / f"{graph_hash}.py"
    module_file.write_text(model_str)

    if len(blobified_modules) > 0:
        warnings.warn(
            "Was not able to save the following children modules as reprs -"
            f"saved as pickled files instead: {blobified_modules}"
        )

def get_anir_mode():
    mode = os.getenv('ANIR_MODE', 'O1')

    if mode not in ["O0", "O1"]:
        raise ValueError(f"Invalid MODE value: {mode}. Allowed values are 'O0' and 'O1'.")
    return mode

def is_fx_dynamic(graph):
    for node in graph.graph.nodes:
        if node.op == "placeholder":
            if 'tensor_meta' in node.meta:
                shape = node.meta['tensor_meta'].shape
                if any(isinstance(dim, torch.SymInt) for dim in shape):
                    return True
        elif node.op == "call_function":
            if isinstance(node.meta['val'], torch.Tensor):
                if any(isinstance(dim, torch.SymInt) for dim in node.meta['val'].shape):
                    return True
    return False

def replace_placeholders(file_path: str, replacements: dict, placeholder_format: str = r'\{\{(\w+)\}\}') -> None:
    """
    替换文件中的占位符
    
    :param file_path: 文件路径
    :param replacements: 替换字典，如 {'function_body': 'your _code'}
    :param placeholder_format: 占位符正则表达式（默认匹配{{xxx}}）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = re.compile(placeholder_format)
    
    def replacer(match: re.Match) -> str:
        placeholder = match.group(1)
        replacement = replacements.get(placeholder, match.group(0))
        
        line_start = content.rfind('\n', 0, match.start()) + 1
        indent = re.match(r'^\s*', content[line_start:match.start()]).group(0)
        
        return '\n'.join([indent + line for line in replacement.split('\n')])
    
    new_content = pattern.sub(replacer, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean",
             device_type="npu"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param fast_flush: Use faster kernel to flush L2 cache between measurements
    :type fast_flush: bool, default is True
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all" Default is "mean".    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]
    import torch

    di = torch._dynamo.device_interface.get_interface_for_device(device_type)

    fn()
    di.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    cache_size = 256 * 1024 * 1024
    if fast_flush:
        cache = torch.empty(int(cache_size // 4), dtype=torch.int, device=device_type)
    else:
        cache = torch.empty(int(cache_size), dtype=torch.int8, device=device_type)

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    return torch.mean(times).item()

def _get_logger(*, level=logging.ERROR, file=None, name=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    try:
        import colorlog
    except ImportError:
        formatter = logging.Formatter(
            '[%(levelname)s] ANIR %(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s[%(levelname)s] ANIR %(asctime)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            })

    if file:
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

logger = _get_logger(name='anir')