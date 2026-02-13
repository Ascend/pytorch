import os
import shutil
import subprocess
import stat
import sysconfig
import sys
import functools
import threading
from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

import torch
import torch_npu
import torch.distributed as dist

# 获取脚本绝对路径
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
asc_path = os.getenv("ASCEND_HOME_PATH", "")
if not asc_path:
    raise RuntimeError("ASCEND_HOME_PATH is not set, source <ascend-toolkit>/set_env.sh first")

torch_npu_path = os.path.dirname(os.path.realpath(torch_npu.__file__))

# 创建全局安装目录
INSTALL_DIR = BASE_DIR / "ascend_npu_ir"
LIB_DIR = INSTALL_DIR / "lib"
os.makedirs(LIB_DIR, mode=0o775, exist_ok=True)

def run_once(func):
    result = None
    has_run = False
    lock = threading.Lock()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal result, has_run
        
        if has_run:
            return result
            
        with lock:
            if not has_run:
                result = func(*args, **kwargs)
                has_run = True
                
        return result
        
    return wrapper

def get_cxx_compiler():
    """获取C++编译器路径"""
    cxx = os.environ.get("CXX") or os.environ.get("CC")
    if cxx:
        return cxx
    for compiler in ["clang++", "g++"]:
        if path := shutil.which(compiler):
            return path
    raise RuntimeError("Failed to find C++ compiler (tried clang++, g++)")

@run_once
def anir_build_libcpp_common(so_path):
    """构建共享库（带缓存检查）"""
    src_path = BASE_DIR / "ascend_npu_ir" / "cpp_common" / "cpp_common.cpp"
    
    # 获取Python头文件路径
    scheme = sysconfig.get_default_scheme()
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include = sysconfig.get_paths(scheme=scheme)["include"]
    
    # 构建编译命令
    cc_cmd = [
        get_cxx_compiler(),
        str(src_path),
        f"-I{py_include}",
        f"-I{BASE_DIR / 'ascend_npu_ir' / '_C' / 'include'}",
        f"-I{asc_path}/include",
        f"-L{asc_path}/lib64",
        f"-I{os.path.dirname(os.path.realpath(torch.__file__))}/include",
        f"-I{os.path.join(torch_npu_path, 'include')}",
        f"-L{os.path.join(torch_npu_path, 'lib')}",
        "-lruntime", "-lascendcl", "-ltorch_npu", "-lprofapi",
        "-std=c++17", 
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-fPIC", "-shared", "-o", str(so_path)
    ]
    
    # 执行编译
    print(f"Executing: {' '.join(cc_cmd)}")
    if (ret := subprocess.call(cc_cmd)) != 0:
        raise RuntimeError(f"Build failed with code {ret}")
    print(f"Successfully built: {so_path}")

@run_once
def anir_build_pybind_extension():
    """构建Python扩展模块"""
    # 确保扩展模块被构建到正确的目录
    build_lib_dir = str(INSTALL_DIR)
    
    extension = Pybind11Extension(
        '_C', 
        [str(BASE_DIR / 'ascend_npu_ir' / '_C' / 'extension.cpp')], 
        include_dirs=[
            str(BASE_DIR / 'ascend_npu_ir' / '_C' / 'include'),
            f'{asc_path}/include'
        ],
        library_dirs=[
            f'{asc_path}/lib64',
            str(LIB_DIR)
        ],
        libraries=['runtime', 'cpp_common'],
        extra_link_args=[
            f'-Wl,-rpath,{asc_path}/lib64',
            f'-Wl,-rpath,{LIB_DIR}'
        ],
        extra_compile_args=["-std=c++17"],
    )
    
    # 切换到项目根目录进行构建
    original_cwd = os.getcwd()
    os.chdir(BASE_DIR)
    
    try:
        setup(
            name="ascend_npu_ir",
            version="0.1",
            ext_modules=[extension],
            script_args=["build_ext", f"--build-lib={build_lib_dir}"],
        )
    finally:
        os.chdir(original_cwd)  # 恢复原始工作目录
        

def get_local_rank() -> int:
    """获取当前进程在 本机(node) 的进程编号 local_rank，兜底兼容所有启动方式"""
    # 适配 torchrun/accelerate/launch.py 等所有分布式启动方式
    local_rank = os.environ.get("LOCAL_RANK", -1)
    if local_rank != -1:
        return int(local_rank)
    # 未初始化分布式环境，默认本机主进程
    return 0


def is_node_main_process() -> bool:
    """
    ✅ 核心判断：是否是【本机(node)的主进程】
    返回 True  → 当前进程是本机的 local_rank=0 (节点主进程)
    返回 False → 当前进程是本机的普通进程
    """
    if not dist.is_available() or not dist.is_initialized():
        return True  # 单机非分布式，默认是主进程
    return get_local_rank() == 0


def node_main_process_only(func):
    """
    装饰器：每个节点(Node)仅 local_rank=0 执行函数，本节点其他进程等待执行完成
    ✅ 无返回值专用（mkdir/print/download/创建目录等）
    ✅ 多机并行执行，各节点互不阻塞，效率最优
    """
    def wrapper(*args, **kwargs):
        result = None
        # 判断：是否是当前节点的主进程
        if is_node_main_process():
            result = func(*args, **kwargs)
        
        # 分布式环境下：节点内所有进程同步，等待本节点主进程执行完毕
        if dist.is_available() and dist.is_initialized():
            try:
                # 这里的barrier是【节点内同步】，不是全局同步
                dist.barrier()
            except Exception:
                # 异常兜底，防止个别进程通信失败导致死锁
                pass
        return result
    return wrapper


@node_main_process_only
def build_ascend_npu_ir_ext():
    try:
        so_path = LIB_DIR / "libcpp_common.so"
        if not so_path.exists():
            print(f"Building libcpp_common.so at {so_path}")
            anir_build_libcpp_common(so_path)
            anir_build_pybind_extension()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
def set_torch_npu_library_path():
    try:
        # 尝试导入 torch_npu 模块
        import torch_npu
        
        # 获取 torch_npu 模块的安装路径
        torch_npu_path = os.path.dirname(torch_npu.__file__)
        
        # 构建库路径
        lib_path = os.path.join(torch_npu_path, '_inductor', 'ascend_npu_ir', 'ascend_npu_ir', 'lib')
        
        # 检查路径是否存在
        if os.path.exists(lib_path):
            # 获取当前的 LD_LIBRARY_PATH
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            
            # 添加新的库路径到开头
            new_ld_path = f"{lib_path}:{current_ld_path}"
            
            # 设置环境变量
            os.environ['LD_LIBRARY_PATH'] = new_ld_path
            return True
        else:
            print(f"Library path does not exist: {lib_path}")
            return False
            
    except ImportError:
        print("torch_npu module not found")
        return False
    except Exception as e:
        print(f"Error setting library path: {e}")
        return False

if __name__ == "__main__":
    build_ascend_npu_ir_ext()