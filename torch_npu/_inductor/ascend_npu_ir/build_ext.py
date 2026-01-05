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
os.makedirs(LIB_DIR, exist_ok=True)
os.chmod(LIB_DIR, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)

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

def main_process_only(func):
    """
    装饰器：仅 rank 0 执行函数，其他进程等待。
    适用于无返回值或无需返回值的函数（如 mkdir, print, download 等）
    """
    def wrapper(*args, **kwargs):
        if dist.is_initialized():
            if dist.get_rank() == 0:
                result = func(*args, **kwargs)
            else:
                result = None  # 非主进程不执行
            dist.barrier()  # 同步：等待 rank 0 完成
            return result
        else:
            # 非分布式环境直接执行
            return func(*args, **kwargs)
    return wrapper


@main_process_only
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