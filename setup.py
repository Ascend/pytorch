import glob
import multiprocessing
import multiprocessing.pool
import os
import re
import shutil
import stat
import subprocess
import sys
import traceback
import platform
import time
import sysconfig
from sysconfig import get_paths
from pathlib import Path
from typing import Union
import hashlib

import distutils.ccompiler
import distutils.command.clean
from distutils.version import LooseVersion
from distutils.command.build_py import build_py
from setuptools import setup, distutils, Extension, find_packages
from setuptools.command.build_clib import build_clib
from setuptools.command.build_ext import build_ext
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install
from wheel.bdist_wheel import bdist_wheel

# Disable autoloading before running 'import torch' to avoid circular dependencies
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

from torchnpugen.utils import PathManager

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
THIRD_PARTY_PATH = os.path.join(BASE_DIR, "third_party")
PathManager.check_directory_path_readable(os.path.join(BASE_DIR, "version.txt"))
with open(os.path.join(BASE_DIR, "version.txt")) as version_f:
    VERSION = version_f.read().strip()
UNKNOWN = "Unknown"
BUILD_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP

DISABLE_TORCHAIR = "FALSE"
if os.environ.get("DISABLE_INSTALL_TORCHAIR") is not None:
    DISABLE_TORCHAIR = os.environ.get("DISABLE_INSTALL_TORCHAIR")
DISABLE_RPC = "FALSE"
if os.environ.get("DISABLE_RPC_FRAMEWORK") is not None:
    DISABLE_RPC = os.environ.get("DISABLE_RPC_FRAMEWORK")
ENABLE_LTO = "FALSE"
if os.environ.get("ENABLE_LTO") is not None:
    ENABLE_LTO = os.environ.get("ENABLE_LTO")
PGO_MODE = 0
if os.environ.get("PGO_MODE") is not None:
    PGO_MODE = int(os.environ.get("PGO_MODE"))

USE_CXX11_ABI = False
if platform.machine() == "aarch64":
    # change to use cxx11.abi in default since 2.6 (arm)
    USE_CXX11_ABI = True

if os.environ.get("_GLIBCXX_USE_CXX11_ABI") is not None:
    if os.environ.get("_GLIBCXX_USE_CXX11_ABI") == "1":
        USE_CXX11_ABI = True
        if platform.machine() == "x86_64":
            VERSION += "+cxx11-abi"
    else:
        USE_CXX11_ABI = False


def get_submodule_folders():
    git_modules_path = os.path.join(BASE_DIR, ".gitmodules")
    default_modules_path = [
        os.path.join(THIRD_PARTY_PATH, name)
        for name in [
            "op-plugin",
        ]
    ]
    if not os.path.exists(git_modules_path):
        return default_modules_path
    with open(git_modules_path) as f:
        return [
            os.path.join(BASE_DIR, line.split("=", 1)[1].strip())
            for line in f.readlines()
            if line.strip().startswith("path")
        ]


def check_submodules():
    def not_exists_or_empty(folder):
        return not os.path.exists(folder) or (
            os.path.isdir(folder) and len(os.listdir(folder)) == 0
        )

    folders = get_submodule_folders()
    # If none of the submodule folders exists, try to initialize them
    if all(not_exists_or_empty(folder) for folder in folders):
        try:
            print(" --- Trying to initialize submodules")
            start = time.time()
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=BASE_DIR)  # Compliant
            end = time.time()
            print(f" --- Submodule initialization took {end - start:.2f} sec")
        except Exception:
            print(" --- Submodule initalization failed")
            print("Please run:\n\tgit submodule init && git submodule update")
            sys.exit(1)


def file_sha256(filename):
    """calculate file sha256"""
    with open(filename, "rb") as f:
        sha256 = hashlib.sha256()
        sha256.update(f.read())
        hash_value = sha256.hexdigest()
        return hash_value


def download_miniz():
    # 设置基础路径
    miniz_url = "https://gitee.com/mirrors/pytorch/raw/v2.6.0/third_party/miniz-3.0.2/miniz.h"
    miniz_dir = os.path.join(BASE_DIR, "third_party/miniz-3.0.2")
    
    if os.path.exists(miniz_dir):  # 检查目录是否存在
        try:
            shutil.rmtree(miniz_dir)  # 递归删除整个目录及内容
            print(f"has cleaned {miniz_dir}")
        except Exception as e:
            print(f"clean {miniz_dir} failed, error: {str(e)}")
            raise RuntimeError(f"clean {miniz_dir} failed") from e
    
    os.makedirs(miniz_dir, exist_ok=True)  # 重建空目录

    # 获取wget绝对路径
    wget_path = shutil.which("wget")
    if not wget_path:
        raise RuntimeError("wget not found, please install wget")

    # 使用绝对路径下载文件
    subprocess.check_call([
        wget_path, 
        miniz_url,
        "--no-check-certificate",
        "-O", os.path.join(miniz_dir, "miniz.h")
    ])
    miniz_hash256 = file_sha256(os.path.join(miniz_dir, "miniz.h"))
    if miniz_hash256 != "f959f5dfb5c5d3ed0f55f3e7e455afbe1e924d64d74cd2dd374740b9d87abfd0":
        raise RuntimeError("the sha256sum of miniz.h is not incorrect.")
    

check_submodules()
download_miniz()


def get_sha(pytorch_root: Union[str, Path]) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=pytorch_root)  # Compliant
            .decode("ascii")
            .strip()
        )
    except Exception:
        return UNKNOWN


def generate_torch_npu_version():
    torch_npu_root = Path(__file__).parent
    version_path = torch_npu_root / "torch_npu" / "version.py"
    if version_path.exists():
        version_path.unlink()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    sha = get_sha(torch_npu_root)
    if os.getenv("BUILD_WITHOUT_SHA") is None:
        global VERSION
        if USE_CXX11_ABI and platform.machine() == "x86_64":
            VERSION += ".git" + sha[:7]
        else:
            VERSION += "+git" + sha[:7]
    with os.fdopen(os.open(version_path, flags, modes), 'w') as f:
        f.write("__version__ = '{version}'\n".format(version=VERSION))
        f.write("git_version = {}\n".format(repr(sha)))
    os.chmod(version_path, mode=stat.S_IRUSR | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)


generate_torch_npu_version()


def read_triton_ascend_req():
    path = os.path.join(BASE_DIR, "triton_version.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        if sys.platform == 'win32':
            exts = os.environ.get('PATHEXT', '').split(os.pathsep)
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None


def get_cmake_command():
    def _get_version(cmd):
        for line in subprocess.check_output([cmd, '--version']).decode('utf-8').split('\n'):
            if 'version' in line:
                return LooseVersion(line.strip().split(' ')[2])
        raise RuntimeError('no version found')
    "Returns cmake command."
    cmake_command = 'cmake'
    if platform.system() == 'Windows':
        return cmake_command
    cmake3 = which('cmake3')
    cmake = which('cmake')
    if cmake3 is not None and _get_version(cmake3) >= LooseVersion("3.18.0"):
        cmake_command = 'cmake3'
        return cmake_command
    elif cmake is not None and _get_version(cmake) >= LooseVersion("3.18.0"):
        return cmake_command
    else:
        raise RuntimeError('no cmake or cmake3 with version >= 3.18.0 found')


def get_build_type():
    build_type = 'Release'
    if os.getenv('DEBUG', default='0').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']:
        build_type = 'Debug'

    if os.getenv('REL_WITH_DEB_INFO', default='0').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']:
        build_type = 'RelWithDebInfo'

    return build_type


def _get_build_mode():
    for i in range(1, len(sys.argv)):
        if not sys.argv[i].startswith('-'):
            return sys.argv[i]

    raise RuntimeError("Run setup.py without build mode.")


def get_pytorch_dir():
    try:
        import torch
        return os.path.dirname(os.path.realpath(torch.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)


def generate_bindings_code(base_dir):
    python_execute = sys.executable
    generate_code_cmd = ["bash", os.path.join(base_dir, 'generate_code.sh'), python_execute, VERSION]
    if subprocess.call(generate_code_cmd) != 0:  # Compliant
        print(
            'Failed to generate ATEN bindings: {}'.format(generate_code_cmd),
            file=sys.stderr)
        sys.exit(1)


def build_stub(base_dir):
    build_stub_cmd = ["sh", os.path.join(base_dir, 'third_party/acl/libs/build_stub.sh')]
    if subprocess.call(build_stub_cmd) != 0:
        print(
            'Failed to build stub: {}'.format(build_stub_cmd),
            file=sys.stderr)
        sys.exit(1)


def check_torchair_valid(base_dir):
    # build with submodule of torchair, if path of torchair is valid
    torchair_path = os.path.join(base_dir, 'third_party/torchair/torchair')
    return os.path.exists(torchair_path) and (
        os.path.isdir(torchair_path) and len(os.listdir(torchair_path)) != 0
    )


def check_tensorpipe_valid(base_dir):
    tensorpipe_path = os.path.join(base_dir, 'third_party/Tensorpipe/tensorpipe')
    return os.path.exists(tensorpipe_path)


def generate_dbg_files_and_strip():
    library_dir = Path(BASE_DIR).joinpath("build/packages/torch_npu")
    dbg_dir = Path(BASE_DIR).joinpath("build/dbg")
    os.makedirs(dbg_dir, exist_ok=True)
    library_files = [Path(i) for i in library_dir.rglob('*.so')]
    for library_file in library_files:
        subprocess.check_call(["eu-strip", library_file, "-f",
                                str(dbg_dir.joinpath(library_file.name)) + ".debug"], cwd=BASE_DIR)  # Compliant


def patchelf_dynamic_library():
    library_dir = Path(BASE_DIR).joinpath("build/packages/torch_npu/lib")
    library_files = [str(i) for i in library_dir.rglob('*.so')]
    for library_file in library_files:
        subprocess.check_call(["patchelf", "--remove-needed", "libgomp.so.1", library_file], cwd=BASE_DIR)  # Compliant


def CppExtension(name, sources, *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for C++.
    '''
    pytorch_dir = get_pytorch_dir()
    temp_include_dirs = kwargs.get('include_dirs', [])
    temp_include_dirs.append(os.path.join(pytorch_dir, 'include'))
    temp_include_dirs.append(os.path.join(pytorch_dir, 'include/torch/csrc/api/include'))
    kwargs['include_dirs'] = temp_include_dirs

    temp_library_dirs = kwargs.get('library_dirs', [])
    temp_library_dirs.append(os.path.join(pytorch_dir, 'lib'))
    temp_library_dirs.append(os.path.join(BASE_DIR, "third_party/acl/libs"))
    kwargs['library_dirs'] = temp_library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    libraries.append('hccl')
    kwargs['libraries'] = libraries
    kwargs['language'] = 'c++'
    return Extension(name, sources, *args, **kwargs)


class Clean(distutils.command.clean.clean):

    def run(self):
        f_ignore = open('.gitignore', 'r')
        ignores = f_ignore.read()
        pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
        for wildcard in filter(None, ignores.split('\n')):
            match = pat.match(wildcard)
            if match:
                if match.group(1):
                    # Marker is found and stop reading .gitignore.
                    break
                # Ignore lines which begin with '#'.
            else:
                for filename in glob.glob(wildcard):
                    if os.path.islink(filename):
                        raise RuntimeError(f"Failed to remove path: {filename}")
                    if os.path.exists(filename):
                        try:
                            shutil.rmtree(filename, ignore_errors=True)
                        except Exception as err:
                            raise RuntimeError(f"Failed to remove path: {filename}") from err
        f_ignore.close()

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)

        remove_files = [
            'torch_npu/csrc/aten/RegisterCPU.cpp',
            'torch_npu/csrc/aten/RegisterNPU.cpp',
            'torch_npu/csrc/aten/RegisterAutogradNPU.cpp',
            'torch_npu/csrc/aten/NPUNativeFunctions.h',
            'torch_npu/csrc/aten/CustomRegisterSchema.cpp',
            'torch_npu/csrc/aten/ForeachRegister.cpp',
            'torch_npu/utils/custom_ops.py',
            'torch_npu/version.py',
        ]
        for remove_file in remove_files:
            file_path = os.path.join(BASE_DIR, remove_file)
            if os.path.exists(file_path):
                os.remove(file_path)


class CPPLibBuild(build_clib, object):
    def run(self):
        cmake = get_cmake_command()

        if cmake is None:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        self.cmake = cmake

        build_dir = os.path.join(BASE_DIR, "build")
        build_type_dir = os.path.join(build_dir)
        output_lib_path = os.path.join(build_type_dir, "packages/torch_npu/lib")
        os.makedirs(build_type_dir, exist_ok=True)
        os.chmod(build_type_dir, mode=BUILD_PERMISSION)
        os.makedirs(output_lib_path, exist_ok=True)
        self.build_lib = os.path.relpath(os.path.join(build_dir, "packages/torch_npu"))
        self.build_temp = os.path.relpath(build_type_dir)

        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + get_build_type(),
            '-DCMAKE_INSTALL_PREFIX=' + os.path.realpath(output_lib_path),
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.realpath(output_lib_path),
            '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=' + os.path.realpath(output_lib_path),
            '-DTORCHNPU_INSTALL_LIBDIR=' + os.path.realpath(output_lib_path),
            '-DPYTHON_INCLUDE_DIR=' + get_paths().get('include'),
            '-DTORCH_VERSION=' + VERSION,
            '-DPYTORCH_INSTALL_DIR=' + get_pytorch_dir()]

        if DISABLE_TORCHAIR == 'FALSE':
            if check_torchair_valid(BASE_DIR):
                cmake_args.append('-DBUILD_TORCHAIR=on')
                torchair_install_prefix = os.path.join(build_type_dir, "packages/torch_npu/dynamo/torchair")
                cmake_args.append(f'-DTORCHAIR_INSTALL_PREFIX={torchair_install_prefix}')
                cmake_args.append(f'-DTORCHAIR_TARGET_PYTHON={sys.executable}')

        if DISABLE_RPC == 'FALSE':
            if check_tensorpipe_valid(BASE_DIR):
                cmake_args.append('-DBUILD_TENSORPIPE=on')
        
        if ENABLE_LTO == "TRUE":
            cmake_args.append('-DENABLE_LTO=on')
        if PGO_MODE != 0:
            cmake_args.append('-DPGO_MODE=' + str(PGO_MODE))
        
        if USE_CXX11_ABI:
            cmake_args.append('-DGLIBCXX_USE_CXX11_ABI=1')

        if os.getenv('_ABI_VERSION') is not None:
            cmake_args.append('-DABI_VERSION=' + os.getenv('_ABI_VERSION'))

        max_jobs = os.getenv("MAX_JOBS", str(multiprocessing.cpu_count()))
        build_args = ['-j', max_jobs]

        subprocess.check_call([self.cmake, BASE_DIR] + cmake_args, cwd=build_type_dir, env=os.environ)
        for base_dir, dirs, files in os.walk(build_type_dir):
            for dir_name in dirs:
                dir_path = os.path.join(base_dir, dir_name)
                os.chmod(dir_path, mode=BUILD_PERMISSION)
            for file_name in files:
                file_path = os.path.join(base_dir, file_name)
                os.chmod(file_path, mode=BUILD_PERMISSION)

        subprocess.check_call(['make'] + build_args, cwd=build_type_dir, env=os.environ)


class Build(build_ext, object):

    def run(self):
        self.run_command('build_clib')
        self.run_command('build_py')
        # proceed with the normal build_ext process
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, "build/packages"))
        self.build_temp = os.path.relpath(os.path.join(BASE_DIR, "build/temp"))
        self.library_dirs.append(
            os.path.relpath(os.path.join(BASE_DIR, "build/packages/torch_npu/lib"))
        )
        super(Build, self).run()


class InstallCmd(install):

    def finalize_options(self) -> None:
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, "build/packages"))
        return super(InstallCmd, self).finalize_options()


def add_ops_files(base_dir, file_list):
    # add ops header files
    plugin_path = os.path.join(base_dir, 'third_party/op-plugin/op_plugin/include')
    if os.path.exists(plugin_path):
        file_list.append('third_party/op-plugin/op_plugin/include/*.h')
    return


def add_ops_python_files(ret_list):
    # add ops python files
    opplugin_path = os.path.join(BASE_DIR, 'third_party/op-plugin/op_plugin/python')

    if os.path.exists(opplugin_path):
        ops_python_files = glob.glob(os.path.join(opplugin_path, '**/*.py'), recursive=True)
        for src in ops_python_files:
            dst = os.path.join(
                os.path.join(BASE_DIR, "build/packages/torch_npu/op_plugin"),
                os.path.relpath(src, opplugin_path))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            ret_list.append((src, dst))
    return


def get_src_py_and_dst():
    ret = []
    generated_python_files = glob.glob(
        os.path.join(BASE_DIR, "torch_npu", '**/*.py'),
        recursive=True) + glob.glob(
        os.path.join(BASE_DIR, "torch_npu", '**/*.yaml'),
        recursive=True) + glob.glob(
        os.path.join(BASE_DIR, "torch_npu", 'acl*.json'),
        recursive=True) + glob.glob(
        os.path.join(BASE_DIR, "torch_npu", 'contrib/apis_config.json'),
        recursive=True)
    for src in generated_python_files:
        dst = os.path.join(
            os.path.join(BASE_DIR, "build/packages/torch_npu"),
            os.path.relpath(src, os.path.join(BASE_DIR, "torch_npu")))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ret.append((src, dst))

    add_ops_python_files(ret)

    header_files = [
        "torch_npu/csrc/*.h",
        "torch_npu/csrc/*/*.h",
        "torch_npu/csrc/*/*.hpp",
        "torch_npu/csrc/*/*/*.h",
        "torch_npu/csrc/*/*/*/*.h",
        "torch_npu/csrc/*/*/*/*/*.h",
        "third_party/acl/inc/*/*.h",
        "third_party/hccl/inc/*/*.h",
        "third_party/acl/inc/*/*/*.h",
        "torch_npu/csrc/distributed/HCCLUtils.hpp",
        "torch_npu/csrc/distributed/ProcessGroupHCCL.hpp",
    ]
    add_ops_files(BASE_DIR, header_files)
    glob_header_files = []
    for regex_pattern in header_files:
        glob_header_files += glob.glob(os.path.join(BASE_DIR, regex_pattern), recursive=True)

    for src in glob_header_files:
        dst = os.path.join(
            os.path.join(BASE_DIR, "build/packages/torch_npu/include/torch_npu"),
            os.path.relpath(src, os.path.join(BASE_DIR, "torch_npu")))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ret.append((src, dst))

    torch_header_files = [
        "*/*.h",
        "*/*/*.h",
        "*/*/*/*.h",
        "*/*/*/*/*.h",
        "*/*/*/*/*/*.h"
    ]
    torch_glob_header_files = []
    for regex_pattern in torch_header_files:
        torch_glob_header_files += glob.glob(os.path.join(BASE_DIR, "patch/include", regex_pattern), recursive=True)

    for src in torch_glob_header_files:
        dst = os.path.join(
            os.path.join(BASE_DIR, "build/packages/torch_npu/include"),
            os.path.relpath(src, os.path.join(BASE_DIR, "patch/include")))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ret.append((src, dst))
    
    aot_inductor_files = [
        # Follow torch v2.6.0.
        # These aoti_runtime/*.cpp don't compile to libtorch_npu,
        # but act like header files when generate cppwrapper in aot-inductor.
        "torch_npu/_inductor/codegen/aoti_runtime/*.cpp"
    ]
    glob_aoti_files = []
    for regex_pattern in aot_inductor_files:
        glob_aoti_files += glob.glob(
            os.path.join(BASE_DIR, regex_pattern), recursive=True
        )

    for src in glob_aoti_files:
        # Dst: torch_npu/_inductor/codegen/aoti_runtime/*.cpp
        dst = os.path.join(
            os.path.join(BASE_DIR, "build/packages/torch_npu/"),
            os.path.relpath(src, os.path.join(BASE_DIR, "torch_npu")),
        )
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ret.append((src, dst))


    def add_torch_npu_codegen(codegen_src_dir, codegen_dst_dir, exclude_root_init=None):
        """
        复制codegen目录下的文件到目标路径
        :param codegen_src_dir: 源codegen目录
        :param codegen_dst_dir: 目标目录
        :param exclude_root_init: 需要排除根目录__init__.py的源目录(仅过滤该目录下的__init__.py)
        """
        # 匹配需要复制的文件类型
        codegen_files = glob.glob(
            os.path.join(codegen_src_dir, '**/*.py'), recursive=True
        ) + glob.glob(
            os.path.join(codegen_src_dir, '**/*.yaml'), recursive=True
        ) + glob.glob(
            os.path.join(codegen_src_dir, '**/*.json'), recursive=True
        ) + glob.glob(
            os.path.join(codegen_src_dir, '**/*.cpp'), recursive=True
        ) + glob.glob(
            os.path.join(codegen_src_dir, '**/*.h'), recursive=True
        )

        # 按原目录结构复制到目标路径
        for src in codegen_files:
            # 仅过滤指定目录下的根级__init__.py
            if (exclude_root_init is not None and 
                os.path.basename(src) == '__init__.py' and 
                os.path.dirname(src) == exclude_root_init):
                continue  # 跳过op-plugin/codegen根目录的__init__.py
            
            # 计算目标路径（保留原目录层级）
            dst = os.path.join(
                codegen_dst_dir,
                os.path.relpath(src, codegen_src_dir)  # 保留torchnpugen内部的目录层级
            )
            print(os.path.relpath(src, codegen_src_dir))
            # 确保目标目录存在
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # 加入文件复制列表
            ret.append((src, dst))

    # 新增：提前创建 torchnpugen 根目录
    torchnpugen_root = os.path.join(BASE_DIR, "build/packages/torchnpugen")
    os.makedirs(torchnpugen_root, exist_ok=True)
    # 将codegen复制到package路径
    codegen_src_dir = os.path.join(BASE_DIR, "torchnpugen")
    codegen_dst_dir = os.path.join(BASE_DIR, "build/packages/torchnpugen")
    # 复制torch_npu的torchnpugen
    add_torch_npu_codegen(codegen_src_dir, codegen_dst_dir)
    # 复制op-plugin的torchnpugen（仅过滤其根目录的__init__.py）
    op_plugin_codegen_src = os.path.join(BASE_DIR, "third_party/op-plugin/torchnpugen")
    add_torch_npu_codegen(
        op_plugin_codegen_src,
        codegen_dst_dir,
        exclude_root_init=op_plugin_codegen_src  # 指定要过滤根目录__init__.py的源目录
    )

    return ret


class EggInfoBuild(egg_info, object):
    def finalize_options(self):
        self.egg_base = os.path.relpath(os.path.join(BASE_DIR, "build/packages"))
        ret = get_src_py_and_dst()
        for src, dst in ret:
            self.copy_file(src, dst)
        super(EggInfoBuild, self).finalize_options()


class PythonPackageBuild(build_py, object):
    def run(self) -> None:
        ret = get_src_py_and_dst()
        for src, dst in ret:
            self.copy_file(src, dst)
        super(PythonPackageBuild, self).finalize_options()


class BdistWheelBuild(bdist_wheel):
    def run(self):
        if which('patchelf') is not None:
            patchelf_dynamic_library()

        if not DEBUG and which('eu-strip') is not None:
            generate_dbg_files_and_strip()

        torch_dependencies = ["libc10.so", "libtorch.so", "libtorch_cpu.so", "libtorch_python.so"]
        cann_dependencies = ["libhccl.so", "libascendcl.so", "libacl_op_compiler.so", "libge_runner.so",
                             "libgraph.so", "libacl_tdt_channel.so", "libfmk_parser.so", "libascend_protobuf.so",
                             "libascend_ml.so"]
        other_dependencies = ["libtorch_npu.so", "libnpu_profiler.so", "libgomp.so.1", "libatb.so"]

        dependencies = torch_dependencies + cann_dependencies + other_dependencies

        bdist_wheel.run(self)

        if is_manylinux:
            file = glob.glob(os.path.join(self.dist_dir, "*linux*.whl"))[0]

            auditwheel_cmd = ["auditwheel", "-v", "repair", "-w", self.dist_dir, file]
            for i in dependencies:
                auditwheel_cmd += ["--exclude", i]

            try:
                subprocess.run(auditwheel_cmd, check=True, stdout=subprocess.PIPE)
            finally:
                os.remove(file)


build_mode = _get_build_mode()
if build_mode not in ['clean']:
    # Generate bindings code, including RegisterNPU.cpp & NPUNativeFunctions.h.
    generate_bindings_code(BASE_DIR)
    if Path(BASE_DIR).joinpath("third_party/Tensorpipe/third_party/acl/libs").exists():
        build_stub(Path(BASE_DIR).joinpath("third_party/Tensorpipe"))
    build_stub(BASE_DIR)

# Setup include directories folders.
include_directories = [
    BASE_DIR,
    os.path.join(BASE_DIR, 'patch/include'),
    os.path.join(BASE_DIR, 'third_party/hccl/inc'),
    os.path.join(BASE_DIR, 'third_party/acl/inc')
]

extra_link_args = []

DEBUG = (os.getenv('DEBUG', default='').upper() in ['ON', '1', 'YES', 'TRUE', 'Y'])

extra_compile_args = [
    '-std=c++17',
    '-Wno-sign-compare',
    '-Wno-deprecated-declarations',
    '-Wno-return-type'
]

if re.match(r'clang', os.getenv('CC', '')):
    extra_compile_args += [
        '-Wno-macro-redefined',
        '-Wno-return-std-move',
    ]

if DEBUG:
    extra_compile_args += ['-O0', '-g']
    extra_link_args += ['-O0', '-g', '-Wl,-z,now']
else:
    extra_compile_args += ['-DNDEBUG']
    extra_link_args += ['-Wl,-z,now']

# valid manylinux tags
manylinux_tags = [
    "manylinux1_x86_64",
    "manylinux2010_x86_64",
    "manylinux2014_x86_64",
    "manylinux2014_aarch64",
    "manylinux_2_5_x86_64",
    "manylinux_2_12_x86_64",
    "manylinux_2_17_x86_64",
    "manylinux_2_17_aarch64",
    "manylinux_2_24_x86_64",
    "manylinux_2_24_aarch64",
    "manylinux_2_27_x86_64",
    "manylinux_2_27_aarch64",
    "manylinux_2_28_x86_64",
    "manylinux_2_28_aarch64",
    "manylinux_2_31_x86_64",
    "manylinux_2_31_aarch64",
    "manylinux_2_34_x86_64",
    "manylinux_2_34_aarch64",
    "manylinux_2_35_x86_64"
    "manylinux_2_35_aarch64",
]
is_manylinux = os.environ.get("AUDITWHEEL_PLAT", None) in manylinux_tags

readme = os.path.join(BASE_DIR, "README.md")
if not os.path.exists(readme):
    raise FileNotFoundError("Unable to find 'README.md'")
with open(readme, encoding="utf-8") as fdesc:
    long_description = fdesc.read()

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX :: Linux",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

requirements = ['torch==2.6.0+cpu' if platform.machine() == 'x86_64' else 'torch==2.6.0']
if USE_CXX11_ABI:
    requirements = ['torch==2.6.0+cpu.cxx11.abi'] if platform.machine() == 'x86_64' else ['torch==2.6.0']
triton_ascend_req = read_triton_ascend_req()
if triton_ascend_req is not None:
    requirements.append(triton_ascend_req)

ext_modules = [CppExtension(
            'torch_npu._C',
            sources=["torch_npu/csrc/InitNpuBindings.cpp"],
            libraries=["torch_npu"],
            include_dirs=include_directories,
            extra_compile_args=extra_compile_args + ['-fstack-protector-all'] + [
                '-D__FILENAME__=\"InitNpuBindings.cpp\"'],
            library_dirs=["lib"],
            extra_link_args=extra_link_args + ['-Wl,-rpath,$ORIGIN/lib', '-Wl,-Bsymbolic-functions'],
            define_macros=[('_GLIBCXX_USE_CXX11_ABI', '1' if USE_CXX11_ABI else '0'),
                           ('GLIBCXX_USE_CXX11_ABI', '1' if USE_CXX11_ABI else '0')]
        )]

setup(
    name=os.environ.get('TORCH_NPU_PACKAGE_NAME', 'torch_npu'),
    version=VERSION,
    description='NPU bridge for PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD License",
    classifiers=classifiers,
    packages=["torch_npu", "torchnpugen"],
    libraries=[('torch_npu', {'sources': list()})],
    package_dir={'': os.path.relpath(os.path.join(BASE_DIR, "build/packages"))},
    ext_modules=ext_modules,
    install_requires=requirements,
    extras_require={
    },
    package_data={
        'torch_npu': [
            '*.so',
            'lib/*.so*',
        ],
        'torchnpugen': [
            '*.py', '**/*.py',
            '*.yaml', '**/*.yaml',
            '*.json', '**/*.json',
            '*.cpp', '**/*.cpp',
            '*.h', '**/*.h',
        ],
    },
    cmdclass={
        'build_clib': CPPLibBuild,
        'build_ext': Build,
        'build_py': PythonPackageBuild,
        'bdist_wheel': BdistWheelBuild,
        'install': InstallCmd,
        'clean': Clean
    },
    entry_points={
        'console_scripts': [
            'torch_npu_run = torch_npu.distributed.run:_main',
        ],
        'torch.backends': [
            'torch_npu = torch_npu:_autoload',
        ],
    }
)