# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
from pathlib import Path
from typing import Union

import distutils.ccompiler
import distutils.command.clean
from sysconfig import get_paths
from distutils.version import LooseVersion
from distutils.command.build_py import build_py
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools import setup, distutils, Extension
from setuptools.command.build_clib import build_clib
from setuptools.command.egg_info import egg_info

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
THIRD_PARTY_PATH = os.path.join(BASE_DIR, "third_party")
VERSION = '1.11.0.post12'
UNKNOWN = "Unknown"
BUILD_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
call_once_flag = 0


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
            subprocess.check_call(["git", "submodule", "init"], cwd=BASE_DIR)  # Compliant
            subprocess.check_call(["git", "submodule", "update"], cwd=BASE_DIR)  # Compliant
            end = time.time()
            print(f" --- Submodule initialization took {end - start:.2f} sec")
        except Exception:
            print(" --- Submodule initalization failed")
            print("Please run:\n\tgit submodule init && git submodule update")
            sys.exit(1)


check_submodules()


def get_sha(pytorch_root: Union[str, Path]) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=pytorch_root)
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
    if os.getenv("BUILD_WITHOUT_SHA") is None:
        global VERSION
        sha = get_sha(torch_npu_root)
        VERSION += "+git" + sha[:7]
    with os.fdopen(os.open(version_path, flags, modes), 'w') as f:
        f.write("__version__ = '{version}'\n".format(version=VERSION))
    os.chmod(version_path, mode=stat.S_IRUSR | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)


generate_torch_npu_version()


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
    if cmake3 is not None and _get_version(cmake3) >= LooseVersion("3.12.0"):
        cmake_command = 'cmake3'
        return cmake_command
    elif cmake is not None and _get_version(cmake) >= LooseVersion("3.12.0"):
        return cmake_command
    else:
        raise RuntimeError('no cmake or cmake3 with version >= 3.12.0 found')


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
    if subprocess.call(generate_code_cmd) != 0:
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


def generate_dbg_files_and_strip():
    global call_once_flag
    if call_once_flag == 1:
        return
    library_dir = Path(BASE_DIR).joinpath("build/packages/torch_npu")
    dbg_dir = Path(BASE_DIR).joinpath("build/dbg")
    os.makedirs(dbg_dir, exist_ok=True)
    library_files = [Path(i) for i in library_dir.rglob('*.so')]
    for library_file in library_files:
        subprocess.check_call(["eu-strip", library_file, "-f",
                                str(dbg_dir.joinpath(library_file.name)) + ".debug"], cwd=BASE_DIR)  # Compliant
    call_once_flag = 1


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
            'torch_npu/csrc/aten/RegisterUnsupprotNPU.cpp',
            'torch_npu/csrc/aten/NPUNativeFunctions.h',
            'torch_npu/csrc/aten/python_custom_functions.cpp',
            'torch_npu/utils/torch_funcs.py',
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


        build_args = ['-j', str(multiprocessing.cpu_count())]

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
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, "build/packages"))
        self.build_temp = os.path.relpath(os.path.join(BASE_DIR, "build"))
        self.library_dirs.append(
            os.path.relpath(os.path.join(BASE_DIR, "build/packages/torch_npu/lib")))
        super(Build, self).run()
        if which('eu-strip') is not None:
            generate_dbg_files_and_strip()


class InstallCmd(install):

    def finalize_options(self) -> None:
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, "build/packages"))
        return super(InstallCmd, self).finalize_options()


def get_src_py_and_dst():
    ret = []
    generated_python_files = glob.glob(
        os.path.join(BASE_DIR, "torch_npu", '**/*.py'),
        recursive=True) + glob.glob(
        os.path.join(BASE_DIR, "torch_npu", '**/*.yaml'),
        recursive=True)
    for src in generated_python_files:
        dst = os.path.join(
            os.path.join(BASE_DIR, "build/packages/torch_npu"),
            os.path.relpath(src, os.path.join(BASE_DIR, "torch_npu")))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ret.append((src, dst))

    header_files = [
        "torch_npu/csrc/*.h",
        "torch_npu/csrc/*/*.h",
        "torch_npu/csrc/*/*/*.h",
        "torch_npu/csrc/*/*/*/*.h",
        "torch_npu/csrc/*/*/*/*/*.h",
        "third_party/acl/inc/*/*.h",
        "third_party/acl/inc/*/*/*.h"
    ]
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


build_mode = _get_build_mode()
if build_mode not in ['clean']:
    # Generate bindings code, including RegisterNPU.cpp & NPUNativeFunctions.h.
    generate_bindings_code(BASE_DIR)
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
    '-std=c++14',
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


setup(
    name=os.environ.get('TORCH_NPU_PACKAGE_NAME', 'torch_npu'),
    version=VERSION,
    description='NPU bridge for PyTorch',
    packages=["torch_npu"],
    libraries=[('torch_npu', {'sources': list()})],
    package_dir={'': os.path.relpath(os.path.join(BASE_DIR, "build/packages"))},
    ext_modules=[
        CppExtension(
            'torch_npu._C',
            sources=["torch_npu/csrc/InitNpuBindings.cpp"],
            libraries=["torch_npu"],
            include_dirs=include_directories,
            extra_compile_args=extra_compile_args + ['-fstack-protector-all'] + ['-D__FILENAME__=\"InitNpuBindings.cpp\"'],
            library_dirs=["lib"],
            extra_link_args=extra_link_args + ['-Wl,-rpath,$ORIGIN/lib'],
        ),
    ],
    extras_require={
    },
    package_data={
        'torch_npu': [
            '*.so', 'lib/*.so*',
        ],
    },
    cmdclass={
        'build_clib': CPPLibBuild,
        'build_ext': Build,
        'build_py': PythonPackageBuild,
        'egg_info': EggInfoBuild,
        'clean': Clean,
        'install': InstallCmd
    })
