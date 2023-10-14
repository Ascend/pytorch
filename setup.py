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
from wheel.bdist_wheel import bdist_wheel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY_PATH = os.path.join(BASE_DIR, "third_party")
VERSION = '2.1.0'
UNKNOWN = "Unknown"
DISABLE_TORCHAIR = "TRUE"
if os.environ.get("DISABLE_INSTALL_TORCHAIR") is not None:
    DISABLE_TORCHAIR = os.environ.get("DISABLE_INSTALL_TORCHAIR")
DISABLE_RPC = "TRUE" 
if os.environ.get("DISABLE_RPC_FRAMEWORK") is not None:
    DISABLE_RPC = os.environ.get("DISABLE_RPC_FRAMEWORK")

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

check_submodules()

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
        VERSION += "+git" + sha[:7]
    with os.fdopen(os.open(version_path, flags, modes), 'w') as f:
        f.write("__version__ = '{version}'\n".format(version=VERSION))
        f.write("git_version = {}\n".format(repr(sha)))


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
        return os.path.dirname(os.path.abspath(torch.__file__))
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
                    shutil.rmtree(filename, ignore_errors=True)
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
        build_type_dir = os.path.join(build_dir, get_build_type())
        output_lib_path = os.path.join(build_type_dir, "packages/torch_npu/lib")
        os.makedirs(build_type_dir, exist_ok=True)
        os.makedirs(output_lib_path, exist_ok=True)
        self.build_lib = os.path.relpath(os.path.join(build_dir, "packages/torch_npu"))
        self.build_temp = os.path.relpath(build_type_dir)

        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + get_build_type(),
            '-DCMAKE_INSTALL_PREFIX=' + os.path.abspath(output_lib_path),
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.abspath(output_lib_path),
            '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=' + os.path.abspath(output_lib_path),
            '-DTORCHNPU_INSTALL_LIBDIR=' + os.path.abspath(output_lib_path),
            '-DPYTHON_INCLUDE_DIR=' + get_paths()['include'],
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

        build_args = ['-j', str(multiprocessing.cpu_count())]

        subprocess.check_call([self.cmake, BASE_DIR] + cmake_args, cwd=build_type_dir, env=os.environ)
        subprocess.check_call(['make'] + build_args, cwd=build_type_dir, env=os.environ)


class Build(build_ext, object):

    def run(self):
        self.run_command('build_clib')
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))
        self.build_temp = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}"))
        self.library_dirs.append(
            os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_npu/lib")))
        super(Build, self).run()


class InstallCmd(install):

    def finalize_options(self) -> None:
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))
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
            os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_npu"),
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
            os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_npu/include/torch_npu"),
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
            os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_npu/include"),
            os.path.relpath(src, os.path.join(BASE_DIR, "patch/include")))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ret.append((src, dst))
    return ret


class EggInfoBuild(egg_info, object):
    def finalize_options(self):
        self.egg_base = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))
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
        torch_dependencies = ["libc10.so", "libtorch.so", "libtorch_cpu.so", "libtorch_python.so"]
        cann_dependencies = ["libhccl.so", "libascendcl.so", "libacl_op_compiler.so", "libge_runner.so",
                             "libgraph.so", "libacl_tdt_channel.so", "libfmk_parser.so", "libascend_protobuf.so"]
        other_dependencies = ["libtorch_npu.so", "libnpu_profiler.so", "libgomp.so.1"]

        dependencies = torch_dependencies + cann_dependencies + other_dependencies

        self.run_command('egg_info')
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
    '-Wno-return-type',
    '-D__FILENAME__=\"$(notdir $(abspath $<))\"'
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
    extra_link_args += ['-Wl,-z,now,-s']

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
]


setup(
        name=os.environ.get('TORCH_NPU_PACKAGE_NAME', 'torch_npu'),
        version=VERSION,
        description='NPU bridge for PyTorch',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://gitee.com/ascend/pytorch',
        download_url="https://gitee.com/ascend/pytorch/tags",
        license="BSD License",
        classifiers=classifiers,
        packages=["torch_npu"],
        libraries=[('torch_npu', {'sources': list()})],
        package_dir={'': os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))},
        ext_modules=[
            CppExtension(
                'torch_npu._C',
                sources=["torch_npu/csrc/InitNpuBindings.cpp"],
                libraries=["torch_npu"],
                include_dirs=include_directories,
                extra_compile_args=extra_compile_args + ['-fstack-protector-all'],
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
            'bdist_wheel': BdistWheelBuild,
            'egg_info': EggInfoBuild,
            'install': InstallCmd,
            'clean': Clean
        })
