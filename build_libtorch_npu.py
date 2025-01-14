import multiprocessing
import glob
import os
import re
import shutil
import subprocess
import sys
import traceback
import platform
from pathlib import Path

from sysconfig import get_paths
from distutils.version import LooseVersion
from distutils import file_util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION = '2.1.0'


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


def clean_generated_files():
    f_ignore = open(os.path.join(BASE_DIR, '.gitignore'), 'r')
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
    clean_files = ['torch_npu/csrc/aten/RegisterCPU.cpp', 'torch_npu/csrc/aten/RegisterNPU.cpp',
                   'torch_npu/csrc/aten/RegisterAutogradNPU.cpp', 'torch_npu/csrc/aten/NPUNativeFunctions.h']
    for file in clean_files:
        if os.path.exists(os.path.join(BASE_DIR, file)):
            os.remove(os.path.join(BASE_DIR, file))
    if os.path.exists(os.path.join(BASE_DIR, "libtorch")):
        shutil.rmtree(os.path.join(BASE_DIR, "libtorch"))


def check_gtest_valid(base_dir):
    # validation of GoogleTest path.
    gtest_path = os.path.join(base_dir, 'third_party/googletest/CMakeLists.txt')
    return os.path.exists(gtest_path)


def generate_dbg_files_and_strip():
    library_dir = Path(BASE_DIR).joinpath("build/packages/torch_npu")
    dbg_dir = Path(BASE_DIR).joinpath("build/dbg_libtorch")
    os.makedirs(dbg_dir, exist_ok=True)
    library_files = [Path(i) for i in library_dir.rglob('*.so')]
    for library_file in library_files:
        subprocess.check_call(["eu-strip", library_file, "-f",
                                str(dbg_dir.joinpath(library_file.name)) + ".debug"], cwd=BASE_DIR)  # Compliant


def run_cmake():
    cmake = get_cmake_command()

    if cmake is None:
        raise RuntimeError(
            "CMake must be installed to build the following extensions: ")

    build_dir = os.path.join(BASE_DIR, "build")
    build_type_dir = os.path.join(build_dir)
    output_lib_path = os.path.join(build_type_dir, "packages/torch_npu/lib")
    os.makedirs(build_type_dir, exist_ok=True)
    os.makedirs(output_lib_path, exist_ok=True)
    cmake_args = [
        '-DCMAKE_BUILD_TYPE=' + get_build_type(),
        '-DCMAKE_INSTALL_PREFIX=' + os.path.abspath(output_lib_path),
        '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.abspath(output_lib_path),
        '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=' + os.path.abspath(output_lib_path),
        '-DTORCHNPU_INSTALL_LIBDIR=' + os.path.abspath(output_lib_path),
        '-DPYTHON_INCLUDE_DIR=' + get_paths().get('include'),
        '-DPYTORCH_INSTALL_DIR=' + get_pytorch_dir(),
        '-DTORCH_VERSION=' + VERSION,
        '-DBUILD_LIBTORCH=' + "ON"]

    if check_opplugin_valid(BASE_DIR):
        cmake_args.append('-DBUILD_OPPLUGIN=on')
        cmake_args.append('-DBUILD_NEW_HEADER=on')

    if check_tensorpipe_valid(BASE_DIR):
        cmake_args.append('-DBUILD_TENSORPIPE=on')

    if os.getenv('_GLIBCXX_USE_CXX11_ABI') is not None:
        cmake_args.append('-DGLIBCXX_USE_CXX11_ABI=' + os.getenv('_GLIBCXX_USE_CXX11_ABI'))
    
    if os.getenv('_ABI_VERSION') is not None:
        cmake_args.append('-DABI_VERSION=' + os.getenv('_ABI_VERSION'))

    build_args = ['-j', str(multiprocessing.cpu_count())]

    subprocess.check_call([cmake, BASE_DIR] + cmake_args, cwd=build_type_dir, env=os.environ)
    subprocess.check_call(['make'] + build_args, cwd=build_type_dir, env=os.environ)
    if which('eu-strip') is not None:
        generate_dbg_files_and_strip()


def check_opplugin_valid(base_dir):
    # build with submodule of op_plugin, if path of op-plugin is valid
    op_plugin_path = os.path.join(base_dir, 'third_party/op-plugin/op_plugin')
    return os.path.exists(op_plugin_path)


def check_tensorpipe_valid(base_dir):
    tensorpipe_path = os.path.join(base_dir, 'third_party/Tensorpipe/tensorpipe')
    return os.path.exists(tensorpipe_path)


def copy_file(infile, outfile, preserve_mode=1, preserve_times=1, link=None, level=1):
    """Copy a file respecting verbose, dry-run and force flags.  (The
    former two default to whatever is in the Distribution object, and
    the latter defaults to false for commands that don't define it.)
    """
    return file_util.copy_file(infile, outfile, preserve_mode, preserve_times, True, link)


def move_special_hpp(ret):
    hpp_name = "torch_npu.h"
    for _, dst in ret:
        if dst.endswith(hpp_name):
            dir_name = os.path.join(os.path.dirname(dst), "../../" + hpp_name)
            copy_file(dst, dir_name)
            return


def copy_hpp():
    def get_src_py_and_dst():
        ret = []

        header_files = [
            "torch_npu/csrc/aten/*.h",
            "torch_npu/csrc/aten/*/*.h",
            "torch_npu/csrc/aten/*/*/*.h",
            "torch_npu/csrc/core/*.h",
            "torch_npu/csrc/core/*/*.h",
            "torch_npu/csrc/core/*/*/*.h",
            "torch_npu/csrc/framework/*.h",
            "torch_npu/csrc/framework/*/*.h",
            "torch_npu/csrc/framework/*/*/*.h",
            "torch_npu/csrc/libs/*.h"
        ]
        glob_header_files = []
        for regex_pattern in header_files:
            glob_header_files += glob.glob(os.path.join(BASE_DIR, regex_pattern), recursive=True)

        for src in glob_header_files:
            dst = os.path.join(
                os.path.join(BASE_DIR, "libtorch_npu/include/torch_npu"),
                os.path.relpath(src, os.path.join(BASE_DIR, "torch_npu")))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            ret.append((src, dst))

        return ret
    ret = get_src_py_and_dst()
    for src, dst in ret:
        copy_file(src, dst)
    move_special_hpp(ret)


def copy_lib():
    lib_file = ["build/packages/torch_npu/lib/*.*"]
    glob_lib_files = []

    for regex_pattern in lib_file:
        glob_lib_files += glob.glob(os.path.join(BASE_DIR, regex_pattern), recursive=True)
    dst_path = os.path.join(BASE_DIR, "libtorch_npu/lib")
    os.makedirs(dst_path, exist_ok=True)

    for src in glob_lib_files:
        _, src_file_name = os.path.split(src)
        copy_file(src, os.path.join(dst_path, src_file_name))


def copy_cmake():
    cmake_file = os.path.join(BASE_DIR, "build/Torch_npuConfig.cmake")
    dst_path = os.path.join(BASE_DIR, "libtorch_npu/share/cmake/Torch_npu")
    os.makedirs(dst_path, exist_ok=True)
    copy_file(cmake_file, dst_path)


def build_libtorch_npu():
    clean_generated_files()
    generate_bindings_code(BASE_DIR)
    if Path(BASE_DIR).joinpath("third_party/Tensorpipe/third_party/acl/libs").exists():
        build_stub(Path(BASE_DIR).joinpath("third_party/Tensorpipe"))
    build_stub(BASE_DIR)
    run_cmake()
    copy_hpp()
    copy_lib()
    copy_cmake()


if __name__ == "__main__":
    build_libtorch_npu()
