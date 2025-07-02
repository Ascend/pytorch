import datetime
import re
import sys
import os
import site
import warnings
from collections import namedtuple

from torch.utils import collect_env as torch_collect_env

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

try:
    import torch_npu
    TORCH_NPU_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_NPU_AVAILABLE = False

# System Environment Information
SystemEnv = namedtuple('SystemEnv', [
    'torch_version',
    'torch_npu_version',
    'is_debug_build',
    'gcc_version',
    'clang_version',
    'cmake_version',
    'os',
    'libc_version',
    'python_version',
    'python_platform',
    'pip_version',  # 'pip' or 'pip3'
    'pip_packages',
    'conda_packages',
    'caching_allocator_config',
    'is_xnnpack_available',
    'cpu_info',
    'cann_version',
])


def get_torch_npu_install_path():
    path = ""
    site_packages = site.getsitepackages()
    if site_packages:
        path = site_packages[0]
    return path


def check_path_owner_consistent(path: str):
    if not os.path.exists(path):
        msg = f"The path does not exist: {path}"
        raise RuntimeError(msg)
    if os.stat(path).st_uid != os.getuid():
        warnings.warn(f"Warning: The {path} owner does not match the current owner.")


def check_directory_path_readable(path):
    check_path_owner_consistent(path)
    if os.path.islink(path):
        msg = f"Invalid path is a soft chain: {path}"
        raise RuntimeError(msg)
    if not os.access(path, os.R_OK):
        msg = f"The path permission check failed: {path}"
        raise RuntimeError(msg)


def get_cann_version():
    ascend_home_path = os.environ.get("ASCEND_HOME_PATH", "")
    cann_version = "not known"
    check_directory_path_readable(os.path.realpath(ascend_home_path))
    for dirpath, _, filenames in os.walk(os.path.realpath(ascend_home_path)):
        install_files = [file for file in filenames if re.match(r"ascend_.*_install\.info", file)]
        if install_files:
            filepath = os.path.realpath(os.path.join(dirpath, install_files[0]))
            check_directory_path_readable(filepath)
            with open(filepath, "r") as f:
                for line in f:
                    if line.find("version") != -1:
                        cann_version = line.strip().split("=")[-1]
                        break
    return cann_version


def get_torch_npu_version():
    torch_npu_version_str = 'N/A'
    if TORCH_NPU_AVAILABLE:
        torch_npu_version_str = torch_npu.__version__
    return torch_npu_version_str


def get_env_info():
    run_lambda = torch_collect_env.run
    pip_version, pip_list_output = torch_collect_env.get_pip_packages(run_lambda)

    if TORCH_AVAILABLE:
        version_str = torch.__version__
        debug_mode_str = str(torch.version.debug)
    else:
        version_str = debug_mode_str = 'N/A'

    sys_version = sys.version.replace("\n", " ")

    return SystemEnv(
        torch_version=version_str,
        torch_npu_version=get_torch_npu_version(),
        is_debug_build=debug_mode_str,
        python_version='{} ({}-bit runtime)'.format(sys_version, sys.maxsize.bit_length() + 1),
        python_platform=torch_collect_env.get_python_platform(),
        pip_version=pip_version,
        pip_packages=pip_list_output,
        conda_packages=torch_collect_env.get_conda_packages(run_lambda),
        os=torch_collect_env.get_os(run_lambda),
        libc_version=torch_collect_env.get_libc_version(),
        gcc_version=torch_collect_env.get_gcc_version(run_lambda),
        clang_version=torch_collect_env.get_clang_version(run_lambda),
        cmake_version=torch_collect_env.get_cmake_version(run_lambda),
        caching_allocator_config=torch_collect_env.get_cachingallocator_config(),
        is_xnnpack_available=torch_collect_env.is_xnnpack_available()
        if hasattr(torch_collect_env, "is_xnnpack_available") else "not known",
        cpu_info=torch_collect_env.get_cpu_info(run_lambda)
        if hasattr(torch_collect_env, "get_cpu_info") else "not known",
        cann_version=get_cann_version(),
    )


env_info_fmt = """
PyTorch version: {torch_version}
Torch-npu version: {torch_npu_version}
Is debug build: {is_debug_build}
OS: {os}
GCC version: {gcc_version}
Clang version: {clang_version}
CMake version: {cmake_version}
Libc version: {libc_version}

Python version: {python_version}
Python platform: {python_platform}
Is XNNPACK available: {is_xnnpack_available}

CPU:
{cpu_info}

CANN:
{cann_version}

Versions of relevant libraries:
{pip_packages}
{conda_packages}
""".strip()


def pretty_str(envinfo):
    def replace_nones(dct, replacement='Could not collect'):
        for key in dct.keys():
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    def replace_bools(dct, true='Yes', false='No'):
        for key in dct.keys():
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    def prepend(text, tag='[prepend]'):
        lines = text.split('\n')
        updated_lines = [tag + line for line in lines]
        return '\n'.join(updated_lines)

    def replace_if_empty(text, replacement='No relevant packages'):
        if text is not None and len(text) == 0:
            return replacement
        return text

    def maybe_start_on_next_line(string):
        # If `string` is multiline, prepend a \n to it.
        if string is not None and len(string.split('\n')) > 1:
            return '\n{}\n'.format(string)
        return string

    mutable_dict = envinfo._asdict()

    # Replace True with Yes, False with No
    mutable_dict = replace_bools(mutable_dict)

    # Replace all None objects with 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)

    # If either of these are '', replace with 'No relevant packages'
    mutable_dict['pip_packages'] = replace_if_empty(mutable_dict['pip_packages'])
    mutable_dict['conda_packages'] = replace_if_empty(mutable_dict['conda_packages'])

    # Tag conda and pip packages with a prefix
    # If they were previously None, they'll show up as ie '[conda] Could not collect'
    if mutable_dict['pip_packages']:
        mutable_dict['pip_packages'] = prepend(mutable_dict['pip_packages'],
                                               '[{}] '.format(envinfo.pip_version))
    if mutable_dict['conda_packages']:
        mutable_dict['conda_packages'] = prepend(mutable_dict['conda_packages'],
                                                 '[conda] ')
    mutable_dict['cpu_info'] = envinfo.cpu_info
    return env_info_fmt.format(**mutable_dict)


def get_pretty_env_info():
    return pretty_str(get_env_info())


def _add_collect_env_methods():
    torch.version.cann = get_cann_version()


def main():
    print("Collecting environment information...")
    output = get_pretty_env_info()
    print(output)

    if TORCH_AVAILABLE and hasattr(torch, 'utils') and hasattr(torch.utils, '_crash_handler'):
        minidump_dir = torch.utils._crash_handler.DEFAULT_MINIDUMP_DIR
        if sys.platform == "linux" and os.path.exists(minidump_dir):
            dumps = [os.path.join(minidump_dir, dump) for dump in os.listdir(minidump_dir)]
            latest = max(dumps, key=os.path.getctime)
            ctime = os.path.getctime(latest)
            creation_time = datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
            msg = (f"\n*** Detected a minidump at {latest} created on {creation_time}, "
                   "if this is related to your bug please include it when you file a report ***")
            print(msg, file=sys.stderr)


if __name__ == '__main__':
    main()
