import datetime
import re
import sys
import os
from collections import namedtuple

import torch
from torch.utils import collect_env as torch_collect_env


# System Environment Information
SystemEnv = namedtuple('SystemEnv', [
    'torch_version',
    'torch_npu_version',
    'is_debug_build',
    'cuda_compiled_version',
    'gcc_version',
    'clang_version',
    'cmake_version',
    'os',
    'libc_version',
    'python_version',
    'python_platform',
    'is_cuda_available',
    'cuda_runtime_version',
    'cuda_module_loading',
    'nvidia_driver_version',
    'nvidia_gpu_models',
    'cudnn_version',
    'pip_version',  # 'pip' or 'pip3'
    'pip_packages',
    'conda_packages',
    'hip_compiled_version',
    'hip_runtime_version',
    'miopen_runtime_version',
    'caching_allocator_config',
    'is_xnnpack_available',
    'cpu_info',
    'cann_version',
    'cann_driver_version',
    'npu_mapping_info',
    'npu_count_info',
])

def get_cann_version():
    ascend_home_path = os.environ.get("ASCEND_HOME_PATH", "")
    cann_version = "not known"
    for dirpath, _, filenames in os.walk(os.path.realpath(ascend_home_path)):
        install_files = [file for file in filenames if re.match(r"ascend_.*_install\.info", file)]
        if len(install_files) == 0:
            install_files = [file for file in filenames if re.match(r"version*.cfg", file)]
        if install_files:
            filepath = os.path.join(dirpath, install_files[0])
            with open(filepath, "r") as f:
                cann_version = " ".join(f.readlines())
                break
    return cann_version

def get_npu_board_info(run_lambda):
    npu_smi = " npu-smi info -t board -i 0"
    rc, out, _ = run_lambda(npu_smi)
    if rc != 0:
        return "not known"
    return out

def get_npu_mapping_info(run_lambda):
    npu_smi = " npu-smi info -m"
    rc, out, _ = run_lambda(npu_smi)
    if rc != 0:
        return "not known"
    return out

def get_npu_count_info(run_lambda):
    npu_smi = " npu-smi info -l"
    rc, out, _ = run_lambda(npu_smi)
    if rc != 0:
        return "not known"
    return out

def get_torch_npu_version():
    try:
        import torch_npu
        TORCH_NPU_AVAILABLE = True
    except (ImportError, NameError, AttributeError, OSError):
        TORCH_NPU_AVAILABLE = False
    torch_npu_version_str = 'N/A'
    if TORCH_NPU_AVAILABLE:
        torch_npu_version_str = torch_npu.__version__
        if hasattr(torch_npu, "version") and hasattr(torch_npu.version, "git_version"):
            torch_npu_version_str = torch_npu_version_str + "+git" + torch_npu.version.git_version[:7]
    return torch_npu_version_str

def get_env_info():
    run_lambda = torch_collect_env.run
    pip_version, pip_list_output = torch_collect_env.get_pip_packages(run_lambda)


    version_str = torch.__version__
    debug_mode_str = str(torch.version.debug)
    cuda_available_str = str(torch.cuda.is_available())
    cuda_version_str = torch.version.cuda
    if not hasattr(torch.version, 'hip') or torch.version.hip is None:  # cuda version
        hip_compiled_version = hip_runtime_version = miopen_runtime_version = 'N/A'
    else:  # HIP version
        cfg = torch._C._show_config().split('\n')
        hip_runtime_version = [s.rsplit(None, 1)[-1] for s in cfg if 'HIP Runtime' in s][0]
        miopen_runtime_version = [s.rsplit(None, 1)[-1] for s in cfg if 'MIOpen' in s][0]
        cuda_version_str = 'N/A'
        hip_compiled_version = torch.version.hip

    sys_version = sys.version.replace("\n", " ")

    return SystemEnv(
        torch_version=version_str,
        torch_npu_version=get_torch_npu_version(),
        is_debug_build=debug_mode_str,
        python_version='{} ({}-bit runtime)'.format(sys_version, sys.maxsize.bit_length() + 1),
        python_platform=torch_collect_env.get_python_platform(),
        is_cuda_available=cuda_available_str,
        cuda_compiled_version=cuda_version_str,
        cuda_runtime_version=torch_collect_env.get_running_cuda_version(run_lambda),
        cuda_module_loading=torch_collect_env.get_cuda_module_loading_config() 
            if hasattr(torch_collect_env, "get_cuda_module_loading_config") else "not known",
        nvidia_gpu_models=torch_collect_env.get_gpu_info(run_lambda),
        nvidia_driver_version=torch_collect_env.get_nvidia_driver_version(run_lambda),
        cudnn_version=torch_collect_env.get_cudnn_version(run_lambda),
        hip_compiled_version=hip_compiled_version,
        hip_runtime_version=hip_runtime_version,
        miopen_runtime_version=miopen_runtime_version,
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
        cann_driver_version=get_npu_board_info(run_lambda),
        npu_mapping_info=get_npu_mapping_info(run_lambda),
        npu_count_info=get_npu_count_info(run_lambda),
    )

env_info_fmt = """
PyTorch version: {torch_version}
Torch-npu version: {torch_npu_version}
Is debug build: {is_debug_build}
CUDA used to build PyTorch: {cuda_compiled_version}
ROCM used to build PyTorch: {hip_compiled_version}

OS: {os}
GCC version: {gcc_version}
Clang version: {clang_version}
CMake version: {cmake_version}
Libc version: {libc_version}

Python version: {python_version}
Python platform: {python_platform}
Is CUDA available: {is_cuda_available}
CUDA runtime version: {cuda_runtime_version}
CUDA_MODULE_LOADING set to: {cuda_module_loading}
GPU models and configuration: {nvidia_gpu_models}
Nvidia driver version: {nvidia_driver_version}
cuDNN version: {cudnn_version}
HIP runtime version: {hip_runtime_version}
MIOpen runtime version: {miopen_runtime_version}
Is XNNPACK available: {is_xnnpack_available}

CPU:
{cpu_info}

CANN:
{cann_version}

CANN driver:
{cann_driver_version}

NPU mapping info:
{npu_mapping_info}

NPU count info:
{npu_count_info}

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

    # If nvidia_gpu_models is multiline, start on the next line
    mutable_dict['nvidia_gpu_models'] = \
        maybe_start_on_next_line(envinfo.nvidia_gpu_models)

    # If the machine doesn't have CUDA, report some fields as 'No CUDA'
    dynamic_cuda_fields = [
        'cuda_runtime_version',
        'nvidia_gpu_models',
        'nvidia_driver_version',
    ]
    all_cuda_fields = dynamic_cuda_fields + ['cudnn_version']
    all_dynamic_cuda_fields_missing = all(
        mutable_dict[field] is None for field in dynamic_cuda_fields)
    if not torch.cuda.is_available() and all_dynamic_cuda_fields_missing:
        for field in all_cuda_fields:
            mutable_dict[field] = 'No CUDA'
        if envinfo.cuda_compiled_version is None:
            mutable_dict['cuda_compiled_version'] = 'None'

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


def main():
    print("Collecting environment information...")
    output = get_pretty_env_info()
    print(output)

    if hasattr(torch, 'utils') and hasattr(torch.utils, '_crash_handler'):
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
