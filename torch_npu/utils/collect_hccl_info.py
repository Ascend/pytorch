import os
import subprocess
import shutil
import argparse

from torch_npu.utils._error_code import ErrCode, pta_error


if 'ASCEND_HOME_PATH' not in os.environ:
    raise RuntimeError("Please run 'source set_env.sh' in the CANN installation path." + pta_error(ErrCode.NOT_FOUND))
ascend_dir = os.environ['ASCEND_HOME_PATH']


def get_tool_path():
    tool_path = os.path.join(ascend_dir, "tools", "hccl_test")
    if os.path.exists(tool_path):
        return tool_path
    else:
        raise RuntimeError("""HCCL test directory doesn't exist.
                           Please check the integrity of CANN package.""" + pta_error(ErrCode.NOT_FOUND))


def get_mpi_install_path():
    mpirun_path = shutil.which("mpirun")
    if not mpirun_path:
        raise FileNotFoundError(
            """MPI package not found. Please download from official website.
            If package already downloaded, please check and set environment variables.""" + pta_error(ErrCode.NOT_FOUND)
        )

    mpi_install_path_list = mpirun_path.decode().strip().split(os.sep)
    mpi_install_path = os.sep
    bin_index = mpi_install_path_list.index("bin")
    for sub_path in mpi_install_path_list[:bin_index]:
        mpi_install_path = os.path.join(mpi_install_path, sub_path)
    return mpi_install_path


build_args = ['-C', get_tool_path(),
              'MPI_HOME=' + get_mpi_install_path(),
              'ASCEND_DIR=' + ascend_dir]


def is_compiled():
    executable_path = os.path.join(get_tool_path(), 'bin')
    if os.path.exists(executable_path) and len(os.listdir(executable_path)):
        return True
    return False


def compile_hccl_test():
    make_path = shutil.which("make")
    if not make_path:
        raise FileNotFoundError("Command 'make' not found. please check and set environment variables." +
                                pta_error(ErrCode.NOT_FOUND))

    try:
        subprocess.check_call(args=[make_path] + build_args, env=os.environ, shell=False)
    except subprocess.CalledProcessError:
        print("HCCL test compile fail.")


"""
-t: test suite type. e.g: -t all_reduce_test denotes running all reduce test.
-b: begin size of data flow. e.g: -b 8k denotes data flow begins with 8KB
-e: end size of data flow. e.g: -e 64M denotes data flow ends with 64MB
-i: step bytes. increment size.
-f: ratio of increment. e.g: -f 2 denotes data flow increases exponentially
-d: data type. e.g: -d fp32 denotes dtype is float32.
-o: operation type. Legal: sum/prod/min/max
-n: iteration count.
-r: root
-w: Iters of warm up. e.g: -w 3 denotes number of warmup is 3.
-c: result verification. e.g: 0 disabled, 1 enabled
-p: number of npus: e.g: -n denotes 8 use 8 NPUs per node.
-h: help info
-file: host file to enable multi-node test
-multinode: whether to use multi-node test. e.g: False: disable, True: enable
"""

parser = argparse.ArgumentParser(description="test options")
parser.add_argument("--t", default="all_reduce_test", help="test suite type")
parser.add_argument("--b", default="8K", help="begin size of data flow")
parser.add_argument("--e", default="64M", help="end size of data flow")
parser.add_argument("--i", help="increment size")
parser.add_argument("--f", default="2", help="ratio of increment")
parser.add_argument("--d", default="fp32", help="data type")
parser.add_argument("--o", help="operation type")
parser.add_argument("--n", help="iteration count")
parser.add_argument("--r", help="root")
parser.add_argument("--w", help="warmup iterators")
parser.add_argument("--c", help="result verification")
parser.add_argument("--p", default="8", help="num of NPUs per node")
parser.add_argument("--h", help="help")

# Below are options if multiple nodes tests are enabled
parser.add_argument("--file", help="host file used by mpirun in multi-node cases.")

# option to execute single node test or multi-node test
parser.add_argument("--multinode", default="False", help="num of nodes.")
args = parser.parse_args()


def get_exe_hccl_test():
    return os.path.join(get_tool_path(), "bin", args.t)


def execute_hccl_test_single_node():
    args_dict = vars(args)
    comm_op_type = get_exe_hccl_test()
    exe_args = [comm_op_type]
    for key, val in args_dict.items():
        if key == "t" or val is None:
            continue
        if key == 'multinode' or key == "file":
            continue
        exe_args.extend(['-' + key, val])
    if args_dict["multinode"] == "False":
        try:
            subprocess.check_call(args=[shutil.which("mpirun")] + exe_args, shell=False)
        except subprocess.CalledProcessError:
            print("HCCL test executes fail.")
    else:
        subprocess.check_call(args=[shutil.which("mpirun"), "-f", args_dict["file"]] + exe_args, shell=False)


if __name__ == "__main__":
    if not is_compiled():
        compile_hccl_test()
    print("Executing HCCL test! Current test suite is: \n", args.t)
    execute_hccl_test_single_node()
