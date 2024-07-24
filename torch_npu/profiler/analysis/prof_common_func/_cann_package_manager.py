import shutil
import subprocess

from ._constant import print_error_msg

__all__ = []


class CannPackageManager:
    @classmethod
    def cann_package_support_export_db(cls) -> bool:
        err_msg = "Failed to check if current CANN package version support export db!"
        try:
            msprof_path = shutil.which("msprof")
            if not msprof_path:
                print_error_msg(f"{err_msg} msprof command not found!")
                raise RuntimeError(f"{err_msg} msprof command not found!")
            
            COMMAND_SUCCESS = 0
            completed_process = subprocess.run([msprof_path, "--help"], capture_output=True, shell=False, text=True)
            if completed_process.returncode != COMMAND_SUCCESS:
                print_error_msg(f"{err_msg} Failed to run command: msprof --help!")
                raise RuntimeError(f"{err_msg} Failed to run command: msprof --help!")
            return "--type" in completed_process.stdout
        except Exception as err:
            raise RuntimeError(err_msg) from err
