import shutil
import subprocess

__all__ = []


def check_cann_package_support_export_db() -> bool:
    COMMAND_SUCCESS = 0
    try:
        msprof_path = shutil.which("msprof")
        if not msprof_path:
            return False
        completed_process = subprocess.run([msprof_path, "--help"], capture_output=True, shell=False, text=True)
        if completed_process.returncode != COMMAND_SUCCESS:
            return False
        return "--type" in completed_process.stdout
    except Exception:
        return False


class CannPackageManager:
    SUPPORT_EXPORT_DB = None

    @classmethod
    def is_support_export_db(cls) -> bool:
        if cls.SUPPORT_EXPORT_DB is None:
            cls.SUPPORT_EXPORT_DB = check_cann_package_support_export_db()
        return cls.SUPPORT_EXPORT_DB
