import shutil
import subprocess
from ..prof_common_func._path_manager import ProfilerPathManager

__all__ = []


def check_msprof_help_output(search_text: str) -> bool:
    COMMAND_SUCCESS = 0
    try:
        msprof_path = shutil.which("msprof")
        if not msprof_path:
            return False
        
        if not ProfilerPathManager.check_path_permission(msprof_path):
            return False
            
        completed_process = subprocess.run([msprof_path, "--help"], capture_output=True, shell=False, text=True)
        if completed_process.returncode != COMMAND_SUCCESS:
            return False
        return search_text in completed_process.stdout
    except Exception:
        return False


def check_cann_package_support_export_db() -> bool:
    return check_msprof_help_output("--type")


def check_cann_package_support_default_export_db() -> bool:
    return check_msprof_help_output("text(which will also export the database)")


class CannPackageManager:
    SUPPORT_EXPORT_DB = None
    SUPPORT_DEFAULT_EXPORT_DB = None

    @classmethod
    def is_support_export_db(cls) -> bool:
        if cls.SUPPORT_EXPORT_DB is None:
            cls.SUPPORT_EXPORT_DB = check_cann_package_support_export_db()
        return cls.SUPPORT_EXPORT_DB

    @classmethod
    def is_support_default_export_db(cls) -> bool:
        if cls.SUPPORT_DEFAULT_EXPORT_DB is None:
            cls.SUPPORT_DEFAULT_EXPORT_DB = check_cann_package_support_default_export_db()
        return cls.SUPPORT_DEFAULT_EXPORT_DB
