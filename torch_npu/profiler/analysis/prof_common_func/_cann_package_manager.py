import os
import shutil
import subprocess

__all__ = []

MSPROF_PY_RELATIVE_PATH = "tools/profiler/profiler_tool/analysis/msprof/msprof.py"
# 与 msprof bin 一致：未显式指定 --python-path 时使用 PATH 中的 python3
MSPROF_PYTHON_CMD = "python3"
# 探针超时：正常都在秒级返回，超时按"不支持"处理，避免解析流程被挂住
MSPROF_HELP_TIMEOUT = 30
MSPROF_PY_PROBE_TIMEOUT = 60


def check_msprof_help_output(search_text: str, timeout: int = None) -> bool:
    COMMAND_SUCCESS = 0
    try:
        msprof_path = shutil.which("msprof")
        if not msprof_path:
            return False

        completed_process = subprocess.run(
            [msprof_path, "--help"], capture_output=True, shell=False, text=True, timeout=timeout
        )
        if completed_process.returncode != COMMAND_SUCCESS:
            return False
        return search_text in completed_process.stdout
    except Exception:
        return False


def find_msprof_py_path(msprof_path: str = "") -> str:
    msprof_path = msprof_path or shutil.which("msprof")
    if not msprof_path:
        return ""
    pre_path = os.path.realpath(msprof_path.strip()).split("tools")[0]
    full_path = os.path.join(pre_path, MSPROF_PY_RELATIVE_PATH)
    return full_path if os.path.exists(full_path) else ""


def check_cann_package_support_default_export_db() -> bool:
    return check_msprof_help_output("text(which will also export the database)")


def check_cann_package_support_export_all() -> bool:
    # 新 bin 的 --type 支持 all：一次调用内部并行导出 timeline 和 summary。
    return check_msprof_help_output("include text, db, all", MSPROF_HELP_TIMEOUT)


def check_analysis_package_support_export_all(msprof_py_path: str) -> bool:
    # 与 bin 的 IsExportAllSupported() 一致：用 export all --help 的退出码判定解析包能力。
    # 新 bin 配老解析包时 bin 会回退成 timeline + summary 两段导出且不写 finish.txt，
    # 这里必须一起判定，否则上层会一直等完成标记。
    if not msprof_py_path:
        return False
    try:
        completed_process = subprocess.run(
            [MSPROF_PYTHON_CMD, msprof_py_path, "export", "all", "--help"],
            capture_output=True, shell=False, timeout=MSPROF_PY_PROBE_TIMEOUT
        )
    except Exception:
        return False
    return completed_process.returncode == 0


class CannPackageManager:
    SUPPORT_DEFAULT_EXPORT_DB = None
    SUPPORT_EXPORT_ALL = None

    @classmethod
    def is_support_default_export_db(cls) -> bool:
        if cls.SUPPORT_DEFAULT_EXPORT_DB is None:
            cls.SUPPORT_DEFAULT_EXPORT_DB = check_cann_package_support_default_export_db()
        return cls.SUPPORT_DEFAULT_EXPORT_DB

    @classmethod
    def is_support_export_all(cls) -> bool:
        if cls.SUPPORT_EXPORT_ALL is None:
            bin_support = check_cann_package_support_export_all()
            cls.SUPPORT_EXPORT_ALL = bin_support and check_analysis_package_support_export_all(
                find_msprof_py_path()
            )
        return cls.SUPPORT_EXPORT_ALL
