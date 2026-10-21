import hashlib
import os
import shutil
from pathlib import Path

_cov = None
_nodeid = ""
_tmp_data = ""
_session_seq = 0


def pytest_configure(config):
    """session 开始时按 .coveragerc 启动覆盖率采集。"""
    global _cov, _nodeid, _tmp_data, _session_seq
    _nodeid = ""

    rcfile = os.environ.get("COVERAGE_PLUGIN_RCFILE")
    if not rcfile:
        return

    import coverage

    _session_seq += 1
    case_dir = os.environ.get("COVERAGE_CASE_DIR", "") or "."
    # 每个 session 使用独立临时目录, 父进程与 spawn 出的子进程都写在其中,
    # session 结束时整体归并并删除目录 (避免并发会话之间互相干扰)
    _tmp_data = str(Path(case_dir) / f".cov-pending-{os.getpid()}-{_session_seq}" / "data")
    Path(_tmp_data).parent.mkdir(parents=True, exist_ok=True)

    # 被测代码 spawn 出的子进程不经过本插件, 通过以下两个环境变量让子进程
    # 在启动时自动开始采集, 并把数据写到同一临时目录 (parallel 模式加进程后缀)
    os.environ["COVERAGE_FILE"] = _tmp_data
    os.environ["COVERAGE_PROCESS_START"] = rcfile

    _cov = coverage.Coverage(config_file=rcfile, data_file=_tmp_data)
    _cov.start()


def pytest_collection_modifyitems(session, config, items):
    """收集结束后记录本 session 对应的用例 nodeid。"""
    global _nodeid
    if not _nodeid and items:
        _nodeid = items[0].nodeid


def _merge_child_data(case_data, tmp_dir: Path) -> None:
    """把子进程写出的并行覆盖率数据归并进当前用例数据, 并清理临时目录。"""
    import coverage

    for child_file in sorted(tmp_dir.glob("*")):
        if not child_file.is_file():
            continue
        child_data = coverage.CoverageData(basename=str(child_file))
        child_data.read()
        case_data.update(child_data)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def pytest_sessionfinish(session, exitstatus):
    """session 结束时把本用例的覆盖数据写出为独立文件。"""
    global _cov, _nodeid, _tmp_data
    if _cov is None:
        return

    case_dir = os.environ.get("COVERAGE_CASE_DIR", "")
    items = list(getattr(session, "items", ()) or ())
    nodeid = _nodeid or (items[0].nodeid if items else "unknown")
    # 与 coverage_report.py 的命名约定保持一致:
    #   :: -> _ , / -> _ , \ -> _ , 截断 150 + nodeid 短哈希 (避免长 nodeid 撞名)
    safe = nodeid.replace("::", "_").replace("/", "_").replace("\\", "_")[:150]
    digest = hashlib.sha1(nodeid.encode("utf-8")).hexdigest()[:8]

    # 先把实例取出并复位全局, 避免异常时状态残留
    _cov.stop()
    case = _cov
    tmp_data = _tmp_data
    _cov = None
    _nodeid = ""
    _tmp_data = ""

    if not case_dir:
        return

    import coverage

    case_data = coverage.CoverageData(basename=str(Path(case_dir) / f"{safe}-{digest}.coverage"))
    case_data.update(case.get_data())
    _merge_child_data(case_data, Path(tmp_data).parent)
    case_data.write()
