import hashlib
import os
import shutil
from pathlib import Path

import pytest

_cov = None
_nodeid = ""
_tmp_data = ""
_session_seq = 0


def pytest_configure(config):
    """Start coverage from .coveragerc at session start."""
    global _cov, _nodeid, _tmp_data, _session_seq
    _nodeid = ""

    rcfile = os.environ.get("COVERAGE_PLUGIN_RCFILE")
    if not rcfile:
        return

    os.environ["COVERAGE_CORE"] = "ctrace"

    import coverage

    _session_seq += 1
    case_dir = os.environ.get("COVERAGE_CASE_DIR", "") or "."
    # Each session uses an isolated temp directory shared by the parent process and
    # spawned children; it is merged and removed at session end to avoid interference
    # between concurrent sessions.
    _tmp_data = str(Path(case_dir) / f".cov-pending-{os.getpid()}-{_session_seq}" / "data")
    Path(_tmp_data).parent.mkdir(parents=True, exist_ok=True)

    # Child processes spawned by the code under test do not load this plugin. These
    # two environment variables make them start coverage automatically and write to
    # the same temp directory (parallel mode appends a process suffix).
    os.environ["COVERAGE_FILE"] = _tmp_data
    os.environ["COVERAGE_PROCESS_START"] = rcfile

    # Default context is import: collection, fixture/module imports, and package
    # initialization are recorded as import. Before each case we switch to
    # test:<nodeid>, and switch back after teardown, so the data itself distinguishes
    # import-time execution from test-body execution.
    _cov = coverage.Coverage(config_file=rcfile, data_file=_tmp_data, context="import")
    _cov.start()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """Switch the coverage context to the current case before setup."""
    if _cov is not None:
        _cov.switch_context("test:" + item.nodeid)
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    """Switch back to import after teardown so later collection/cleanup is not counted in the test body."""
    yield
    if _cov is not None:
        _cov.switch_context("import")


def pytest_collection_modifyitems(session, config, items):
    """Record the nodeid of this session after collection."""
    global _nodeid
    if not _nodeid and items:
        _nodeid = items[0].nodeid


def _merge_child_data(case_data, tmp_dir: Path) -> None:
    """Merge parallel coverage data written by child processes into the case data and clean up the temp directory."""
    import coverage

    for child_file in sorted(tmp_dir.glob("*")):
        if not child_file.is_file():
            continue
        child_data = coverage.CoverageData(basename=str(child_file))
        child_data.read()
        case_data.update(child_data)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def pytest_sessionfinish(session, exitstatus):
    """Write this case's coverage data to a standalone file at session end."""
    global _cov, _nodeid, _tmp_data
    if _cov is None:
        return

    case_dir = os.environ.get("COVERAGE_CASE_DIR", "")
    items = list(getattr(session, "items", ()) or ())
    nodeid = _nodeid or (items[0].nodeid if items else "unknown")
    # Keep in sync with the naming convention in coverage_report.py:
    #   :: -> _, / -> _, \ -> _, truncate to 150 + short nodeid hash
    #   (avoids collisions between long nodeids that share the same prefix).
    safe = nodeid.replace("::", "_").replace("/", "_").replace("\\", "_")[:150]
    digest = hashlib.sha1(nodeid.encode("utf-8")).hexdigest()[:8]

    # Take the instance out and reset globals first to avoid stale state on exceptions.
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
