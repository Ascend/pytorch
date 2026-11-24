import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Fixed group name for coverage data that cannot be attributed to a test case.
UNMAPPED = "_unmapped"

# Files ignored when snapshotting the source/test tree.
_SNAPSHOT_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def _load_json(path: Path):
    """Load a JSON file, returning None on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _collect_cases(report_dir: Path):
    """Collect cases from all shard_*_cases.json files under report_dir."""
    cases = []
    for results_file in sorted(report_dir.glob("shard_*_cases.json")):
        data = _load_json(results_file) or {}
        cases.extend(data.get("cases", []))
    return cases


def _case_data_name(nodeid: str) -> str:
    """Convert a nodeid to the raw coverage data file name (matches the pytest plugin).

    The nodeid is escaped and truncated, then suffixed with a short hash, so long
    parameterized cases that share the same truncated prefix do not overwrite each
    other's data files.
    """
    safe = nodeid.replace("::", "_").replace("/", "_").replace("\\", "_")[:150]
    digest = hashlib.sha1(nodeid.encode("utf-8")).hexdigest()[:8]
    return f"{safe}-{digest}.coverage"


def _find_case_data(raw_dir: Path, nodeid: str):
    """Locate the raw coverage data file for a nodeid, accepting paths with or without the test/ prefix."""
    alternatives = [nodeid]
    alternatives.append(nodeid[5:] if nodeid.startswith("test/") else f"test/{nodeid}")
    for candidate in alternatives:
        path = raw_dir / _case_data_name(candidate)
        if path.is_file():
            return path
    return None


def _write_failed_cases(report_dir: Path, out_file: Path) -> None:
    """Write per-shard case execution stats (total/passed/failed/skipped) to failed_cases.json."""
    status_counts = {}
    failed = []
    total = 0
    for results_file in sorted(report_dir.glob("shard_*_cases.json")):
        data = _load_json(results_file) or {}
        for case in data.get("cases", []):
            total += 1
            status = case.get("status", "") or "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1
            if status in ("failed", "error", "timeout"):
                failed.append({
                    "nodeid": case.get("nodeid", ""),
                    "file": case.get("file", ""),
                    "status": status,
                })
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(
            {
                "total_cases": total,
                "total_passed": status_counts.get("passed", 0),
                "total_failed": len(failed),
                "total_errors": status_counts.get("error", 0),
                "total_timeout": status_counts.get("timeout", 0),
                "total_skipped": status_counts.get("skipped", 0),
                "status_counts": status_counts,
                "failed_cases": failed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"  用例统计: 共 {total}, 通过 {status_counts.get('passed', 0)}, "
        f"失败 {len(failed)}, 跳过 {status_counts.get('skipped', 0)} -> {out_file}"
    )


def _write_file_logs(report_dir: Path, out_dir: Path) -> None:
    """Aggregate case logs by test file into out_dir/<test file>.log."""
    cases = _collect_cases(report_dir)
    logs_dir = report_dir / "cases_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Case log naming: {prefix}-{shard}_{case_idx}_{safe}.log
    # Build a case_idx -> log path index first to avoid scanning all logs per case (O(N*M)).
    log_index = {}
    if logs_dir.is_dir():
        for log_file in logs_dir.glob("*.log"):
            parts = log_file.name.split("_")
            if len(parts) >= 3:
                log_index.setdefault(parts[1], log_file)

    groups = {}
    for case in cases:
        groups.setdefault(case.get("file") or "unknown", []).append(case)

    for file_path, file_cases in sorted(groups.items()):
        log_path = out_dir / f"{file_path}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write(f"=== {file_path}: {len(file_cases)} case(s) ===\n")
            for case in file_cases:
                nodeid = case.get("nodeid", "")
                case_idx = case.get("case_idx", "")
                fh.write(f"\n----- {case.get('status', '?')}  {nodeid} -----\n")
                log_file = log_index.get(str(case_idx)) if case_idx != "" else None
                if log_file is not None:
                    fh.write(log_file.read_text(encoding="utf-8", errors="replace"))
                else:
                    fh.write("(未找到用例日志)\n")
    print(f"  文件日志: {len(groups)} 个 -> {out_dir}")


def _group_case_covdata(report_dir: Path, raw_dir: Path, shard_dir: Path) -> None:
    """Group per-case raw coverage data by test file into shard_dir/<file>/covdata/.

    Each group is written twice:
    - ``shard_dir/<file>/covdata/coverage``: test-body lines only, consumed directly by selection;
    - ``shard_dir/_full/<file>/covdata/coverage``: full data, used for coverage reports.
    """
    import coverage

    cases = _collect_cases(report_dir)

    # Group by the test file path recorded when the case was collected.
    groups = {}
    claimed = set()
    for case in cases:
        group = case.get("file") or "unknown"
        data_file = _find_case_data(raw_dir, case.get("nodeid", ""))
        if data_file is None:
            continue
        groups.setdefault(group, []).append(data_file)
        claimed.add(data_file)

    # Raw data that cannot be attributed goes to _unmapped.
    unmapped = 0
    for data_file in sorted(raw_dir.glob("*.coverage")):
        if data_file in claimed:
            continue
        groups.setdefault(UNMAPPED, []).append(data_file)
        unmapped += 1

    written = 0
    missing_context = []
    for group, files in sorted(groups.items()):
        # Full data (including import-time lines) is written to _full/<group>/ for coverage reports.
        full_dir = shard_dir / "_full" / group / "covdata"
        full_dir.mkdir(parents=True, exist_ok=True)
        full_file = full_dir / "coverage"
        group_data = coverage.CoverageData(basename=str(full_file))
        for path in files:
            src_data = coverage.CoverageData(basename=str(path))
            src_data.read()
            group_data.update(src_data)
        group_data.write()

        # The published copy keeps only test-body lines: consumers (e.g. precision-test)
        # can use it directly without distinguishing import-time from test-body execution.
        clean_dir = shard_dir / group / "covdata"
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_file = clean_dir / "coverage"
        shutil.copy2(full_file, clean_file)
        contextual, _ = _strip_import_from_db(clean_file)
        if not contextual:
            missing_context.append(group)
        written += len(files)

    if missing_context:
        print(
            f"  警告: {len(missing_context)} 个测试分组的覆盖率数据没有 "
            f"import/test context，无法剔除导入期覆盖: {missing_context[:5]}",
            file=sys.stderr,
        )

    executed_files = {case.get("file") or "unknown" for case in cases if case.get("status") == "passed"}
    missing_data = sorted(executed_files - set(groups))
    if missing_data:
        print(
            f"  警告: {len(missing_data)} 个测试文件有用例通过但未产生覆盖数据: {missing_data}",
            file=sys.stderr,
        )
    if unmapped:
        print(f"  警告: {unmapped} 个覆盖数据文件未匹配到任何已报告用例", file=sys.stderr)
    print(f"  归集 {written} 个用例覆盖数据到 {len(groups)} 个测试文件分组")


def cmd_group(args) -> int:
    """Entry point for the group subcommand."""
    report_dir = Path(args.report_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    shard_dir = Path(args.shard_dir).resolve()

    if not report_dir.is_dir():
        print(f"错误: 报告目录不存在: {report_dir}", file=sys.stderr)
        return 1
    shard_dir.mkdir(parents=True, exist_ok=True)

    _write_failed_cases(report_dir, shard_dir / "logs" / "failed_cases.json")
    _write_file_logs(report_dir, shard_dir / "logs")
    if raw_dir.is_dir():
        _group_case_covdata(report_dir, raw_dir, shard_dir)
    else:
        print(f"  警告: 原始覆盖率目录不存在, 跳过归集: {raw_dir}", file=sys.stderr)
    return 0


def _merge_shard_logs(staging: Path, shards) -> None:
    """Merge shard logs into staging/logs, appending files with the same name."""
    for src in shards:
        logs_root = src / "logs"
        if not logs_root.is_dir():
            continue
        for log_file in sorted(logs_root.rglob("*.log")):
            rel = log_file.relative_to(logs_root)
            target = staging / "logs" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with target.open("a", encoding="utf-8") as fh:
                    fh.write(log_file.read_text(encoding="utf-8", errors="replace"))
            else:
                shutil.copy2(log_file, target)


def _merge_covdata(staging: Path, shards) -> dict:
    """Merge shard covdata: generate staging/.coverage and logs/failed_cases.json.

    Returns merge statistics for the manifest and summary. Returns None if no data
    can be attributed to any test file, which is treated as a failure so the
    previous result is kept.
    """
    import coverage

    groups = {}
    full_groups = {}
    all_failed = []
    status_counts = {}
    sources = []
    failed_files = set()
    total = 0

    for shard in shards:
        if not shard.is_dir():
            continue
        sources.append(shard.name)

        for group_dir in sorted(shard.rglob("covdata")):
            if not group_dir.is_dir():
                continue
            group = group_dir.parent.relative_to(shard).as_posix()
            data_file = group_dir / "coverage"
            if not data_file.is_file():
                continue
            if group.startswith("_full/"):
                # Full data saved by the collector (including import-time lines), used only for coverage reports.
                full_groups.setdefault(group[len("_full/"):], []).append(data_file)
            else:
                # Published path: the collector has already filtered it to test-body lines.
                groups.setdefault(group, []).append(data_file)

        shard_file = shard / "logs" / "failed_cases.json"
        data = _load_json(shard_file)
        if data:
            total += data.get("total_cases", 0)
            for key, value in (data.get("status_counts") or {}).items():
                status_counts[key] = status_counts.get(key, 0) + value
            for case in data.get("failed_cases", []):
                all_failed.append(case)
                if case.get("file"):
                    failed_files.add(case["file"])

    if not groups:
        print("错误: 所有分片均未包含可用的覆盖率数据", file=sys.stderr)
        return None

    mapped_groups = [g for g in groups if not g.startswith(UNMAPPED)]
    unmapped_files = sum(len(files) for g, files in groups.items() if g.startswith(UNMAPPED))
    if not mapped_groups:
        print(
            f"错误: 存在 {len(groups)} 个数据分组但没有任何一个能映射到已报告测试文件",
            file=sys.stderr,
        )
        return None

    combined = coverage.CoverageData(basename=str(staging / ".coverage"))
    written = 0
    for group in sorted(set(groups) | set(full_groups)):
        # Report scope: union of full and clean data (clean is a subset, so duplicates
        # are harmless); clean data still works when an older shard has no _full data.
        for path in full_groups.get(group, []) + groups.get(group, []):
            src_data = coverage.CoverageData(basename=str(path))
            src_data.read()
            combined.update(src_data)

        # Publish scope: write only clean group data; unmapped data goes only into overall coverage.
        if group.startswith(UNMAPPED) or group not in groups:
            continue
        group_dir = staging / group / "covdata"
        group_dir.mkdir(parents=True, exist_ok=True)
        group_data = coverage.CoverageData(basename=str(group_dir / "coverage"))
        for path in groups[group]:
            src_data = coverage.CoverageData(basename=str(path))
            src_data.read()
            group_data.update(src_data)
        group_data.write()
        written += len(groups[group])

    combined.write()
    if unmapped_files:
        print(f"  警告: {unmapped_files} 个未归属数据文件仅并入整体覆盖率, 未写入产物目录", file=sys.stderr)

    out = staging / "logs" / "failed_cases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "total_cases": total,
                "total_failed": len(all_failed),
                "shards": sources,
                "failed_cases": all_failed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Stats after merging (fields aligned with ci-test coverage artifacts).
    stats = {
        "total_cases": total,
        "total_passed": status_counts.get("passed", 0),
        "total_failed": len(all_failed),
        "total_errors": status_counts.get("error", 0),
        "total_timeout": status_counts.get("timeout", 0),
        "total_skipped": status_counts.get("skipped", 0),
        "status_counts": status_counts,
        "shards": sources,
        "total_files": len(set(mapped_groups) | failed_files),
        "total_covered_files": len(mapped_groups),
        "total_failed_files": len(failed_files),
        "unmapped_files": unmapped_files,
    }
    print(
        f"  合并用例统计: 文件 {stats['total_files']}, 用例 {total} "
        f"(通过 {stats['total_passed']}, 失败 {stats['total_failed']}, 跳过 {stats['total_skipped']}) -> {out}"
    )
    print(f"  逐文件覆盖率: {written} 个数据文件归并为 {len(mapped_groups)} 个测试文件分组")
    return stats


def _strip_import_from_db(db_path: Path) -> tuple[bool, int]:
    """Remove import-only contexts from a coverage database in place.

    Returns (has_context_data, removed_rows). has_context_data is False for data
    collected without contexts, in which case nothing is removed.
    """
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "context" not in tables:
            return False, 0

        contexts = {
            cid: (ctx or "")
            for cid, ctx in conn.execute("SELECT id, context FROM context")
        }
        trace_like = any("test:" in ctx for ctx in contexts.values()) or any(
            ctx == "import" or ctx.startswith("import|") for ctx in contexts.values()
        )
        if not trace_like:
            return False, 0

        drop = [
            cid
            for cid, ctx in contexts.items()
            if ctx
            and "test:" not in ctx
            and (ctx == "import" or ctx.startswith("import|"))
        ]
        placeholders = ",".join("?" for _ in drop)
        removed = 0
        with conn:
            if drop:
                if "arc" in tables:
                    removed += conn.execute(
                        f"DELETE FROM arc WHERE context_id IN ({placeholders})", drop
                    ).rowcount
                if "line_bits" in tables:
                    removed += conn.execute(
                        f"DELETE FROM line_bits WHERE context_id IN ({placeholders})", drop
                    ).rowcount
        return True, removed
    except sqlite3.Error as exc:
        print(f"  警告: 过滤覆盖率库失败 {db_path}: {exc}", file=sys.stderr)
        return False, 0
    finally:
        conn.close()


def _build_coverage_xml(staging: Path, source: str) -> bool:
    """Generate staging/coverage.xml from staging/.coverage (source package only).

    Note: coverage needs the source under test to build the report. If the merge
    environment does not have the package installed (source cannot be parsed), this
    step fails; the coverage data itself is still usable, so the failure is non-fatal.
    """
    include_patterns = f"*/{source}/*"
    env = dict(os.environ)
    env["COVERAGE_FILE"] = str(staging / ".coverage")
    cmd = [
        sys.executable, "-m", "coverage", "xml",
        f"--include={include_patterns}",
        "-o", str(staging / "coverage.xml"),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"警告: 未生成 coverage.xml (合并环境无法解析被测源码时属正常): "
            f"{result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'unknown'}",
            file=sys.stderr,
        )
        return False

    report = subprocess.run(
        [sys.executable, "-m", "coverage", "report", f"--include={include_patterns}"],
        env=env, capture_output=True, text=True,
    )
    tail = "\n".join(report.stdout.strip().splitlines()[-5:])
    print(f"=== 合并覆盖率 ({source}) ===\n{tail}")
    return True


def _write_manifest(merged: Path, source: str, workflow_name: str, repository: str,
                    run_id: str, run_attempt: str, event: str, sha: str,
                    test_groups: int, total_cases: int, total_failed: int) -> dict:
    """Write manifest.json and return the manifest dict (fields aligned with the ci-test coverage pipeline)."""
    manifest = {
        "workflow": workflow_name,
        "repository": repository,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "event": event,
        f"{source}_sha": sha,
        f"{source}_short": sha[:10] if sha else "",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_groups": test_groups,
        "total_cases": total_cases,
        "total_failed": total_failed,
    }
    (merged / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _write_summary(manifest: dict, stats: dict, merged: Path, source: str, summary_file) -> None:
    """Build a markdown summary (files/cases/passed/failed/etc.), print it, and optionally write it to a file."""
    import xml.etree.ElementTree as ET

    total_cases = stats.get("total_cases", 0)
    total_passed = stats.get("total_passed", 0)
    total_failed = stats.get("total_failed", 0)
    pass_rate = (total_passed / total_cases * 100) if total_cases else 0.0
    pure_failed = total_failed - stats.get("total_errors", 0) - stats.get("total_timeout", 0)

    lines = [
        f"## {source} 用例覆盖率 ({manifest.get('generated_at', '')})",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 统计包 | `{source}` |",
        f"| 执行提交 | `{manifest.get(f'{source}_sha', '') or 'n/a'}` |",
        f"| 测试文件 | {stats.get('total_files', 0)} 个执行 "
        f"(其中有覆盖数据 {stats.get('total_covered_files', 0)} 个, "
        f"有失败 {stats.get('total_failed_files', 0)} 个) |",
        f"| 用例总数 | {total_cases} |",
        f"| 成功 | {total_passed} |",
        f"| 失败 | {total_failed} "
        f"(failed {pure_failed}, "
        f"error {stats.get('total_errors', 0)}, timeout {stats.get('total_timeout', 0)}) |",
        f"| 跳过 | {stats.get('total_skipped', 0)} |",
        f"| 用例通过率 | {pass_rate:.2f}% |",
    ]
    xml_file = merged / "coverage.xml"
    if not xml_file.is_file():
        lines.append("| 行覆盖率 | 未生成 (合并环境缺少被测源码, 仅产出 .coverage 数据) |")
    else:
        try:
            root = ET.parse(xml_file).getroot()
            valid = int(root.get("lines-valid", 0))
            covered = int(root.get("lines-covered", 0))
            rate = float(root.get("line-rate", 0)) * 100
            lines.append(f"| 行覆盖率 | {rate:.2f}% ({covered}/{valid}) |")
        except Exception as exc:  # noqa: BLE001 - summary failures must not break the main flow
            lines.append(f"| 行覆盖率 | 解析失败 ({exc}) |")

    text = "\n".join(lines) + "\n"
    print(text)
    if summary_file:
        Path(summary_file).write_text(text, encoding="utf-8")


def cmd_merge(args) -> int:
    """Entry point for the merge subcommand."""
    shards = [Path(p).resolve() for p in args.inputs.split(",") if p.strip()]
    out_dir = Path(args.out_dir).resolve()
    source = args.source
    out_name = source

    staging = out_dir / f"{out_name}@staging"
    merged = out_dir / f"{out_name}@latest"

    if not shards:
        print("错误: --inputs 未提供任何分片目录", file=sys.stderr)
        return 1

    shutil.rmtree(staging, ignore_errors=True)
    (staging / "logs").mkdir(parents=True, exist_ok=True)

    _merge_shard_logs(staging, shards)
    stats = _merge_covdata(staging, shards)
    if stats is None:
        # Data exists but none of it can be mapped: treat as failure and keep the previous result.
        print("错误: 无可提升的覆盖率数据, 保留上次结果", file=sys.stderr)
        return 3

    # coverage.xml needs the source under test; the merge environment usually does not
    # have the package installed, so a failure only warns and does not block other artifacts.
    _build_coverage_xml(staging, source)

    shutil.rmtree(merged, ignore_errors=True)
    staging.rename(merged)
    print(f"=== 已提升: {merged} ===")

    manifest = _write_manifest(
        merged, source, args.workflow_name,
        args.repository, args.run_id, args.run_attempt, args.event, args.sha,
        stats["total_covered_files"], stats["total_cases"], stats["total_failed"],
    )
    _write_summary(manifest, stats, merged, source, args.summary_file)
    return 0


def _collect_executed_files(executed_dir: Path) -> set:
    """Collect test files actually executed from the merged artifacts (paths relative to the test directory)."""
    files = set()
    if not executed_dir.is_dir():
        return files
    for covdata in sorted(executed_dir.rglob("covdata")):
        if not covdata.is_dir():
            continue
        group = covdata.parent.relative_to(executed_dir).as_posix()
        if group.startswith(UNMAPPED):
            continue
        files.add(group[5:] if group.startswith("test/") else group)
    return files


def _trim_snapshot(snapshot: Path, keep: set, label: str) -> None:
    """Trim the test tree snapshot to files that were actually executed."""
    if not keep:
        print(f"警告: 未找到{label}执行记录, 保留完整快照", file=sys.stderr)
        return
    removed = 0
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        if path.relative_to(snapshot).as_posix() in keep:
            continue
        path.unlink()
        removed += 1
    for path in sorted((p for p in snapshot.rglob("*") if p.is_dir()), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass
    print(f"{label}快照: 保留 {len(keep)} 个执行过的测试文件, 移除 {removed} 个")


def cmd_covstub(args) -> int:
    """Entry point for the covstub subcommand: package source snapshots into covstub/{<pkg>, test}."""
    src = Path(args.src).resolve()
    test = Path(args.test).resolve()
    out_dir = Path(args.out_dir).resolve()
    covstub = out_dir / "covstub"

    shutil.rmtree(covstub, ignore_errors=True)
    covstub.mkdir(parents=True, exist_ok=True)

    # 1) Target package source snapshot.
    if src.is_dir():
        shutil.copytree(src, covstub / src.name, ignore=_SNAPSHOT_IGNORE)
        print(f"源码快照: {src} -> {covstub / src.name}")
    else:
        print(f"警告: 源码目录不存在, 跳过: {src}", file=sys.stderr)

    # 1.1) fxrt source snapshot: after installation, fxrt/fxrt/python/fxrt lives under
    #      <pkg>/fxrt, outside the <pkg> source tree. Merge it by install layout,
    #      otherwise fxrt coverage cannot be mapped back to source.
    fxrt_src = src.parent / "fxrt" / "fxrt" / "python" / "fxrt"
    fxrt_dst = covstub / src.name / "fxrt"
    if fxrt_src.is_dir():
        shutil.copytree(fxrt_src, fxrt_dst, ignore=_SNAPSHOT_IGNORE)
        print(f"fxrt 源码快照: {fxrt_src} -> {fxrt_dst}")
    else:
        print(f"警告: fxrt 源码目录不存在, 跳过: {fxrt_src}", file=sys.stderr)

    # 2) Test tree snapshot, trimmed to test files that were actually executed.
    test_snapshot = covstub / test.name
    if not test.is_dir():
        print(f"警告: 测试目录不存在, 跳过: {test}", file=sys.stderr)
        return 0
    shutil.copytree(test, test_snapshot, ignore=_SNAPSHOT_IGNORE)

    keep = _collect_executed_files(Path(args.executed_dir).resolve()) if args.executed_dir else set()
    _trim_snapshot(test_snapshot, keep, "仓内测试")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="TorchNPU 用例覆盖率数据归集与合并",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_group = sub.add_parser(
        "group",
        help="把单分片 per-case 覆盖率数据按测试文件归集（测试体口径 + 完整口径）",
    )
    p_group.add_argument("--report-dir", required=True, help="测试报告目录 (含 shard_*_cases.json)")
    p_group.add_argument("--raw-dir", required=True, help="per-case 原始覆盖率数据目录")
    p_group.add_argument("--shard-dir", required=True, help="归集输出目录 (作为分片数据上传)")
    p_group.set_defaults(func=cmd_group)

    p_merge = sub.add_parser("merge", help="归并所有分片, 生成 coverage.xml 与 manifest.json")
    p_merge.add_argument("--inputs", required=True, help="逗号分隔的分片数据目录列表")
    p_merge.add_argument("--out-dir", required=True, help="合并输出根目录")
    p_merge.add_argument("--source", default="torch_npu", help="覆盖率统计的目标包 (默认 torch_npu)")
    p_merge.add_argument("--workflow-name", default="torch_npu_coverage_nightly", help="写入 manifest 的流水线名")
    p_merge.add_argument("--summary-file", default="", help="可选: 输出 markdown 摘要的文件路径")
    p_merge.add_argument("--repository", default="", help="写入 manifest 的仓库")
    p_merge.add_argument("--run-id", default="", help="写入 manifest 的运行 ID")
    p_merge.add_argument("--run-attempt", default="1", help="写入 manifest 的运行重试次数")
    p_merge.add_argument("--event", default="", help="写入 manifest 的触发事件")
    p_merge.add_argument("--sha", default="", help="写入 manifest 的统计提交 SHA")
    p_merge.set_defaults(func=cmd_merge)

    p_covstub = sub.add_parser("covstub", help="打包源码快照 covstub/{<pkg>, test} (对齐 ci-test)")
    p_covstub.add_argument("--src", required=True, help="目标包源码目录 (如 torch_npu)")
    p_covstub.add_argument("--test", required=True, help="测试树目录 (如 test)")
    p_covstub.add_argument("--out-dir", required=True, help="输出根目录, covstub 建在其中")
    p_covstub.add_argument("--executed-dir", default="", help="已合并产物目录, 用于把测试树裁剪为执行过的文件")
    p_covstub.set_defaults(func=cmd_covstub)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
