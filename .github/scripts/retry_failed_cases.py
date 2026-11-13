#!/usr/bin/env python3
"""
Retry failed/error/timeout cases from a finished shard run, in isolated
rounds whose concurrency is configurable per round.

Runs AFTER the shard runner (run_npu_test_shard.py or
run_npu_test_shard_batch.py) has written
test-reports/shard_<prefix>-<n>_cases.json and BEFORE the workflow parses
those results, so flaky/infra-transient failures get second chances:

    - Only cases whose recorded status is failed / error / timeout are
      retried; none -> print a skip note and exit 0.
    - Execution reuses run_tests_with_tasks_concurrent, keeping the original
      case_idx so per-case junit XML / case log filenames in the retry
      report dir match the main-run filenames. The worker count comes from
      --max-workers, a comma-separated per-round plan (e.g. '8,1'); a
      single '1' (the default) keeps the historical strictly serial
      behavior (one worker subprocess at a time, one pytest.main() per
      case).
    - Each case gets up to --max-attempts isolated rounds (default
      1 = the historical single retry). A case that passes ANY round is
      recovered and its later rounds are skipped; only a case that fails
      ALL of its rounds keeps a failed/error/timeout status. Rounds after
      the first only re-run the still-failing residue, so extra rounds cost
      seconds unless a case genuinely hangs (bounded by --timeout per
      round). With --max-attempts > 1 each round writes its evidence to
      retry/attempt_<n>/ so later rounds cannot overwrite earlier
      round's junit XML / case logs; with --max-attempts 1 the layout is
      the historical flat retry/ directory.
    - Retry-passed (or newly skipped) cases: the final passing round's
      junit XML and case log are copied back over the main-run files
      (refreshed) and the cases.json entry is updated to the new status
      plus retry_attempts (how many rounds the case consumed).
    - Still failed/error/timeout cases: the main-run files stay untouched;
      the retry evidence remains in the retry report dir (uploaded with
      the test-reports artifact) and cases.json records the retry outcome
      plus retried=true / first_attempt_status / retry_attempts markers.
    - The shard cases.json / stats.json summary counts are recomputed after
      the merge so every downstream consumer (parse step, report generator,
      upstream dashboard) sees post-retry numbers.
    - The script ALWAYS exits 0: case failures are recorded in cases.json,
      the shard's pass/fail verdict is decided by the later workflow steps.

Usage:
    python retry_failed_cases.py \
        --cases-file test-reports/shard_core-1_cases.json \
        --test-dir pytorch/test \
        --report-dir test-reports \
        --retry-report-dir test-reports/retry \
        --timeout 1200 \
        --device-env privateuse1 \
        --max-retry-cases 0 \
        --max-attempts 3 \
        --max-workers 8,1 \
        --verbose
"""

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import run_npu_test_shard as runner
import parse_test_results


# Statuses that qualify for a retry, and the statuses that count as "still
# not passed" after one (mirrors the workflow's failed+errors+timeout
# total_failed bucket in the parse step).
RETRY_STATUSES = ("failed", "error", "timeout")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Serially retry failed/error/timeout cases from a finished shard run"
    )
    parser.add_argument("--cases-file", type=str, required=True,
                        help="Path to the shard cases.json produced by the shard runner")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Path to the PyTorch test directory")
    parser.add_argument("--report-dir", type=str, default="test-reports",
                        help="Main report directory holding the cases.json (default: test-reports)")
    parser.add_argument("--retry-report-dir", type=str, default="test-reports/retry",
                        help="Report directory for the retry attempt (default: test-reports/retry)")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Per-case timeout in seconds, same as the main run (default: 1200)")
    parser.add_argument("--device-env", type=str, default="privateuse1",
                        help="PYTORCH_TESTING_DEVICE_ONLY_FOR value; must match the main run")
    parser.add_argument("--max-retry-cases", type=int, default=100,
                        help="Max cases to retry serially, 0 = unlimited (default: 100)")
    parser.add_argument("--max-attempts", type=int, default=1,
                        help="Max isolated rounds per case, 1 = single "
                             "retry (historical behavior). A case that passes "
                             "any round is recovered; only a case failing all "
                             "rounds keeps its failed status (default: 1)")
    parser.add_argument("--max-workers", type=str, default="1",
                        help="Comma-separated worker count per retry round, "
                             "e.g. '8,1' (round 1 with 8 workers, round 2 "
                             "serially); shorter lists reuse the last value, "
                             "longer lists ignore the extra values. '1' = "
                             "strictly serial, the historical behavior "
                             "(default: 1)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.max_retry_cases < 0:
        parser.error("--max-retry-cases must be >= 0 (0 = unlimited)")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be >= 1")
    # Expand the per-round worker plan here so a malformed value fails fast
    # (parse_worker_plan is defined below; it resolves at call time).
    args.worker_plan = parse_worker_plan(args.max_workers, args.max_attempts, parser)
    return args


def parse_worker_plan(raw: str, attempts: int, parser) -> List[int]:
    """Expand a comma-separated --max-workers value into one worker count per round.

    '8,1' with 3 rounds -> [8, 1, 1]: a list shorter than the round count
    reuses its last value, a longer list has its extra values ignored.
    """
    try:
        values = [int(part) for part in str(raw).split(",") if part.strip()]
    except ValueError:
        parser.error(f"--max-workers must be comma-separated ints, got {raw!r}")
    if not values or any(value < 1 for value in values):
        parser.error(f"--max-workers entries must be >= 1, got {raw!r}")
    return [
        values[index] if index < len(values) else values[-1]
        for index in range(attempts)
    ]


def write_github_outputs(pairs) -> None:
    """Append key=value pairs to $GITHUB_OUTPUT when running in Actions."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if not output_file:
        return
    with open(output_file, "a", encoding="utf-8") as f:
        for key, value in pairs:
            f.write(f"{key}={value}\n")


def count_statuses(cases_list: List[Dict]) -> Dict[str, int]:
    """Count cases.json-style summary fields over a cases list."""
    counts = {"passed": 0, "failed": 0, "errors": 0, "timeout": 0, "skipped": 0}
    for case in cases_list:
        status = case.get("status")
        if status == "passed":
            counts["passed"] += 1
        elif status == "failed":
            counts["failed"] += 1
        elif status == "timeout":
            counts["timeout"] += 1
        elif status == "skipped":
            counts["skipped"] += 1
        else:
            # "error" and anything unrecognized land in the error bucket,
            # matching ConcurrentResultAggregator.add_case_result.
            counts["errors"] += 1
    return counts


def merge_retry_results(
    cases_data: Dict,
    retry_results: List[Dict],
    retry_duration: Optional[float] = None,
    max_retry_cases: Optional[int] = None,
    excluded_by_cap: int = 0,
) -> Dict:
    """
    Merge serial retry outcomes into the shard cases.json payload.

    Pure function: returns a new cases_data dict, never mutates the input.
    Each retried entry keeps its nodeid / file / case_idx and takes the
    retry attempt's status / duration / returncode / message / command plus
    retried=true and first_attempt_status markers. Summary counts are
    recomputed and a "retry" summary block is attached.
    """
    retry_by_nodeid = {r.get("nodeid"): r for r in retry_results if r.get("nodeid")}

    merged_cases = []
    retried = 0
    recovered = 0
    still_not_passed = 0
    for case in cases_data.get("cases", []):
        retry = retry_by_nodeid.get(case.get("nodeid"))
        if retry is None:
            # Not selected for retry, or the retry machinery produced no
            # result for it: keep the original entry untouched.
            merged_cases.append(case)
            continue
        retried += 1
        new_case = dict(case)
        new_case["status"] = retry.get("status", case.get("status"))
        new_case["duration"] = retry.get("duration", case.get("duration", 0.0))
        new_case["returncode"] = retry.get("returncode", case.get("returncode", 1))
        new_case["message"] = retry.get("message", case.get("message", ""))
        new_case["command"] = retry.get("command", case.get("command", ""))
        new_case["retried"] = True
        new_case["first_attempt_status"] = case.get("status")
        if new_case["status"] == "passed":
            recovered += 1
        elif new_case["status"] in RETRY_STATUSES:
            still_not_passed += 1
        merged_cases.append(new_case)

    merged = dict(cases_data)
    merged["cases"] = merged_cases
    merged.update(count_statuses(merged_cases))
    merged["total_cases"] = len(merged_cases)
    retry_block: Dict = {
        "cases": retried,
        "recovered": recovered,
        "still_not_passed": still_not_passed,
        "excluded_by_cap": excluded_by_cap,
    }
    if retry_duration is not None:
        retry_block["duration"] = retry_duration
    if max_retry_cases is not None:
        retry_block["max_retry_cases"] = max_retry_cases
    merged["retry"] = retry_block
    return merged


def copy_back_case_artifacts(
    src_dir: Path,
    report_dir: Path,
    shard: int,
    shard_type: str,
    case: Dict,
) -> int:
    """
    Refresh a retry-passed case's main-run result files.

    Copies the attempt's junit XML and case log (same filename, built from
    the preserved case_idx and nodeid) from the round the case finally
    passed over the main-run copies so the artifact contents match the
    updated cases.json status. Only called for cases whose retry status is
    no longer failed/error/timeout.

    Returns the number of files copied.
    """
    prefix = parse_test_results.get_shard_type_prefix(shard_type)
    safe_name = runner.sanitize_nodeid_for_filename(case["nodeid"])
    base = f"{prefix}-{shard}_{case['case_idx']}_{safe_name}"
    copied = 0
    for subdir, ext in (("junit_xmls", ".xml"), ("cases_logs", ".log")):
        src = src_dir / subdir / f"{base}{ext}"
        if not src.exists():
            continue
        dst_dir = report_dir / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / f"{base}{ext}")
        copied += 1
    return copied


def main() -> int:
    """Main entry point. Always returns 0 (see module docstring)."""
    args = parse_args()

    cases_file = Path(args.cases_file).resolve()
    report_dir = Path(args.report_dir).resolve()
    retry_report_dir = Path(args.retry_report_dir).resolve()
    test_dir = Path(args.test_dir).resolve()
    script_dir = Path(__file__).resolve().parent

    if not test_dir.is_dir():
        print(f"RETRY: test directory not found: {test_dir}; nothing to retry")
        write_github_outputs([("retry_needed", "false")])
        return 0

    if not cases_file.exists():
        # The shard runner crashed before recording results; the later
        # "Fail shard on missing test results" step fails the shard.
        print(f"RETRY: cases file {cases_file} not found; nothing to retry")
        write_github_outputs([("retry_needed", "false")])
        return 0

    try:
        cases_data = json.loads(cases_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"RETRY: could not parse {cases_file}: {exc}; nothing to retry")
        write_github_outputs([("retry_needed", "false")])
        return 0

    shard = cases_data.get("shard", 1)
    shard_type = cases_data.get("shard_type", "regular")
    prefix = parse_test_results.get_shard_type_prefix(shard_type)
    original_cases = cases_data.get("cases", [])

    failed = [
        (idx, case)
        for idx, case in enumerate(original_cases)
        if case.get("status") in RETRY_STATUSES
    ]
    print("=" * 80)
    print(f"Retry pass for shard {prefix}-{shard} ({shard_type})")
    print(f"Cases file: {cases_file}")
    print(f"Failed/error/timeout cases: {len(failed)}")
    print("=" * 80)

    if not failed:
        print("RETRY: no failed/error/timeout cases, skipping retry")
        write_github_outputs([
            ("retry_needed", "false"),
            ("retry_attempted", 0),
            ("retry_recovered", 0),
            ("retry_still_failed", 0),
        ])
        return 0

    excluded = 0
    if args.max_retry_cases > 0 and len(failed) > args.max_retry_cases:
        excluded = len(failed) - args.max_retry_cases
        failed = failed[:args.max_retry_cases]
        print(
            f"RETRY: capping serial retry at {args.max_retry_cases} cases; "
            f"{excluded} case(s) keep their original results"
        )

    # Preserve each case's original case_idx so the per-case junit XML / case
    # log filenames the retry workers write match the main-run filenames
    # (refresh = overwrite, still-failing = original files stay untouched).
    retry_report_dir.mkdir(parents=True, exist_ok=True)
    env_updates = runner.build_execution_env(
        test_dir, script_dir, None, shard, shard_type, args.device_env,
    )

    max_attempts = args.max_attempts
    worker_plan = args.worker_plan

    # Per-case final outcome across rounds: a case leaves the pending set as
    # soon as any round passes it; later rounds only re-run the
    # still-failing residue, so extra rounds cost seconds unless a case
    # genuinely hangs (bounded by --timeout per round).
    final_results: List[Dict] = []
    final_attempt_dir: Dict[str, Path] = {}
    attempts_used: Dict[str, int] = {}
    attempts_meta: List[Dict] = []
    total_retry_duration = 0.0

    console = runner.NonBlockingConsole()
    console.start()
    try:
        pending = list(failed)
        for attempt_no in range(1, max_attempts + 1):
            if not pending:
                break
            if max_attempts == 1:
                # Historical flat layout: retry evidence directly under the
                # retry report dir, no per-round subdirs.
                attempt_dir = retry_report_dir
            else:
                # Per-round subdirs: per-case junit/log filenames are
                # deterministic, so a flat layout would let later rounds
                # overwrite earlier rounds' evidence.
                attempt_dir = retry_report_dir / f"attempt_{attempt_no}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            tasks = []
            for idx, case in pending:
                tasks.append(runner.CaseExecutionTask(
                    case_idx=case.get("case_idx", idx + 1),
                    nodeid=case["nodeid"],
                    test_file=case.get("file", ""),
                ))

            worker_count = worker_plan[attempt_no - 1]
            attempt_desc = (
                f"Retrying {len(tasks)} case(s) "
                + (
                    "serially (single worker, one case at a time, "
                    f"per-case timeout {args.timeout}s)"
                    if worker_count == 1
                    else f"with {worker_count} concurrent workers "
                         f"(per-case timeout {args.timeout}s)"
                )
            )
            if max_attempts > 1:
                attempt_desc += f" [attempt {attempt_no}/{max_attempts}]"
            print(attempt_desc)

            _, round_duration, round_results = runner.run_tests_with_tasks_concurrent(
                tasks,
                shard,
                test_dir,
                attempt_dir,
                env_updates,
                args.timeout,
                args.verbose,
                shard_type,
                worker_count,  # max_workers: per-round, from --max-workers
                parse_test_results,
                None,  # quick_test
                console=console,
            )
            total_retry_duration += round_duration

            still_nodeids = set()
            recovered_round = 0
            for res in round_results:
                nid = res.get("nodeid")
                if not nid:
                    continue
                final_results.append(res)
                final_attempt_dir[nid] = attempt_dir
                attempts_used[nid] = attempt_no
                if res.get("status") in RETRY_STATUSES:
                    still_nodeids.add(nid)
                else:
                    recovered_round += 1
            pending = [pair for pair in pending if pair[1]["nodeid"] in still_nodeids]
            attempts_meta.append({
                "attempt": attempt_no,
                "attempted": len(tasks),
                "recovered": recovered_round,
                "still_not_passed": len(pending),
                "max_workers": worker_count,
            })
            print(f"[attempt {attempt_no}] {recovered_round} recovered, "
                  f"{len(pending)} still failed/error/timeout")

        # Deduplicate per nodeid keeping the LAST record (the case's final
        # round) so a case that failed round 1 and passed round 2 merges the
        # round-2 outcome.
        dedup: Dict[str, Dict] = {}
        for res in final_results:
            nid = res.get("nodeid")
            if nid:
                dedup[nid] = res
        merge_and_save(
            cases_data, cases_file, report_dir, retry_report_dir,
            shard, shard_type, list(dedup.values()), total_retry_duration,
            args.max_retry_cases, excluded,
            attempts_used=attempts_used,
            final_attempt_dir=final_attempt_dir,
            attempts_meta=attempts_meta,
            max_attempts=max_attempts,
        )
    except Exception as exc:
        # Best-effort by design: log the failure, keep whatever results are
        # already on disk (original cases.json if the merge did not run),
        # and let the workflow proceed.
        print(f"RETRY: unexpected error, keeping existing results: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        write_github_outputs([
            ("retry_needed", "true"),
            ("retry_error", "true"),
            ("retry_attempted", len(final_attempt_dir)),
            ("retry_recovered", 0),
            ("retry_still_failed", 0),
        ])
        return 0
    finally:
        # Stdout health gate, mirroring the shard runner: if the drain
        # thread is still blocked after the run, results are already safe
        # on disk — record it and continue quietly instead of hanging the
        # step on the final direct-print summary.
        if not console.finish(timeout=30):
            runner._write_console_stall_report(retry_report_dir, console)

    return 0


def merge_and_save(
    cases_data: Dict,
    cases_file: Path,
    report_dir: Path,
    retry_report_dir: Path,
    shard: int,
    shard_type: str,
    retry_results: List[Dict],
    retry_duration: float,
    max_retry_cases: int,
    excluded_by_cap: int,
    attempts_used: Optional[Dict[str, int]] = None,
    final_attempt_dir: Optional[Dict[str, Path]] = None,
    attempts_meta: Optional[List[Dict]] = None,
    max_attempts: int = 1,
) -> Dict:
    """Merge retry outcomes into cases.json/stats.json and refresh files."""
    prefix = parse_test_results.get_shard_type_prefix(shard_type)

    merged = merge_retry_results(
        cases_data, retry_results,
        retry_duration=retry_duration,
        max_retry_cases=max_retry_cases,
        excluded_by_cap=excluded_by_cap,
    )
    retry_block = merged["retry"]

    # Rounds each retried case consumed (1 = passed/failed its only round;
    # N = passed or finally failed on round N). Only meaningful with
    # --max-attempts > 1.
    if attempts_used:
        for case in merged["cases"]:
            if case.get("retried"):
                case["retry_attempts"] = attempts_used.get(case.get("nodeid"), 1)

    # Refresh the main-run result files only for cases whose outcome is no
    # longer failed/error/timeout (retry-passed, or newly skipped) so the
    # artifact contents match the updated cases.json status. The source is
    # the round the case finally passed (flat retry dir when single-round).
    # Still-failing cases keep their original files; their retry evidence
    # stays under the retry report dir, which is uploaded with the
    # test-reports artifact.
    refreshed_files = 0
    for case in merged["cases"]:
        if case.get("retried") and case.get("status") not in RETRY_STATUSES:
            src_dir = (final_attempt_dir or {}).get(case.get("nodeid"), retry_report_dir)
            refreshed_files += copy_back_case_artifacts(
                src_dir, report_dir, shard, shard_type, case,
            )

    # Rewrite the main cases.json / stats.json so every downstream consumer
    # (parse step, report generator, upstream dashboard) sees post-retry
    # numbers.
    parse_test_results.save_cases_file(str(report_dir), shard, merged, shard_type)

    stats_file = report_dir / f"shard_{prefix}-{shard}_stats.json"
    if stats_file.exists():
        try:
            stats = json.loads(stats_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stats = {}
        stats.update(count_statuses(merged["cases"]))
        stats["total"] = merged["total_cases"]
        stats["retried_cases"] = retry_block["cases"]
        stats["retry_recovered"] = retry_block["recovered"]
        parse_test_results.save_stats_file(str(report_dir), shard, stats, shard_type)

    # Retry-side summary. The name deliberately avoids the
    # shard_*_cases.json glob the report generator uses to discover
    # per-shard results, so this file is never double-counted.
    summary = {
        "shard": shard,
        "shard_type": shard_type,
        "max_attempts": max_attempts,
        "attempted": retry_block["cases"],
        "recovered": retry_block["recovered"],
        "still_not_passed": retry_block["still_not_passed"],
        "excluded_by_cap": retry_block["excluded_by_cap"],
        "duration": retry_duration,
        "refreshed_files": refreshed_files,
    }
    if attempts_meta:
        summary["attempts"] = attempts_meta
    (retry_report_dir / "retry_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 80)
    rounds_note = f" over up to {max_attempts} round(s)" if max_attempts > 1 else ""
    print(f"Retry summary:{rounds_note} {retry_block['cases']} retried, "
          f"{retry_block['recovered']} recovered, "
          f"{retry_block['still_not_passed']} still failed/error/timeout"
          + (f", {retry_block['excluded_by_cap']} skipped by cap" if excluded_by_cap else ""))
    print(f"Refreshed {refreshed_files} result file(s) for recovered cases")
    print(f"Shard totals after retry: passed={merged['passed']}, "
          f"failed={merged['failed']}, errors={merged['errors']}, "
          f"timeout={merged['timeout']}, skipped={merged['skipped']}, "
          f"total={merged['total_cases']}")
    print(f"Updated {cases_file}")
    print("=" * 80)

    write_github_outputs([
        ("retry_needed", "true"),
        ("retry_attempted", retry_block["cases"]),
        ("retry_recovered", retry_block["recovered"]),
        ("retry_still_failed", retry_block["still_not_passed"]),
    ])
    return merged


if __name__ == "__main__":
    # _exit_now: interpreter shutdown can deadlock on the stdout lock held
    # by extension daemon threads and abort with status 134 after results
    # are already on disk (same guard as the shard runners).
    runner._exit_now(main())
