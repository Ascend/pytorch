#!/usr/bin/env python3
"""Confirm passed/skipped cases from a finished blacklist shard run.

Blacklist pipelines expect every listed case to FAIL, so a case that reports
``passed`` or ``skipped`` in the main shard run is a removal candidate. A
single unexpected pass may be flaky, so this step re-runs those cases before
the workflow parses the results:

    - Only cases whose recorded status is in ``--statuses`` (default
      ``passed,skipped``) are re-run; none -> print a skip note and exit 0.
    - ``--max-rounds`` N is the TOTAL number of executions including the main
      run (default 2 = one confirmation round). N <= 1 disables the step.
    - A case is CONFIRMED only when every additional round reports a
      non-failing status (passed/skipped). The first failed/error/timeout
      round wins: the case takes that failing status, so it stays in the
      blacklist and is NOT reported as a removal candidate. Confirmed cases
      keep their original status and get ``confirmed_pass=true``.
    - The failing round's junit XML / case log are copied back over the
      main-run files for flipped cases, so the artifact contents match the
      rewritten status.
    - cases.json / stats.json are rewritten so every downstream consumer
      (parse step, removal candidates, Excel report) sees the confirmed
      outcome.
    - The script ALWAYS exits 0: the shard verdict is decided by the later
      workflow steps.

Usage:
    python retry_passed_cases.py \
        --cases-file test-reports/shard_reg-1_cases.json \
        --test-dir pytorch/test \
        --report-dir test-reports \
        --retry-report-dir test-reports/retry-passed \
        --timeout 500 \
        --device-env privateuse1,cpu \
        --max-rounds 2 \
        --max-workers 8 \
        --verbose
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import run_npu_test_shard as runner
import parse_test_results
import retry_failed_cases as rfc


FAILING_STATUSES = rfc.RETRY_STATUSES  # failed / error / timeout
DEFAULT_STATUSES = "passed,skipped"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Confirm passed/skipped cases from a finished blacklist shard run"
    )
    parser.add_argument("--cases-file", type=str, required=True,
                        help="Shard cases.json written by the shard runner")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Path to the pytorch/test directory")
    parser.add_argument("--report-dir", type=str, default="test-reports",
                        help="Main-run report directory (default: test-reports)")
    parser.add_argument("--retry-report-dir", type=str,
                        default="test-reports/retry-passed",
                        help="Directory for confirmation-round evidence "
                             "(default: test-reports/retry-passed)")
    parser.add_argument("--timeout", type=int, default=500,
                        help="Per-case timeout in seconds, same as the main run "
                             "(default: 500)")
    parser.add_argument("--device-env", type=str, default="privateuse1",
                        help="PYTORCH_TESTING_DEVICE_ONLY_FOR value (default: privateuse1)")
    parser.add_argument("--statuses", type=str, default=DEFAULT_STATUSES,
                        help="Comma-separated statuses to confirm "
                             f"(default: {DEFAULT_STATUSES})")
    parser.add_argument("--max-rounds", type=int, default=2,
                        help="TOTAL executions per case including the main run; "
                             "2 = one confirmation round (default: 2)")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Concurrent worker subprocesses per confirmation "
                             "round (default: 8)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.max_rounds < 1:
        parser.error("--max-rounds must be >= 1")
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")
    args.status_set = {
        part.strip() for part in str(args.statuses).split(",") if part.strip()
    }
    if not args.status_set:
        parser.error("--statuses must name at least one status")
    return args


def write_outputs(pairs) -> None:
    """Append key=value pairs to $GITHUB_OUTPUT when running in Actions."""
    rfc.write_github_outputs(pairs)


def main() -> int:
    """Main entry point. Always returns 0 (see module docstring)."""
    args = parse_args()

    cases_file = Path(args.cases_file).resolve()
    report_dir = Path(args.report_dir).resolve()
    retry_report_dir = Path(args.retry_report_dir).resolve()
    test_dir = Path(args.test_dir).resolve()
    script_dir = Path(__file__).resolve().parent

    extra_rounds = args.max_rounds - 1
    if extra_rounds < 1:
        print("CONFIRM: --max-rounds <= 1; no confirmation round to run")
        write_outputs([("confirm_needed", "false"), ("confirm_targets", 0),
                       ("confirm_confirmed", 0), ("confirm_flipped", 0)])
        return 0

    if not test_dir.is_dir():
        print(f"CONFIRM: test directory not found: {test_dir}; nothing to confirm")
        write_outputs([("confirm_needed", "false"), ("confirm_targets", 0),
                       ("confirm_confirmed", 0), ("confirm_flipped", 0)])
        return 0

    if not cases_file.exists():
        # The shard runner crashed before recording results; the later
        # "Fail shard on missing test results" step fails the shard.
        print(f"CONFIRM: cases file {cases_file} not found; nothing to confirm")
        write_outputs([("confirm_needed", "false"), ("confirm_targets", 0),
                       ("confirm_confirmed", 0), ("confirm_flipped", 0)])
        return 0

    try:
        cases_data = json.loads(cases_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"CONFIRM: could not parse {cases_file}: {exc}; nothing to confirm")
        write_outputs([("confirm_needed", "false"), ("confirm_targets", 0),
                       ("confirm_confirmed", 0), ("confirm_flipped", 0)])
        return 0

    shard = cases_data.get("shard", 1)
    shard_type = cases_data.get("shard_type", "regular")
    prefix = parse_test_results.get_shard_type_prefix(shard_type)
    original_cases = cases_data.get("cases", [])

    targets: List[Tuple[int, Dict]] = [
        (idx, case)
        for idx, case in enumerate(original_cases)
        if case.get("status") in args.status_set
    ]
    print("=" * 80)
    print(f"Confirmation pass for shard {prefix}-{shard} ({shard_type})")
    print(f"Cases file: {cases_file}")
    print(f"Statuses to confirm: {sorted(args.status_set)}")
    print(f"Cases to confirm: {len(targets)} "
          f"(total executions per case: {args.max_rounds})")
    print("=" * 80)

    if not targets:
        print("CONFIRM: no passed/skipped cases, skipping confirmation")
        write_outputs([("confirm_needed", "false"), ("confirm_targets", 0),
                       ("confirm_confirmed", 0), ("confirm_flipped", 0)])
        return 0

    retry_report_dir.mkdir(parents=True, exist_ok=True)
    env_updates = runner.build_execution_env(
        test_dir, script_dir, None, shard, shard_type, args.device_env,
    )

    final_results: Dict[str, Dict] = {}
    final_dirs: Dict[str, Path] = {}
    rounds_used: Dict[str, int] = {}
    total_duration = 0.0

    console = runner.NonBlockingConsole()
    console.start()
    try:
        pending = list(targets)
        for round_no in range(1, extra_rounds + 1):
            if not pending:
                break
            attempt_dir = retry_report_dir / f"attempt_{round_no}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            tasks = [
                runner.CaseExecutionTask(
                    case_idx=case.get("case_idx", idx + 1),
                    nodeid=case["nodeid"],
                    test_file=case.get("file", ""),
                )
                for idx, case in pending
            ]
            print(f"Confirming {len(tasks)} case(s) with {args.max_workers} "
                  f"worker(s) (round {round_no}/{extra_rounds}, "
                  f"per-case timeout {args.timeout}s)")

            _, round_duration, round_results = runner.run_tests_with_tasks_concurrent(
                tasks,
                shard,
                test_dir,
                attempt_dir,
                env_updates,
                args.timeout,
                args.verbose,
                shard_type,
                args.max_workers,
                parse_test_results,
                None,  # quick_test
                console=console,
            )
            total_duration += round_duration

            by_nodeid = {
                res.get("nodeid"): res
                for res in round_results
                if res.get("nodeid")
            }
            still_pending: List[Tuple[int, Dict]] = []
            flipped = 0
            for idx, case in pending:
                nodeid = case["nodeid"]
                res = by_nodeid.get(nodeid)
                if res is None:
                    # No result for this case in this round: try again next
                    # round; if it never resolves, its original status stays.
                    still_pending.append((idx, case))
                    continue
                final_results[nodeid] = res
                final_dirs[nodeid] = attempt_dir
                rounds_used[nodeid] = round_no
                if res.get("status") in FAILING_STATUSES:
                    flipped += 1
                    continue
                still_pending.append((idx, case))
            pending = still_pending
            print(f"[round {round_no}/{extra_rounds}] {flipped} flipped to "
                  f"failing, {len(pending)} still awaiting confirmation")

        merged_cases = []
        confirmed = 0
        flipped = 0
        for case in original_cases:
            nodeid = case.get("nodeid")
            new_case = dict(case)
            res = final_results.get(nodeid)
            if res is not None:
                new_case["confirm_rounds"] = rounds_used.get(nodeid, 1)
                if res.get("status") in FAILING_STATUSES:
                    # Any failing round un-confirms the case: keep it in the
                    # blacklist by restoring the failing status.
                    new_case["status"] = res.get("status")
                    new_case["duration"] = res.get("duration", case.get("duration", 0.0))
                    new_case["returncode"] = res.get("returncode", case.get("returncode", 1))
                    new_case["message"] = res.get("message", case.get("message", ""))
                    new_case["command"] = res.get("command", case.get("command", ""))
                    new_case["first_attempt_status"] = case.get("status")
                    new_case["confirmed_pass"] = False
                    flipped += 1
                else:
                    new_case["confirmed_pass"] = True
                    confirmed += 1
            merged_cases.append(new_case)

        merged = dict(cases_data)
        merged["cases"] = merged_cases
        merged.update(rfc.count_statuses(merged_cases))
        merged["total_cases"] = len(merged_cases)
        merged["confirm_passed"] = {
            "targets": len(targets),
            "confirmed": confirmed,
            "flipped_to_failed": flipped,
            "total_rounds": args.max_rounds,
            "duration": total_duration,
        }

        # Refresh main-run artifacts for flipped cases so junit XML / case log
        # match the rewritten status.
        refreshed_files = 0
        for case in merged_cases:
            nodeid = case.get("nodeid")
            if nodeid in final_results and not case.get("confirmed_pass", True):
                src_dir = final_dirs.get(nodeid, retry_report_dir)
                refreshed_files += rfc.copy_back_case_artifacts(
                    src_dir, report_dir, shard, shard_type, case,
                )

        parse_test_results.save_cases_file(
            str(report_dir), shard, merged, shard_type
        )

        stats_file = report_dir / f"shard_{prefix}-{shard}_stats.json"
        if stats_file.exists():
            try:
                stats = json.loads(stats_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                stats = {}
            stats.update(rfc.count_statuses(merged_cases))
            stats["total"] = merged["total_cases"]
            stats["confirm_passed"] = confirmed
            stats["confirm_flipped_to_failed"] = flipped
            parse_test_results.save_stats_file(str(report_dir), shard, stats, shard_type)

        summary = {
            "shard": shard,
            "shard_type": shard_type,
            "statuses": sorted(args.status_set),
            "total_rounds": args.max_rounds,
            "targets": len(targets),
            "confirmed": confirmed,
            "flipped_to_failed": flipped,
            "duration": total_duration,
            "refreshed_files": refreshed_files,
        }
        (retry_report_dir / "retry_passed_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        print("=" * 80)
        print(f"Confirmation summary: {len(targets)} targeted, {confirmed} "
              f"confirmed passed/skipped, {flipped} flipped to failing")
        print(f"Refreshed {refreshed_files} result file(s) for flipped cases")
        print(f"Shard totals after confirmation: passed={merged['passed']}, "
              f"failed={merged['failed']}, errors={merged['errors']}, "
              f"timeout={merged['timeout']}, skipped={merged['skipped']}, "
              f"total={merged['total_cases']}")
        print("=" * 80)

        write_outputs([
            ("confirm_needed", "true"),
            ("confirm_targets", len(targets)),
            ("confirm_confirmed", confirmed),
            ("confirm_flipped", flipped),
        ])
    except Exception as exc:
        # Best-effort by design: log the failure, keep whatever results are
        # already on disk, and let the workflow proceed.
        print(f"CONFIRM: unexpected error, keeping existing results: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        write_outputs([
            ("confirm_needed", "true"),
            ("confirm_error", "true"),
            ("confirm_targets", len(targets)),
            ("confirm_confirmed", 0),
            ("confirm_flipped", 0),
        ])
        return 0
    finally:
        # Stdout health gate, mirroring the shard runner: results are already
        # on disk, so a stalled drain thread must not hang the step.
        if not console.finish(timeout=30):
            runner._write_console_stall_report(retry_report_dir, console)

    return 0


if __name__ == "__main__":
    # _exit_now: same interpreter-shutdown deadlock guard as the shard runners.
    runner._exit_now(main())
