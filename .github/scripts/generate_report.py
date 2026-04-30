#!/usr/bin/env python3
"""
Generate consolidated test report from all shard results.

Reads all shard result JSON files and generates a Markdown summary report
and a detailed JSON report.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def load_shard_results(reports_root: str) -> List[Dict]:
    """Load all shard result JSON files from the reports directory."""
    reports_path = Path(reports_root)
    if not reports_path.exists():
        raise FileNotFoundError(f"Reports directory not found: {reports_root}")

    results = []
    for result_file in reports_path.rglob("*_results.json"):
        with open(result_file) as f:
            data = json.load(f)
            results.append(data)

    return results


def aggregate_stats(shard_results: List[Dict]) -> Dict:
    """Aggregate statistics from all shard results."""
    total_stats = {
        "passed": 0,
        "failed": 0,
        "error": 0,
        "skipped": 0,
        "timeout": 0,
        "crashed": 0,
        "xfail": 0,
        "xpass": 0,
        "unknown": 0,
    }

    total_cases = 0
    for shard in shard_results:
        total_cases += shard.get("total_cases", 0)
        stats = shard.get("stats", {})
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

    total_stats["total_cases"] = total_cases
    total_stats["pass_rate"] = (
        round(total_stats["passed"] / total_cases * 100, 2) if total_cases > 0 else 0
    )

    return total_stats


def generate_markdown_report(stats: Dict, shard_results: List[Dict]) -> str:
    """Generate a Markdown summary report."""
    lines = [
        "# PyTorch NPU Test Report",
        "",
        f"**Generated:** {stats.get('timestamp', 'N/A')}",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Total Cases | {stats['total_cases']} |",
        f"| Passed | {stats['passed']} |",
        f"| Failed | {stats['failed']} |",
        f"| Error | {stats['error']} |",
        f"| Timeout | {stats['timeout']} |",
        f"| Crashed | {stats['crashed']} |",
        f"| Skipped | {stats['skipped']} |",
        f"| XFail | {stats['xfail']} |",
        f"| XPass | {stats['xpass']} |",
        f"| Pass Rate | {stats['pass_rate']}% |",
        "",
        "## Shard Details",
        "",
    ]

    for shard in sorted(shard_results, key=lambda x: x.get("shard_index", 0)):
        shard_idx = shard.get("shard_index", "?")
        shard_stats = shard.get("stats", {})
        lines.append(f"### Shard {shard_idx}")
        lines.append("")
        lines.append(f"- Total cases: {shard.get('total_cases', 0)}")
        lines.append(f"- Passed: {shard_stats.get('passed', 0)}")
        lines.append(f"- Failed: {shard_stats.get('failed', 0)}")
        lines.append(f"- Error: {shard_stats.get('error', 0)}")
        lines.append(f"- Timeout: {shard_stats.get('timeout', 0)}")
        lines.append(f"- Crashed: {shard_stats.get('crashed', 0)}")
        lines.append("")

    # Add failed cases section
    failed_cases = []
    for shard in shard_results:
        for result in shard.get("results", []):
            if result.get("status") in ["failed", "error", "timeout", "crashed"]:
                failed_cases.append({
                    "case_id": result.get("case_id", "?"),
                    "status": result.get("status", "?"),
                    "duration": result.get("duration", 0),
                })

    if failed_cases:
        lines.append("## Failed Cases")
        lines.append("")
        lines.append("| Case ID | Status | Duration |")
        lines.append("|---------|--------|----------|")
        for case in failed_cases[:100]:  # Limit to first 100 for readability
            lines.append(f"| {case['case_id']} | {case['status']} | {case['duration']}s |")

        if len(failed_cases) > 100:
            lines.append(f"\n*Showing first 100 of {len(failed_cases)} failed cases*")

    return "\n".join(lines)


def generate_json_report(stats: Dict, shard_results: List[Dict]) -> Dict:
    """Generate a detailed JSON report."""
    report = {
        "summary": stats,
        "shards": shard_results,
        "failed_cases": [],
    }

    for shard in shard_results:
        for result in shard.get("results", []):
            if result.get("status") in ["failed", "error", "timeout", "crashed"]:
                report["failed_cases"].append(result)

    return report


def main():
    parser = argparse.ArgumentParser(description="Generate consolidated test report")
    parser.add_argument("--reports-root", required=True, help="Root directory with shard results")
    parser.add_argument("--output-markdown", required=True, help="Output Markdown file path")
    parser.add_argument("--output-json", required=True, help="Output JSON file path")

    args = parser.parse_args()

    shard_results = load_shard_results(args.reports_root)
    print(f"Loaded {len(shard_results)} shard results")

    stats = aggregate_stats(shard_results)
    stats["timestamp"] = datetime.utcnow().isoformat()

    print(f"Total cases: {stats['total_cases']}")
    print(f"Pass rate: {stats['pass_rate']}%")

    # Generate Markdown report
    markdown = generate_markdown_report(stats, shard_results)
    with open(args.output_markdown, "w") as f:
        f.write(markdown)
    print(f"Markdown report saved to {args.output_markdown}")

    # Generate JSON report
    json_report = generate_json_report(stats, shard_results)
    with open(args.output_json, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"JSON report saved to {args.output_json}")


if __name__ == "__main__":
    main()