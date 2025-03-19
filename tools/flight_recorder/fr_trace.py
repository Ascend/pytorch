import os
import re
import pickle
import logging
from collections import defaultdict
import argparse

from components.utils import get_valid_read_path

__all__ = []
exp = re.compile(r"^([a-zA-Z0-9_]{0,100}?)(\d+)$")
MAX_DEPTH = 3


logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # Set format
    handlers=[logging.StreamHandler()],  # Output to console
)


SAFE_CLASSES = {
    # Built-in security type
    "builtins": {"str", "int", "float", "list", "dict", "tuple"},
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Check if the module and class are in the whitelist
        if module in SAFE_CLASSES and name in SAFE_CLASSES[module]:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Forbidden class: {module}.{name}")


def determine_prefix(files):
    possible_prefixes: defaultdict[str, set[int]] = defaultdict(set)
    for f in files:
        m = exp.search(f)
        if m:
            p, r = m.groups()
            possible_prefixes[p].add(int(r))
    if len(possible_prefixes) == 1:
        prefix = next(iter(possible_prefixes))
        return prefix
    else:
        raise ValueError(
            "Unable to automatically determine the common prefix for the trace file names. "
            "Please specify --prefix argument manually"
        )


def load_recorder_data(path, prefix):
    """Load recorder data for all ranks"""
    details = {}
    for root, _, files in os.walk(path):
        current_depth = root.count(os.sep) - path.count(os.sep)
        if current_depth > MAX_DEPTH:
            logging.error("The current file depth has exceeded the maximum depth limit, which is set to {MAX_DEPTH}.")
            break
        if prefix is None:
            prefix = determine_prefix(files)
        for f in files:
            if "py_traceback" in f:
                continue
            if f.find(prefix) != 0:
                continue
            rank, dump = read_dump(prefix, os.path.join(root, f))
            details[rank] = dump
    return details


def read_dump(prefix, filename):
    basename = os.path.basename(filename)
    try:
        rank = int(basename[len(prefix):])
    except ValueError as e:
        raise ValueError(f"Cannot extract rank from '{basename}' with prefix '{prefix}'.") from e
    filename = get_valid_read_path(filename)
    try:
        with open(filename, "rb") as infile:
            dump = SafeUnpickler(infile).load()
    except Exception as e:
        logging.error(f"Failed to load data from {filename}: {e}")
    return rank, dump


def extract_hccl_info(recorder_dict):
    """Extract HCCL related information from recorder data"""
    hccl_dict = {}
    for rank, recorder in recorder_dict.items():
        entries = recorder.get("entries", [])
        if not entries:
            continue
        last_entry = entries[-1]
        hccl_dict[rank] = {
            "state": last_entry.get("state", None),
            "record_id": last_entry.get("record_id", None),
            "pg_id": last_entry.get("pg_id", None),
            "time_discovered_completed_ns": last_entry.get("time_discovered_completed_ns", None),
            "name": last_entry.get("frames", [{}])[0].get("name", None),
        }
    return hccl_dict


def analyze_pg_groups(hccl_dict):
    """Analyze HCCL data, group by pg_id and check for problems"""
    pg_groups = defaultdict(list)
    for _, op in hccl_dict.items():
        pg_groups[op["pg_id"]].append(op)

    for pg_id, group in pg_groups.items():
        scheduled_ops = [op for op in group if op["state"] == "scheduled"]
        completed_ops = [op for op in group if op["state"] == "completed"]
        # Case 1: All NPUs are scheduled and have the same record_id and name
        if len(scheduled_ops) == len(group):
            record_id = scheduled_ops[0]["record_id"]
            name = scheduled_ops[0]["name"]
            if all(op["record_id"] == record_id and op["name"] == name for op in scheduled_ops):
                logging.info(
                    f"The pg_id {pg_id}'s Communication Operator {name} "
                    "executed too slowly, causing the HCCL to time out."
                )
                continue

        # Case 2: There is a completed operator and its record_id is 1 less than other scheduled operators
        if completed_ops and scheduled_ops:
            completed_op = completed_ops[0]
            scheduled_record_id = scheduled_ops[0]["record_id"]
            if completed_op["record_id"] == scheduled_record_id - 1:
                logging.info(
                    f"The pg_id {pg_id}'s rank {completed_op['pg_id']}'s "
                    "Computational task took too long, causing the other ranks' "
                    "HCCL task to time out."
                )
                continue

        # Case 3: All operators are completed
        if not scheduled_ops and completed_ops:
            latest_op = max(completed_ops, key=lambda x: x["time_discovered_completed_ns"] or 0)
            logging.info(
                f"The computational task of the pg_id {pg_id} "
                f"after the communication operator {latest_op['name']} "
                "took too long."
            )
            continue

        # Unrecognized cases
        logging.info(f"The situation for pg_id {pg_id} cannot be recognized!")


def main():
    # Parsing command line arguments with argparse
    parser = argparse.ArgumentParser(description="PyTorch Flight recorder analyzing script.")
    parser.add_argument(
        "trace_dir",
        type=str,
        help="Directory containing one trace file per rank, named with <prefix>_<rank>.",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        help=(
            "Common filename prefix to strip such that rank can be extracted. "
            "If not specified, will attempt to infer a common prefix."
        ),
        default=None,
    )
    args = parser.parse_args()

    path = get_valid_read_path(args.trace_dir, is_dir=True)

    recorder_dict = load_recorder_data(path, args.prefix)
    if not recorder_dict:
        logging.error("No valid recorder data found.")
        return

    # Extract HCCL information
    hccl_dict = extract_hccl_info(recorder_dict)

    # Analyzing HCCL data
    analyze_pg_groups(hccl_dict)


if __name__ == "__main__":
    main()
