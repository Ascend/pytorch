import os
import pickle
import re
from collections import defaultdict

from tools.flight_recorder.components.fr_logger import FlightRecorderLogger
from tools.flight_recorder.components.utils import get_valid_read_path

MAX_DEPTH = 3

logger: FlightRecorderLogger = FlightRecorderLogger()

SAFE_CLASSES = {
    # Built-in security type
    "builtins": {"str", "int", "float", "list", "dict", "tuple"},
}

exp = re.compile(r"^([a-zA-Z0-9_]{0,100}?)(\d+)$")


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Check if the module and class are in the whitelist
        if module in SAFE_CLASSES and name in SAFE_CLASSES[module]:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Forbidden class: {module}.{name}")


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
        logger.error(f"Failed to load data from {filename}: {e}")
    return rank, dump


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


def read_dir(args):
    """Load recorder data for all ranks"""
    prefix = args.prefix
    path = args.trace_dir
    details = {}
    version = ""
    for root, _, files in os.walk(path):
        current_depth = root.count(os.sep) - path.count(os.sep)
        if current_depth > MAX_DEPTH:
            logger.error("The current file depth has exceeded the maximum depth limit, which is set to {MAX_DEPTH}.")
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
            if not version:
                version = str(details[rank]["version"])
    return details, version
