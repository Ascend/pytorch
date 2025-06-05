__all__ = []

import argparse
import math
from typing import Any
import os
import re
import sys
import stat

from tools.flight_recorder.components.fr_logger import FlightRecorderLogger
from tools.flight_recorder.components.types import (
    Group,
    MatchInfo,
    MatchState,
    MatchStateRecord,
    Membership,
    Op,
)

logger: FlightRecorderLogger = FlightRecorderLogger()

try:
    from tabulate import tabulate
except ModuleNotFoundError:
    logger.debug("tabulate is not installed. Proceeding without it.")

PATH_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.-]")
MAX_READ_FILE_SIZE_4G = 4294967296  # 4G, 4 * 1024 * 1024 * 1024
MAX_READ_FILE_SIZE_32G = 34359738368  # 32G, 32 * 1024 * 1024 * 1024
MAX_READ_FILE_SIZE_512G = 549755813888  # 512G, 512 * 1024 * 1024 * 1024

# group not writable, others no permission, max stat is 750
WRITE_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH | stat.S_IROTH | stat.S_IXOTH
# group not writable, others not writable, max stat is 755
READ_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH


def type_to_str(value_type):
    return " or ".join(ii.__name__ for ii in value_type) if isinstance(value_type, tuple) else value_type.__name__


def check_type(value, value_type, param_name="value"):
    if not isinstance(value, value_type):
        raise TypeError("{} must be {}, not {}.".format(param_name, type_to_str(value_type), type(value).__name__))


def get_valid_path(path):
    check_type(path, str, "path")
    if not path or len(path) == 0:
        raise ValueError("The value of the path cannot be empty.")
    if PATH_WHITE_LIST_REGEX.search(path):  # Check special char
        raise ValueError("Input path contains invalid characters.")  # Not printing out the path value for invalid char
    path = os.path.expanduser(path)
    if os.path.islink(os.path.abspath(path)):  # when checking link, get rid of the "/" at the path tail if any
        raise ValueError("The value of the path cannot be a symbolic link: {}.".format(path))

    real_path = os.path.realpath(path)

    if len(real_path) > 4096:
        raise ValueError("The length of file path should be less than 4096.")

    if real_path != path and PATH_WHITE_LIST_REGEX.search(real_path):  # Check special char again
        raise ValueError("Input path contains invalid characters.")  # Not printing out the path value for invalid char

    return real_path


def is_belong_to_user_or_group(file_stat):
    return file_stat.st_uid == os.getuid() or file_stat.st_gid in os.getgroups()


def get_valid_read_path(path, size_max=MAX_READ_FILE_SIZE_4G, check_user_stat=True, is_dir=False):
    real_path = get_valid_path(path)
    if not is_dir and not os.path.isfile(real_path):
        raise ValueError("The path {} doesn't exists or not a file.".format(path))
    if is_dir and not os.path.isdir(real_path):
        raise ValueError("The path {} doesn't exists or not a directory.".format(path))

    file_stat = os.stat(real_path)
    if check_user_stat and not sys.platform.startswith("win") and not is_belong_to_user_or_group(file_stat):
        raise ValueError("The file {} doesn't belong to the current user or group.".format(path))
    if check_user_stat and os.stat(path).st_mode & READ_FILE_NOT_PERMITTED_STAT > 0:
        raise ValueError("The file {} is group writable, or is others writable.".format(path))
    if not os.access(real_path, os.R_OK) or file_stat.st_mode & stat.S_IRUSR == 0:  # At least been 400
        raise ValueError("Current user doesn't have read permission to the file {}.".format(path))
    if not is_dir and size_max > 0 and file_stat.st_size > size_max:
        raise ValueError("The file {} exceeds size limitation of {}.".format(path, size_max))
    return real_path


def check_write_directory(dir_name, check_user_stat=True):
    real_dir_name = get_valid_path(dir_name)
    if not os.path.isdir(real_dir_name):
        raise ValueError("The file writen directory {} doesn't exists.".format(dir_name))

    file_stat = os.stat(real_dir_name)
    if check_user_stat and not sys.platform.startswith("win") and not is_belong_to_user_or_group(file_stat):
        raise ValueError("The file writen directory {} doesn't belong to the current user or group.".format(dir_name))
    if not os.access(real_dir_name, os.W_OK):
        raise ValueError("Current user doesn't have writen permission to file writen directory {}.".format(dir_name))


def get_valid_write_path(path, check_user_stat=True, is_dir=False, warn_exists=True):
    real_path = get_valid_path(path)
    real_path_dir = real_path if is_dir else os.path.dirname(real_path)
    check_write_directory(real_path_dir, check_user_stat=check_user_stat)

    if not is_dir and os.path.exists(real_path):
        if os.path.isdir(real_path):
            raise ValueError("The file {} exist and is a directory.".format(path))
        if check_user_stat and os.stat(real_path).st_uid != os.getuid():  # Has to be exactly belonging to current user
            raise ValueError("The file {} doesn't belong to the current user.".format(path))
        if check_user_stat and os.stat(real_path).st_mode & WRITE_FILE_NOT_PERMITTED_STAT > 0:
            raise ValueError("The file {} permission for others is not 0, or is group writable.".format(path))
        if not os.access(real_path, os.W_OK):
            raise ValueError("The file {} exist and not writable.".format(path))
        if warn_exists:
            logger.warning("%s already exist. The original file will be overwritten.", path)
    return real_path


def format_frame(frame: dict[str, str]) -> str:
    name = frame["name"]
    filename = frame["filename"]
    line = frame["line"]
    return f"{name} at {filename}:{line}"


def format_frames(frames: list[dict[str, str]]) -> str:
    formatted_frames = []
    for frame in frames:
        formatted_frames.append(format_frame(frame))
    return "\n".join(formatted_frames)


def match_one_event(
    event_a: dict[Any, Any],
    event_b: dict[Any, Any],
    memberships: dict[str, set[Any]],
    pg_name: str,
) -> MatchInfo:
    op_a = Op(event_a, memberships, pg_name)
    op_b = Op(event_b, memberships, pg_name)
    return op_a.match(op_b)


def check_size_alltoall(alltoall_cases: list[dict[str, Any]]) -> tuple[bool, int, int]:
    input_numel = 0
    output_numel = 0
    for e in alltoall_cases:
        input_numel += math.prod(e["input_sizes"][0])
        output_numel += math.prod(e["output_sizes"][0])
    return input_numel != output_numel, input_numel, output_numel


class ProcessGroupData:
    def __init__(self, pg_guids: dict[tuple[str, int], str], pg_name: str, desc: str, mismatch: dict[str, int]):
        self.pg_guids, self.pg_name, self.desc, self.mismatch = pg_guids, pg_name, desc, mismatch


def check_current_entry_match(
    all_entries: dict[int, list[dict[str, Any]]],
    current_entry: dict[str, Any],
    _memberships: dict[str, set[Any]],
    pg_data: ProcessGroupData,
    match_record: MatchStateRecord,
) -> None:
    pg_guids, pg_name, mismatch, desc = pg_data.pg_guids, pg_data.pg_name, pg_data.mismatch, pg_data.desc
    for rank in match_record.expected_ranks.intersection(set(match_record.other_ranks)):
        for entry_idx, entry in enumerate(all_entries[rank]):
            # step over ops from other PGs
            # only check match state when seq_id matches
            if (
                pg_guids[(entry["process_group"][0], rank)] == pg_name
                and entry["collective_seq_id"] == match_record.entry_state.collective_seq_id
            ):
                match_info = match_one_event(current_entry, entry, _memberships, pg_name)
                if match_info.state in [MatchState.FULLY_MATCHED, MatchState.UNDECIDED] and mismatch[pg_name] == 0:
                    match_record.found_ranks.add(rank)
                    match_record.found_idx[rank] = entry_idx
                    match_record.has_undecided_case = match_info.state == MatchState.UNDECIDED
                else:
                    match_record.candidate_ranks.add(rank)
                    match_record.candidate_idx[rank] = entry_idx
                    if match_info.state not in [
                        MatchState.FULLY_MATCHED,
                        MatchState.UNDECIDED,
                    ]:
                        match_record.errors.add((rank, match_info))
                break


class EntryContext:
    def __init__(self, all_entries, current_entry, dumps_ranks, first_rank):
        self.all_entries = all_entries
        self.current_entry = current_entry
        self.dumps_ranks = dumps_ranks
        self.first_rank = first_rank


def error_analysis(
    entry_context: EntryContext,
    match_record: MatchStateRecord, # all
    mismatch: dict[str, int],  # all
    version: tuple[int, int],  # 2
    pg_name: str,  # all, mismatch
) -> None:
    all_entries = entry_context.all_entries
    current_entry = entry_context.current_entry
    dumps_ranks = entry_context.dumps_ranks
    first_rank = entry_context.first_rank
    major_v, minor_v = version[0], version[1]
    # case one: not every rank join the collective or in the flight recorder.
    if (
        match_record.candidate_ranks | match_record.found_ranks
    ) != match_record.expected_ranks and match_record.expected_ranks - (
        match_record.candidate_ranks | match_record.found_ranks
    ) <= dumps_ranks:
        mismatch[pg_name] += 1
        logger_msg = "Not all ranks joining collective, sequence number: %s"
        missing_ranks = match_record.expected_ranks - (match_record.candidate_ranks | match_record.found_ranks)
        match_record.entry_state.log(
            logger, logger_msg, format_frames, additional_info={"missing_ranks": missing_ranks}
        )
        match_record.candidate_ranks.update(match_record.found_ranks)
        match_record.candidate_idx.update(match_record.found_idx)
        match_record.found_idx.clear()
        match_record.found_ranks.clear()
    elif len(match_record.candidate_ranks) == 1 and dumps_ranks == match_record.expected_ranks:
        # case two: alltoall or alltoall_base case.
        if match_record.has_undecided_case:
            alltoall_cases = [current_entry] + [
                all_entries[rank][match_record.found_idx[rank]] for rank in match_record.found_ranks
            ]
            fail_check, total_input_numel, total_output_numel = check_size_alltoall(alltoall_cases)
            if major_v <= 2 and minor_v <= 3:
                # We don't log the input/output sizes for alltoall before v2.4,
                # so we don't consider the size mismatch as an error for now.
                fail_check = False
            if fail_check:
                # When we see errors in all_to_all, it's hard to tell which rank is the source of the error.
                mismatch[pg_name] += 1
                logger_msg = "Input/output mismatch in the collective sequence number: %s"
                match_record.entry_state.log(
                    logger,
                    logger_msg,
                    format_frames,
                    additional_info={"total_numel": (total_input_numel, total_output_numel)},
                )
                match_record.candidate_ranks.update(match_record.found_ranks)
                match_record.candidate_idx.update(match_record.found_idx)
                match_record.found_idx.clear()
                match_record.found_ranks.clear()
                match_record.errors.add((first_rank, MatchInfo(MatchState.SIZE_OR_SYNTAX_MISMATCH)))
            else:
                match_record.found_ranks.update(match_record.candidate_ranks)
                match_record.found_idx.update(match_record.candidate_idx)
                match_record.candidate_idx.clear()
                match_record.candidate_ranks.clear()
        # case three: all joined and everything matches on all ranks.
        else:
            match_record.found_ranks.update(match_record.candidate_ranks)
            match_record.found_idx.update(match_record.candidate_idx)
            match_record.candidate_idx.clear()
            match_record.candidate_ranks.clear()
    # case four: mismatch cases due to not same type, size mismatch or state mismatch.
    elif len(match_record.errors) > 0:
        mismatch[pg_name] += 1
        logger_msg = "Collective sequence number: %s has errors"
        match_record.entry_state.log(logger, logger_msg, format_frames, errors=match_record.errors)
        match_record.candidate_ranks.update(match_record.found_ranks)
        match_record.candidate_idx.update(match_record.found_idx)
        match_record.found_idx.clear()
        match_record.found_ranks.clear()
    # partial analysis case when we cannot decide what's wrong with this collective entry.
    else:
        match_record.candidate_ranks.update(match_record.found_ranks)
        match_record.candidate_idx.update(match_record.found_idx)
        match_record.found_idx.clear()
        match_record.found_ranks.clear()
        if match_record.expected_ranks - dumps_ranks:
            mismatch[pg_name] += 1
            logger.info(
                "We cannot decide what's wrong with this collective entry "
                "because we missed FR dumps from ranks (%s) so we don't have enough "
                "information. If you want to debug further use -j to dump all raw trace",
                str(match_record.expected_ranks - dumps_ranks),
            )
        else:
            logger.info(
                "No errors found for this collective entry, There could be some "
                "other reasons why we see collective timeout."
            )


def just_print_entries(
    all_entries: dict[int, list[dict[str, Any]]],
    _groups: dict[str, Group],
    _memberships: dict[str, set[Any]],
    _pg_guids: dict[tuple[str, int], str],
    args: argparse.Namespace,
) -> None:
    rows = []
    ranks = sorted(all_entries.keys())
    headers = [f"Rank {rank}" for rank in ranks if args.selected_ranks is None or rank in args.selected_ranks]
    progress = True
    while progress:
        progress = False
        row = []
        for rank in ranks:
            if args.selected_ranks is not None and rank not in args.selected_ranks:
                continue
            if len(all_entries[rank]) == 0:
                row.append("")
            else:
                entry = all_entries[rank].pop(0)
                pg_name = _pg_guids[(entry["process_group"][0], rank)]
                if (
                    args.pg_filters is None
                    or entry["process_group"][1] in args.pg_filters
                    or entry["process_group"][0] in args.pg_filters
                ):
                    row.append(str(Op(entry, _memberships, pg_name)))
                else:
                    row.append("")
                progress = True
        if progress:
            rows.append(row)

    logger.info(tabulate(rows, headers=headers))


def check_no_missing_dump_files(entries: dict[int, Any], memberships: list[Membership]) -> None:
    all_ranks = {int(m.global_rank) for m in memberships}

    dumps_ranks = {int(key) for key in entries.keys()}
    missing_ranks = all_ranks - dumps_ranks
    if missing_ranks:
        raise ValueError(
            f"Missing dump files for {len(missing_ranks)} ranks: {sorted(missing_ranks)}\n"
            f"Expected ranks: {sorted(all_ranks)}\n"
            f"Found dumps for: {sorted(dumps_ranks)}"
        )


def check_version(version_by_ranks: dict[str, str], expected_version: str) -> None:
    for rank, actual_version in version_by_ranks.items():
        if actual_version != expected_version:
            raise ValueError(f"Version mismatch at rank {rank}: " f"expected {expected_version}, got {actual_version}")


def get_version_detail(version_str: str) -> tuple[int, int]:
    parts = version_str.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid version format: expected 'X.Y', got '{version_str}'")

    try:
        major, minor = int(parts[0]), int(parts[1])
    except ValueError as e:
        raise ValueError(f"Version components must be integers: '{version_str}'") from e

    return major, minor


def align_trace_from_beginning(
    entries: dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    """
    Align the trace entries by record ID for entries.
    This function takes a dictionary of rank names to lists of trace entries as input.
    Each trace entry is a dictionary containing information about a collective operation,
    including its unique identifier (`record_id` is monotonically increasing as we write into the ring buffer).
    The function finds the largest starting point across all ranks by taking the maximum
    `record_id` value of the first entry in each rank. Finally, it filters out any
    entries with `record_id` values less than the maximum starting point.
    The function returns the updated dictionary of sorted and filtered trace entries.

    Args:
        entries (Dict[str, List[Dict[str, Any]]]): A dictionary of rank names to lists of trace entries.

    Returns:
        entries (Dict[str, List[Dict[str, Any]]]): Entries sorted by record ID and filtered by the maximum starting point.
    """

    maximum_starting_record_id = 0
    for rank in entries:
        # Although this is a ring buffer, we already sort the entries by `record_id` when dumping, we just
        # need to find the largest starting point. For example, if the buffer has the following entries:
        # Rank 0: [0, 1, 2, 3, 4, 5, 6]
        # Rank 1: [1, 2, 3, 4, 5, 6, 7]
        # Rank 2: [2, 3, 4, 5, 6, 7, 8]
        # Rank 3: [0, 1, 2, 3, 4, 5, None]
        # Then we should start from collective 2 not 0 because any collective before,
        # we don't have complete records from all ranks so we need to ignore them.
        first_record_id = entries[rank][0]["record_id"]
        maximum_starting_record_id = max(maximum_starting_record_id, first_record_id)

    for rank in entries:
        entries[rank] = [entry for entry in entries[rank] if entry["record_id"] >= maximum_starting_record_id]

    return entries
