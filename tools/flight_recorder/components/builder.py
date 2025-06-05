import argparse
import ast
import os
from typing import Any

from tools.flight_recorder.components.fr_logger import FlightRecorderLogger
from tools.flight_recorder.components.types import (
    Collective,
    Database,
    EntryState,
    Group,
    MatchStateRecord,
    Membership,
    HCCLCall,
    Op,
    Traceback,
)
from tools.flight_recorder.components.utils import (
    ProcessGroupData,
    align_trace_from_beginning,
    check_current_entry_match,
    check_no_missing_dump_files,
    check_version,
    EntryContext,
    error_analysis,
    get_version_detail,
    just_print_entries,
)


# Set up logging
logger: FlightRecorderLogger = FlightRecorderLogger()


try:
    from tabulate import tabulate
except ModuleNotFoundError:
    logger.warning("tabulate is not installed. Proceeding without it.")

    # Define a no-op tabulate function
    def tabulate(data: Any, headers: Any = None) -> Any:  # type: ignore[misc]
        return data


"""
Flat DB builder
"""


def build_groups_memberships(
    pg_config: Any,
) -> tuple[
    list[Group],
    dict[Any, Group],
    list[Membership],
    dict[str, set[Any]],
    dict[tuple[str, int], str],
]:
    """
    pg_config: {
        global_rank: {
            (pg_guid, desc, ranks)
        }
    }

    `pg_guid` is a system generated id, but depending on the mode of PG creation it could be a globally incrementing int
          or a hash of the ranks.  See `_process_group_name` in distributed_c10d.py.
    `desc` is provided by the user (optionally) and should be 'meaningful' (e.g. TP/PP/DP group)
    `ranks` is a list of the 'global ranks' that are members of the PG.

    (pg_guid, desc, ranks) tuples are appended lazily to the flight buffer when `getHCCLComm` is called on a PG and
    the `enabled_` flag is true for that PG.
        - the order of calling (init_process_group, new_group, etc) does not affect the order of the tuples in the list

    Returns:
        `groups`: a groups table where each row is a Group namedtuple.
        `_groups`: a dict that is indexed by pg_guid with Group namedtuple as value.
        `memberships`: a membership table where each row is a Membership namedtuple.
        `_memberships`: a dict that is indexed by pg_guid with set of ranks (int) as value.
        `_pg_guids`: a dict that is indexed by (pg_uid, global_rank) with pg_guid as value.
    """
    # flat lists for return
    groups = []
    memberships = []

    # dicts for faster cross-rank validation
    _groups = {}
    _memberships = {}
    _pg_guids = {}
    for global_rank in pg_config:
        for pg_uid in pg_config[global_rank]:
            desc = pg_config[global_rank][pg_uid]["desc"]
            ranks = ast.literal_eval(pg_config[global_rank][pg_uid]["ranks"])
            # With the adoption of the split_group API, we can have multiple PGs with the same pg_guid (PG Name)
            # So we need to add the hash of all its ranks within the PG as well.
            # Also guid must be a string because `_process_group_name` returns a string.
            pg_guid = pg_uid + str(hash(frozenset(ranks)))
            _pg_guids[(pg_uid, global_rank)] = pg_guid
            if isinstance(ranks, str):
                ranks = ast.literal_eval(ranks)
            if pg_guid not in _groups:
                groups.append(Group(id=pg_guid, desc=desc, size=len(ranks)))
                for rank in ranks:
                    memberships.append(Membership(group_id=pg_guid, global_rank=rank))
                _groups[pg_guid] = groups[-1]
                _memberships[pg_guid] = set(ranks)
            else:
                # validation across ranks
                if _groups[pg_guid].desc != desc:
                    raise ValueError(
                        f"Description mismatch for group {pg_guid}: "
                        f"expected '{desc}', got '{_groups[pg_guid].desc}'"
                    )

                if _memberships[pg_guid] != set(ranks):
                    raise ValueError(
                        f"Membership mismatch for group {pg_guid}: "
                        f"expected {set(ranks)}, got {_memberships[pg_guid]}"
                    )

    return groups, _groups, memberships, _memberships, _pg_guids


def build_collectives(
    all_entries: dict[int, list[dict[str, Any]]],
    _groups: dict[str, Group],
    _memberships: dict[str, set[Any]],
    _pg_guids: dict[tuple[str, int], str],
    version: str,
) -> tuple[list[Traceback], list[Collective], list[HCCLCall]]:
    """
    groups, memberships are the non-flat dicts that are indexable
    all_entries is a raw dict from the original dumps:

    all_entries: {
        global_rank: [
            {
                record_id: ordered id of the event in the trace buffer
                pg_id: ProcessGroupHCCL::uid_
                    *note: `pg_id` corresponds to nothing in groups table
                process_group: (pg_name, desc)
                    *note: `pg_name`, `desc` corresponds to `pg_id`, `desc` in groups table
                collective_seq_id: ordered id for collective operations and coalesced group operations
                p2p_seq_id: ordered id for point-to-point operations
                op_id: ordered id including individual ops inside coalescing group
                profiling_name: descriptive name of the operation
                'time_created_ns',
                'input_sizes',
                'output_sizes',
                'state',
                'time_discovered_started_ns',
                'time_discovered_completed_ns',
                'retired',
                'frames',
            }
        ]
    }
    """
    tracebacks: list[Traceback] = []

    collectives: list[Collective] = []
    hccl_calls: list[HCCLCall] = []

    # once we find one mismatch, we stop pairing up collectives since the pairing is possibly incorrect
    # instead, just record the remaining ops as HCCLCalls
    mismatch = {_groups[g].id: 0 for g in _groups}
    MISMATCH_TAIL = 10

    # For best effort partial analysis.
    dumps_ranks = set()
    for key in all_entries.keys():
        try:
            dumps_ranks.add(int(key))
        except ValueError as e:
            raise ValueError(f"Cannot extract rank from '{key}") from e
    """
    - it doesn't matter what order I put collectives/hcclops into their table. we can later on re-sort it by start time
    - there could be multiple options for the "first" collective to pair up (rank 0,1 might do a bcast while rank 2,3 do a bcast)
    - within a group, the first collective must be the same on all ranks in the group, then it can be marked as a
    collective and removed
    """
    while all_entries:
        # we greedily match collectives, starting arbitrarily with the trace from the first rank
        # later, if we exhaust the first rank, we continue with the next 'first rank'
        rank_iter = iter(all_entries)
        first_rank = next(rank_iter)
        other_ranks = list(rank_iter)

        if len(all_entries[first_rank]) == 0:
            all_entries.pop(first_rank)
            continue

        # lets match the first collective! we need to know which ranks are involved, and ensure that this same
        # collective is also the first one on those ranks within that group
        entries = all_entries[first_rank]
        current_entry = entries[0]

        desc = current_entry["process_group"][1] if current_entry["process_group"][1] else "default_pg"
        # For db build and logs printing, we want to use the original pg_name, not the hash one.
        original_pg_name = current_entry["process_group"][0]
        pg_name = _pg_guids[(original_pg_name, first_rank)]
        expected_ranks = set(_memberships[pg_name])
        entry_state = EntryState(current_entry, expected_ranks)
        match_record = MatchStateRecord(
            expected_ranks=expected_ranks,
            other_ranks=other_ranks,
            entry_state=entry_state,
            candidate_ranks={first_rank},
            candidate_idx={},
            found_ranks=set(),
            found_idx={},
            errors=set(),
        )

        check_current_entry_match(
            all_entries=all_entries,
            current_entry=current_entry,
            _memberships=_memberships,
            pg_data=ProcessGroupData(pg_guids=_pg_guids, pg_name=pg_name, desc=desc, mismatch=mismatch),
            match_record=match_record,
        )

        # Use heuristics to decide what type of errors and error messages we should print.
        error_analysis(
            entry_context=EntryContext(all_entries, current_entry, dumps_ranks, first_rank),
            match_record=match_record,
            mismatch=mismatch,
            version=get_version_detail(version),
            pg_name=pg_name,
        )
        # at this point there are 3 possibilities
        # 1. we found a match on all the ranks that are members of the group
        #  -> we create a Collective and remove the individual entries from their original lists
        if match_record.found_ranks == expected_ranks and mismatch[pg_name] == 0:
            collectives.append(match_record.entry_state.to_collective(len(collectives)))
            idx_map = {r: match_record.found_idx[r] if r != first_rank else 0 for r in match_record.found_ranks}
            hccl_calls.extend(
                match_record.entry_state.to_hccl_call(all_entries, idx_map, len(hccl_calls), collectives[-1].id)
            )

        # 2. we found a partial match but some ranks are missing
        # 3. we found no match
        else:
            logger.debug("appending a non-matching collective")
            idx_map = {r: match_record.candidate_idx[r] if r != first_rank else 0 for r in match_record.candidate_ranks}
            collectives.append(
                match_record.entry_state.to_collective(
                    len(collectives),
                    errors=match_record.errors,
                    idx_map=idx_map,
                    all_entries=all_entries,
                )
            )
            hccl_calls.extend(match_record.entry_state.to_hccl_call(all_entries, idx_map, len(hccl_calls), None))

        if mismatch[pg_name] > MISMATCH_TAIL:
            logger.error("Too many mismatches for process_group %s: %s aborting", pg_name, desc)
            break
    return tracebacks, collectives, hccl_calls


def build_db(details: dict[str, dict[str, Any]], args: argparse.Namespace, version: str) -> Database:
    if args.verbose:
        os.environ["FR_TRACE_VERBOSE_OUTPUT"] = "1"
    # temporary state used for building database
    entries = {}
    pg_config = {}
    version_by_ranks = {}
    for rank, dump in details.items():
        entries[rank] = dump["entries"]
        version_by_ranks[rank] = dump["version"]
        pg_config[rank] = dump["pg_config"]

    # Ensure version is consistent across all ranks.
    check_version(version_by_ranks, version)
    entries = align_trace_from_beginning(entries)

    # flattened database
    groups, _groups, memberships, _memberships, _pg_guids = build_groups_memberships(pg_config)
    logger.debug("built groups, memberships")

    if not args.allow_incomplete_ranks:
        check_no_missing_dump_files(entries, memberships)

    if args.just_print_entries:
        just_print_entries(entries, _groups, _memberships, _pg_guids, args)
        return None

    tracebacks, collectives, hccl_calls = build_collectives(entries, _groups, _memberships, _pg_guids, version)
    logger.debug("built collectives, hccl_calls")
    if args.verbose:
        logger.debug("Groups")
        logger.debug(tabulate(groups, headers=Group._fields))
        logger.debug("Memberships")
        logger.debug(tabulate(memberships, headers=Membership._fields))
        logger.debug("Collectives")
        logger.debug(tabulate(collectives, headers=Collective._fields))
        logger.debug("HCCLCalls")
        logger.debug(tabulate(hccl_calls, headers=HCCLCall._fields))
    db = Database(
        tracebacks=tracebacks,
        collectives=collectives,
        hcclcalls=hccl_calls,
        groups=groups,
        memberships=memberships,
    )
    return db
