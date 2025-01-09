from typing import List
import json
import time
from collections import defaultdict
from contextlib import contextmanager
import torch
from torch.distributed import run as torch_run
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.elastic.agent.server.api import (
    WorkerSpec,
    Worker,
    _RoleInstanceInfo,
)
from torch.distributed.elastic.utils.store import timedelta
from torch.distributed.elastic.multiprocessing.api import SignalException
import torch_npu

__all__ = []

logger = get_logger(__name__)
_NUM_MEMBERS = "/num_members"
_LAST_MEMBER_CHECKIN = "/last_member"
_TERMINAL_STATE_SYNC_ID = "torchelastic/agent/terminal_state"


def _assign_worker_ranks_new(
    self, store, group_rank: int, group_world_size: int, spec: WorkerSpec
) -> List[Worker]:
    """Determine proper ranks for worker processes.

    The rank assignment is done according to the following algorithm:

    1. Each agent writes its configuration(group_rank, group_world_size
        , num_workers) to the common store.
    2. The rank 0 agent reads all the role_info from the store and
        determines each agents worker ranks.
    3. Determine the global rank: the global rank of the workers is computed
        by cumulative sum of the local_world_size for all workers in front of it.
        For efficiency reasons each worker is assigned a base global rank
        such that it's workers are in the range [base_global_rank,
        base_global_rank + local_world_size).
    4. Determine the role rank: The role rank is determined using the algorithms
        in the point 3 with the exception that the ranks are calculated with
        respect to the role name.
    5. The rank 0 agent writes the assigned ranks to the store.
    6. Each agent reads the assigned ranks from the store.

    Time complexity: each worker O(1), rank0 O(n), overall O(n)
    """

    ROLE_INFO_PREFIX = "torchelastic/role_info/"
    ASSIGNED_RANKS_PREFIX = "torchelastic/assigned_ranks/"

    agent_role_info = _RoleInstanceInfo(
        spec.role, group_rank, spec.local_world_size
    )
    store.set(f"{ROLE_INFO_PREFIX}{group_rank}", agent_role_info.serialize())

    # tcp store is collocated with rank 0 so we can use it to do extra compute to reduce overall # of operations.
    if group_rank == 0:
        role_infos_bytes = store.multi_get(
            [f"torchelastic/role_info/{i}" for i in range(group_world_size)]
        )
        role_infos = [
            _RoleInstanceInfo.deserialize(info_bytes)
            for info_bytes in role_infos_bytes
        ]

        role_sizes = defaultdict(lambda: 0)
        global_size = 0
        for role_info in role_infos:
            role_sizes[role_info.role] += role_info.local_world_size
            global_size += role_info.local_world_size

        base_global_rank = 0
        role_ranks = defaultdict(lambda: 0)

        keys = []
        values = []
        for i, role_info in enumerate(role_infos):
            keys.append(f"{ASSIGNED_RANKS_PREFIX}{i}")
            values.append(
                json.dumps(
                    [
                        base_global_rank,
                        global_size,
                        role_ranks[role_info.role],
                        role_sizes[role_info.role],
                    ]
                )
            )

            base_global_rank += role_info.local_world_size
            role_ranks[role_info.role] += role_info.local_world_size

        store.multi_set(keys, values)

    # get will block until the data is available in the store.
    base_global_rank, global_world_size, base_role_rank, role_world_size = json.loads(
        store.get(f"{ASSIGNED_RANKS_PREFIX}{group_rank}")
    )

    workers = []
    for local_rank in range(spec.local_world_size):
        worker = Worker(
            local_rank=local_rank,
            global_rank=base_global_rank + local_rank,
            role_rank=base_role_rank + local_rank,
            world_size=global_world_size,
            role_world_size=role_world_size,
        )
        workers.append(worker)
    return workers


@contextmanager
def _store_timeout(store, timeout: float):
    """
    This sets the timeout and then restores the old timeout when the context
    manager exits.

    Args:
        store: the store to set the timeout on
        timeout: the timeout to set
    """

    old_timeout = store.timeout
    store.set_timeout(timedelta(seconds=timeout))
    yield
    store.set_timeout(old_timeout)


def _barrier_nonblocking(store, world_size: int, key_prefix: str) -> str:
    """
    Does all the non-blocking operations for a barrier and returns the final key
    that can be waited on.
    """
    num_members_key = key_prefix + _NUM_MEMBERS
    last_member_key = key_prefix + _LAST_MEMBER_CHECKIN

    idx = store.add(num_members_key, 1)
    if idx == world_size:
        store.set(last_member_key, "<val_ignored>")

    return last_member_key


def _barrier(
    store, world_size: int, key_prefix: str, barrier_timeout: float = 300
) -> None:
    """
    A global lock between agents. This will pause all workers until at least
    ``world_size`` workers respond.

    This uses a fast incrementing index to assign waiting ranks and a success
    flag set by the last worker.

    Time complexity: O(1) per worker, O(N) globally.

    Note: Since the data is not removed from the store, the barrier can be used
        once per unique ``key_prefix``.
    """

    with _store_timeout(store, barrier_timeout):
        last_member_key = _barrier_nonblocking(store=store, world_size=world_size, key_prefix=key_prefix)
        store.get(last_member_key)


def _exit_barrier_new(self):
    """
    Define a barrier that keeps the agent process alive until all workers finish.

    Wait for ``exit_barrier_timeout`` seconds for all agents to finish
    executing their local workers (either successfully or not). This
    acts as a safety guard against user scripts that terminate at different
    times.
    """
    logger.info(
        "Local worker group finished (%s). "
        "Waiting %s seconds for other agents to finish",
        self._worker_group.state, self._exit_barrier_timeout
    )
    start = time.time()
    try:
        _barrier(
            store=self._store,
            world_size=self._worker_group.group_world_size,
            key_prefix=_TERMINAL_STATE_SYNC_ID,
            barrier_timeout=self._exit_barrier_timeout,
        )
        logger.info(
            "Done waiting for other agents. Elapsed: %s seconds", time.time() - start
        )
    except SignalException as e:
        logger.warning("Got termination signal: %s", e.sigval)
        raise
    except Exception:
        logger.exception(
            "Error waiting on exit barrier. Elapsed: %s seconds",
            time.time() - start
        )


def _apply_torch_npu_run_patch():
    torch.distributed.elastic.agent.server.api.SimpleElasticAgent._assign_worker_ranks = _assign_worker_ranks_new
    torch.distributed.elastic.agent.server.api.SimpleElasticAgent._exit_barrier = _exit_barrier_new


@record
def _main(args=None):
    _apply_torch_npu_run_patch()
    args = torch_run.parse_args(args)
    args.rdzv_backend = 'parallel'
    torch_run.run(args)


if __name__ == "__main__":
    _main()
