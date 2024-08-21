try:
    from urllib.parse import urlparse, urlunparse
except ImportError as e:
    raise ImportError(
        "urllib cannot be found, urlparse from python2 is no longer supported."
    ) from e

import os
import logging
from datetime import timedelta
from typing import Dict, Optional, Union, cast, Tuple
from torch.distributed.rendezvous import register_rendezvous_handler as register_rendezvous_handler
from torch._C._distributed_c10d import _DEFAULT_PG_TIMEOUT
from torch.distributed import Store, PrefixStore
from torch.distributed.elastic.rendezvous.api import RendezvousParameters, RendezvousHandler
from torch.distributed.elastic.rendezvous.api import rendezvous_handler_registry as handler_registry
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint
from torch.distributed.run import parse_args as torch_parse_cmd_args
from torch_npu.distributed import ParallelStore

log = logging.getLogger(__name__)

_default_timeout_seconds = 600

__all__ = []


def _rendezvous_error(msg):
    return ValueError("Error initializing torch_npu.distributed using " + msg)


def _torchelastic_use_agent_store() -> bool:
    return os.environ.get("TORCH_NPU_ELASTIC_USE_AGENT_STORE", None) == str(True)


def _create_c10d_store(hostname, port, rank, world_size, timeout) -> Store:
    """
    Smartly creates a c10d Store object on ``rank`` based on whether
    we need to re-use agent store. The TCPStore server is assumed to be hosted
    on ``hostname:port``.

    If ``torchelastic_use_agent_store()`` is ``True``, then it is assumed that
    the agent leader (node rank 0) hosts the TCPStore server (for which the
    endpoint is specified by the given ``hostname:port``). Hence
    ALL ranks will create and return a ParallelStore client (e.g. ``start_daemon=False``).

    If ``torchelastic_use_agent_store()`` is ``False``, then rank 0 will host
    the TCPStore (with multi-tenancy) and it is assumed that rank 0's hostname
    and port are correctly passed via ``hostname`` and ``port``. All
    non-zero ranks will create and return a ParallelStore client.
    """
    # check if port is uint16_t
    if not 0 <= port < 2**16:
        raise ValueError(f"port must have value from 0 to 65535 but was {port}.")

    if _torchelastic_use_agent_store():
        attempt = os.environ["TORCHELASTIC_RESTART_COUNT"]
        tcp_store = ParallelStore(hostname, port, world_size, False, timeout)
        return PrefixStore(f"/worker/attempt_{attempt}", tcp_store)
    else:
        start_daemon = rank == 0
        return ParallelStore(
            hostname, port, world_size, start_daemon, timeout, multi_tenant=True
        )


def _parallel_rendezvous_handler(
    url: str, timeout: timedelta = _DEFAULT_PG_TIMEOUT, **kwargs
):
    def _error(msg):
        return _rendezvous_error("parallel:// rendezvous: " + msg)

    result = urlparse(url)
    if not result.port:
        raise _error("port number missing")
    query: Dict[str, Union[int, str]]
    # mypy doesn't allow dict() to accept List of values (#257)
    query = dict(pair.split("=") for pair in filter(None, result.query.split("&")))  # type: ignore[misc, arg-type]
    if "rank" not in query:
        raise _error("rank parameter missing")
    if "world_size" not in query:
        raise _error("world size parameter missing")

    rank = int(query["rank"])
    world_size = int(query["world_size"])
    store = _create_c10d_store(result.hostname, result.port, rank, world_size, timeout)

    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform re-rendezvous using parallel:// method")


class _ParallelTCPRendezvous(RendezvousHandler):
    """
    Parallel rendezvous that is a wrapper around the ParallelStore.
    Creates ParallelStore based on the input parameters with the
    listener on the agent with group_rank=0
    """

    def __init__(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        run_id: str,
        timeout: int,
    ):
        self.master_addr = master_addr
        self.master_port = master_port
        self.rank = rank
        self.world_size = world_size
        self.run_id = run_id
        self.timeout = timedelta(seconds=timeout)
        self._store: Optional[Store] = None

    def get_backend(self) -> str:
        return "parallel"

    def next_rendezvous(self) -> Tuple[Store, int, int]:
        log.info("Creating ParallelStore as the c10d::Store implementation")
        if not self._store:
            is_master = self.rank == 0
            self._store = ParallelStore(  # type: ignore[call-arg]
                self.master_addr,
                self.master_port,
                self.world_size,
                is_master,
                self.timeout,
                multi_tenant=True,
            )
        store = PrefixStore(self.run_id, self._store)
        return store, self.rank, self.world_size

    def is_closed(self):
        return False

    def set_closed(self):
        pass

    def num_nodes_waiting(self):
        return 0

    def get_run_id(self) -> str:
        return self.run_id

    def shutdown(self) -> bool:
        return True


def _create_parallel_handler(params: RendezvousParameters) -> RendezvousHandler:
    origin_args = torch_parse_cmd_args(args=None)
    if 'node_rank' not in origin_args:
        raise ValueError(
            "rank is absent in RendezvousParameters."
            "Try add --node_rank to the cmd request"
        )
    params.config["rank"] = origin_args.node_rank

    if 'master_addr' not in origin_args or 'master_port' not in origin_args:
        raise ValueError(
            "endpoint is absent in RendezvousParameters"
            "Try add --master_port and --master_addr to the cmd request"
        )
    params.endpoint = f'{origin_args.master_addr}:{origin_args.master_port}'
    endpoint = params.endpoint.strip()
    master_addr, master_port = parse_rendezvous_endpoint(endpoint, -1)
    if master_port == -1:
        raise ValueError(
            f"Port is absent in endpoint: {endpoint}. Try launching with --master_port"
        )
    world_size = params.max_nodes
    rank = cast(int, params.config.get("rank"))
    run_id = params.run_id
    if "timeout" in params.config:
        timeout = int(params.config["timeout"])
    else:
        timeout = _default_timeout_seconds

    os.environ.setdefault("TORCH_NPU_ELASTIC_USE_AGENT_STORE", str(True))
    return _ParallelTCPRendezvous(
        master_addr, master_port, rank, world_size, run_id, timeout
    )


def _rendezvous_init():
    register_rendezvous_handler("parallel", _parallel_rendezvous_handler)
    handler_registry.register("parallel", _create_parallel_handler)
