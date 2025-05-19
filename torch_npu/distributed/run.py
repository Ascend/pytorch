__all__ = ["parse_args"]

from torch.distributed import run as torch_run
from torch.distributed.argparse_util import check_env, env
from torch.distributed.run import get_args_parser
from torch.distributed.elastic.multiprocessing.errors import record
import torch_npu


def parse_args(args):
    parser = get_args_parser()
    parser.add_argument(
        "--enable_tiered_parallel_tcpstore",
        "--enable_tiered_parallel_tcpstore",
        action=env,
        type=str,
        default="false",
        help="Turn parallel tcpstore tiered optimization, if true, The agent adds a proxy role," 
        "the worker on this node will connect to the server through the proxy.",
    )
    return parser.parse_args(args)


@record
def _main(args=None):
    args = parse_args(args)
    args.rdzv_backend = 'parallel'
    if not args.rdzv_endpoint:
        args.rdzv_endpoint = f"{args.master_addr}:{args.master_port}"
    torch_run.run(args)


if __name__ == "__main__":
    _main()
