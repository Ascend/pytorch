from torch.distributed import run as torch_run
from torch.distributed.elastic.multiprocessing.errors import record
import torch_npu

__all__ = []


@record
def _main(args=None):
    args = torch_run.parse_args(args)
    args.rdzv_backend = 'parallel'
    torch_run.run(args)


if __name__ == "__main__":
    _main()
