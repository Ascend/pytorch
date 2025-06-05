from collections.abc import Sequence
from typing import Optional
import pickle

from tools.flight_recorder.components.builder import build_db
from tools.flight_recorder.components.config_manager import JobConfig
from tools.flight_recorder.components.loader import read_dir
from tools.flight_recorder.components.types import types
from tools.flight_recorder.components.utils import get_valid_read_path, get_valid_write_path


def main(args: Optional[Sequence[str]] = None) -> None:
    config = JobConfig()
    args = config.parse_args(args)
    get_valid_read_path(args.trace_dir, is_dir=True)

    details, version = read_dir(args)
    db = build_db(details, args, version)
    if args.output:
        args.output = get_valid_write_path(args.output)
        with open(args.output, "wb") as f:
            pickle.dump((types, db), f)


if __name__ == "__main__":
    main()
