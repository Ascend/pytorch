import fcntl
import os
import csv
import logging
from datetime import datetime
import psutil


class AutotuneStatsManager:
    def __init__(self, enabled: bool, log: logging.Logger):
        self.enabled = enabled

        if not enabled:
            return

        self.run_ts = self._get_run_ts()
        self.logs_dir = self._create_logs_dir()

        self.csv_paths = {}
        self.schemas = {}

        self.log = log
        self._setup_logger()

        self.register_csv(
            "duration-stats",
            ["Kernel", "Stage", "Duration(ms)", "Configs",
             "Start TS", "End TS"],
            create_now=True
        )


    def _get_run_ts(self):
        """
        Return a stable timestamp string identifying the current logical run

        Notes
        -----
        This implementation assumes the current torch.async_compile behavior.

        If in the future the first kernel is executed concurrently with other
        kernels, the run identifier should be derived from the grandparent
        process ID (i.e., write grandparent.pid as RUN_PID) to preserve a
        consistent run scope.
        """
        def write_run_info():
            ts = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            with open("/tmp/run_info", "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(f"{proc.pid},{ts}\n")
                fcntl.flock(f, fcntl.LOCK_UN)
            return ts

        proc = psutil.Process(os.getpid())
        grandparent = proc.parent().parent()

        try:

            with open("/tmp/run_info", "r") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                run_pid, ts = f.read().strip().split(",", 1)
                fcntl.flock(f, fcntl.LOCK_UN)
        except FileNotFoundError:
            return write_run_info()

        if not grandparent or grandparent.pid != int(run_pid):
            return write_run_info()

        return ts

    def _create_logs_dir(self):
        path = os.path.join(os.getcwd(), "autotune_logs")
        os.makedirs(path, exist_ok=True)
        return path

    def _setup_logger(self):
        if not self.log.handlers:
            log_file = os.path.join(
                self.logs_dir,
                f"npu_triton_heuristics-{self.run_ts}.log",
            )

            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            ))

            self.log.addHandler(handler)
            self.log.setLevel(logging.INFO)
            print("log file:", log_file)

    def register_csv(self, name, headers, create_now=False):
        self.schemas[name] = headers

        if create_now:
            self._ensure_csv_created(name)

    def _ensure_csv_created(self, name):
        if name in self.csv_paths:
            return self.csv_paths[name]

        path = os.path.join(
            self.logs_dir,
            f"{name}-{self.run_ts}.csv"
        )

        try:
            fd = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644
            )
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w", newline="") as f:
                csv.writer(f).writerow(self.schemas[name])

        self.csv_paths[name] = path
        return path

    def write(self, name, row):
        if not self.enabled:
            return

        path = self._ensure_csv_created(name)

        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)
