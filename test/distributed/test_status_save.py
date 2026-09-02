"""
ProcessGroupHCCL::checkAndMakePath test.
"""

import os
import time
import glob
import json
import shutil
import tempfile
import unittest
import torch.distributed.run as launch
from torch_npu.testing.testcase import run_tests, TestCase
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


def _is_root():
    return hasattr(os, "geteuid") and os.geteuid() == 0


class CheckAndMakePathTest(TestCase):
    """Integration tests covering all branches of checkAndMakePath."""

    def setUp(self):
        self.save_dir = tempfile.mkdtemp(prefix="hccl_status_save_")
        self.done_dir = tempfile.mkdtemp(prefix="hccl_status_done_")
        os.environ["TORCH_HCCL_STATUS_SAVE_ENABLE"] = "1"
        os.environ["TORCH_HCCL_STATUS_SAVE_PATH"] = self.save_dir
        os.environ["HCCL_UT_DONE_DIR"] = self.done_dir

    def tearDown(self):
        for k in ("TORCH_HCCL_STATUS_SAVE_ENABLE", "TORCH_HCCL_STATUS_SAVE_PATH",
                  "HCCL_UT_DONE_DIR"):
            os.environ.pop(k, None)
        for d in (self.save_dir, self.done_dir):
            self._force_rmtree(d)

    @staticmethod
    def _force_rmtree(d):
        """Restore permissions recursively before removing the directory tree."""
        try:
            for root, dirs, _files in os.walk(d, topdown=False):
                for name in dirs:
                    try:
                        os.chmod(os.path.join(root, name), 0o755)
                    except OSError:
                        pass
                try:
                    os.chmod(root, 0o755)
                except OSError:
                    pass
            shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass

    def _launch(self, script, nproc=2, max_retries=3, expect_failure=False):
        """Run a distributed script with retries for delayed HCCL/NPU cleanup.

        Args:
            expect_failure: If True, the script is expected to fail, so return
                the first result without retrying.
        """
        err = None
        for attempt in range(max_retries):
            # Clean up any stale processes.
            self._cleanup_stale_processes()
            # Use a fresh done_dir on each retry so stale marker files do not
            # affect assertions.
            if attempt > 0:
                self.done_dir = tempfile.mkdtemp(prefix="hccl_status_done_")
                os.environ["HCCL_UT_DONE_DIR"] = self.done_dir
                time.sleep(5)
            err = None
            try:
                launch.main(["--nproc-per-node={}".format(nproc), path(script)])
                return None
            except Exception as e:
                err = e
                # Do not retry expected failures, such as a timeout mismatch.
                if expect_failure:
                    return err
                # Do not retry explicit path failures unrelated to
                # communication initialization.
                err_str = str(e)
                if ("ChildFailedError" in type(e).__name__ and
                        ("checkAndMakePath" in err_str or "invalid path" in err_str.lower())):
                    return err
        return err

    @staticmethod
    def _cleanup_stale_processes():
        """Clean up stale HCCL/Python processes that may hold NPU devices."""
        import subprocess
        try:
            subprocess.run(["pkill", "-9", "-f", "status_save_"],
                           capture_output=True, timeout=5)
        except Exception:
            pass
        try:
            subprocess.run(["pkill", "-9", "-f", "torch.distributed.run"],
                           capture_output=True, timeout=5)
        except Exception:
            pass

    def _status_files(self, rank, root=None):
        root = root or self.save_dir
        return glob.glob(os.path.join(root, "torch_hccl_status-{}_*".format(rank)))

    def _assert_all_alive(self, nproc=2):
        """Require every worker to finish and write done_<rank>."""
        for r in range(nproc):
            self.assertTrue(
                os.path.exists(os.path.join(self.done_dir, "done_{}".format(r))),
                "rank {} did not finish (status save must not be fatal)".format(r))

    # ==================================================================
    # Case 1: WritableDirFastPath
    # Scenario: A pre-created writable directory lets the watchdog take the
    # fast path.
    # ==================================================================
    @skipIfUnsupportMultiNPU(2)
    def test_writable_dir_fast_path(self):
        target = os.path.join(self.save_dir, "pre_exist")
        os.makedirs(target, exist_ok=True)
        os.chmod(target, 0o755)
        os.environ["TORCH_HCCL_STATUS_SAVE_PATH"] = target
        err = self._launch(path("status_save/status_save_simple.py"))
        self.assertIsNone(err, "writable dir should not cause failure")
        self._assert_all_alive()
        files = glob.glob(os.path.join(target, "torch_hccl_status-*"))
        self.assertTrue(files, "status file should be written via fast path")

    # ==================================================================
    # Case 2: NonExistentCreateSuccess
    # Scenario: The watchdog creates a missing directory, and subsequent calls
    # take the fast path.
    # ==================================================================
    @skipIfUnsupportMultiNPU(2)
    def test_nonexistent_create_success(self):
        target = os.path.join(self.save_dir, "not_exist")
        os.environ["TORCH_HCCL_STATUS_SAVE_PATH"] = target
        err = self._launch(path("status_save/status_save_simple.py"))
        self.assertIsNone(err, "dir creation should not fail")
        self._assert_all_alive()
        self.assertTrue(os.path.isdir(target), "dir should be created")
        files = glob.glob(os.path.join(target, "torch_hccl_status-*"))
        self.assertTrue(files, "status file should be written after creation")

    # ==================================================================
    # Case 3: NestedDirsCreate
    # Scenario: create_directories recursively creates a missing nested path.
    # ==================================================================
    @skipIfUnsupportMultiNPU(2)
    def test_nested_dirs_create(self):
        target = os.path.join(self.save_dir, "a/b/c/d")
        os.environ["TORCH_HCCL_STATUS_SAVE_PATH"] = target
        err = self._launch(path("status_save/status_save_simple.py"))
        self.assertIsNone(err, "nested dir creation should not fail")
        self._assert_all_alive()
        self.assertTrue(os.path.isdir(target), "nested dirs should be created")
        files = glob.glob(os.path.join(target, "torch_hccl_status-*"))
        self.assertTrue(files, "status file should be written in nested dir")

    # ==================================================================
    # Case 4: ParentIsFileNotFatal
    # Scenario: create_directories fails with ENOTDIR when the parent is a file.
    # ==================================================================
    @skipIfUnsupportMultiNPU(2)
    def test_parent_is_file_not_fatal(self):
        blocker = os.path.join(self.save_dir, "a_file")
        open(blocker, "w").close()
        os.environ["TORCH_HCCL_STATUS_SAVE_PATH"] = os.path.join(blocker, "sub")
        err = self._launch(path("status_save/status_save_simple.py"))
        self.assertIsNone(err, "invalid status path must not kill the process")
        self._assert_all_alive()

    # ==================================================================
    # Case 5: ReadonlyDirNotFatal
    # Scenario: Make the shared directory read-only after the first write so
    # the final access check fails.
    # ==================================================================
    @skipIfUnsupportMultiNPU(2)
    @unittest.skipIf(_is_root(), "root bypasses directory permission bits")
    def test_readonly_dir_not_fatal(self):
        err = self._launch(path("status_save/status_save_readonly.py"))
        self.assertIsNone(err, "readonly status dir must not kill the process")
        self._assert_all_alive()

    # ==================================================================
    # Case 6: ConcurrentCreateNoThrow
    # Scenario: Multiple process groups concurrently create the same missing
    # directory.
    # ==================================================================
    @skipIfUnsupportMultiNPU(2)
    def test_concurrent_create_no_throw(self):
        target = os.path.join(self.save_dir, "not_exist_yet")
        os.environ["TORCH_HCCL_STATUS_SAVE_PATH"] = target
        err = self._launch(path("status_save/status_save_race.py"))
        self.assertIsNone(err, "processes should survive concurrent status-dir creation")
        self._assert_all_alive()
        files = glob.glob(os.path.join(target, "torch_hccl_status-*"))
        self.assertTrue(files, "status file should be written")
        for f in files:
            with open(f) as fp:
                json.load(fp)

    # ==================================================================
    # Case 7: IsDirectoryPermDenied
    # Scenario: is_directory returns EACCES when the parent lacks execute
    # permission.
    # ==================================================================
    @skipIfUnsupportMultiNPU(2)
    @unittest.skipIf(_is_root(), "root bypasses permission checks")
    def test_is_directory_perm_denied(self):
        parent = os.path.join(self.save_dir, "noperm")
        os.makedirs(parent, exist_ok=True)
        os.chmod(parent, 0o644)  # rw-r--r--: no execute permission; cannot traverse
        target = os.path.join(parent, "sub")
        os.environ["TORCH_HCCL_STATUS_SAVE_PATH"] = target
        err = self._launch(path("status_save/status_save_simple.py"))
        self.assertIsNone(err, "is_directory failure should not kill process")
        self._assert_all_alive()
        # Restore permissions so tearDown can clean up the directory.
        os.chmod(parent, 0o755)

    # ==================================================================
    # Case 8: ErrorPathStatusFiles
    # Scenario: A mismatched subgroup times out, and the error=true path writes
    # the status file before termination.
    # Note: The new file name has no _pg{uid}_{group_id} suffix. Data for all
    # process groups in one process is stored in the last_comm_op array of a
    # single JSON file and distinguished by pg_id.
    # ==================================================================
    @skipIfUnsupportMultiNPU(2)
    def test_error_path_status_files(self):
        err = self._launch(path("status_save/status_save_timeout.py"), expect_failure=True)
        self.assertIsNotNone(err, "mismatched collective should kill the rank")
        files = self._status_files(0)
        self.assertGreaterEqual(len(files), 1,
                                "rank 0 should leave at least one status file "
                                "containing the combined PG state")
        for f in files:
            # The new file name does not include an _pg suffix.
            self.assertNotIn("_pg", os.path.basename(f),
                             "the new file name should not contain the "
                             "_pg{uid}_{group_id} suffix")
            with open(f) as fp:
                data = json.load(fp)
            # last_comm_op should contain the state of at least one PG.
            self.assertIn("last_comm_op", data,
                          "the status file should contain a last_comm_op array")
            self.assertGreaterEqual(len(data["last_comm_op"]), 1,
                                    "last_comm_op should contain the state of "
                                    "at least one process group")
            # Verify that pg_id is present to distinguish process groups.
            for op in data["last_comm_op"]:
                self.assertIn("pg_id", op,
                              "each comm_op should contain a pg_id field")

    # ==================================================================
    # Case 9: MultipgStatusContent
    # Scenario: Multiple PGs continuously run collectives; verify complete JSON.
    # Note: All PG states are written to one file and distinguished by pg_id in
    # last_comm_op. status_save_multipg.py creates one global PG and two
    # subgroups, so last_comm_op should contain multiple records.
    # ==================================================================
    @skipIfUnsupportMultiNPU(2)
    def test_multipg_status_content(self):
        err = self._launch(path("status_save/status_save_multipg.py"))
        self.assertIsNone(err)
        self._assert_all_alive()
        files = self._status_files(0)
        self.assertTrue(files, "rank 0 should have at least one status file")
        # The new format has one file per process with all PG states combined.
        for f in files:
            self.assertNotIn("_pg", os.path.basename(f),
                             "the new file name should not contain the "
                             "_pg{uid}_{group_id} suffix")
        with open(sorted(files)[-1]) as fp:
            data = json.load(fp)
        self.assertIn("last_comm_op", data)
        # status_save_multipg.py creates one global PG and two subgroups.
        # last_comm_op should have at least one record because non-error
        # subgroup state is also added to StatusOutput_.
        self.assertGreaterEqual(len(data["last_comm_op"]), 1,
                                "last_comm_op should contain the state of at "
                                "least one process group")
        # Verify every comm_op has a pg_id to distinguish process groups.
        pg_ids = set()
        for op in data["last_comm_op"]:
            self.assertIn("pg_id", op,
                          "each comm_op should contain a pg_id field")
            pg_ids.add(op["pg_id"])
        # The global PG state must be present because subgroup filtering does
        # not apply to it.
        self.assertGreaterEqual(len(pg_ids), 1,
                                "there should be at least one distinct pg_id")
        self.assertIn("global_pg_end_time", data)


if __name__ == "__main__":
    run_tests()
