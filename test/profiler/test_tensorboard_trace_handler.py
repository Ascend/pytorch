import os
import tarfile
import tempfile
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest import mock

from torch_npu.profiler.profiler import tensorboard_trace_handler


class TestTensorboardTraceHandler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.prof_path = os.path.join(
            self.temp_dir.name, "msprof_1784298_20260114085947065_ascend_pt"
        )
        os.mkdir(self.prof_path)
        framework_path = os.path.join(self.prof_path, "FRAMEWORK")
        output_path = os.path.join(self.prof_path, "ASCEND_PROFILER_OUTPUT")
        os.mkdir(framework_path)
        os.mkdir(output_path)
        with open(os.path.join(framework_path, "raw.bin"), "wb") as file:
            file.write(b"raw")
        with open(os.path.join(output_path, "trace_view.json"), "w", encoding="utf-8") as file:
            file.write("{}")
        self.prof_if = SimpleNamespace(prof_path=self.prof_path, analyse=mock.Mock())
        self.prof_inst = SimpleNamespace(prof_if=self.prof_if)

    def test_default_does_not_create_archive(self):
        tensorboard_trace_handler(self.temp_dir.name)(self.prof_inst)

        self.prof_if.analyse.assert_called_once_with(async_mode=False)
        self.assertFalse(os.path.exists(f"{self.prof_path}.tar.gz"))
        self.assertTrue(os.path.isdir(self.prof_path))

    def test_use_gzip_archives_complete_directory_and_removes_source(self):
        def finish_analysis(**_kwargs):
            with open(os.path.join(self.prof_path, "analyse.done"), "w", encoding="utf-8"):
                pass

        self.prof_if.analyse.side_effect = finish_analysis
        tensorboard_trace_handler(self.temp_dir.name, use_gzip=True)(self.prof_inst)

        with tarfile.open(f"{self.prof_path}.tar.gz", "r:gz") as archive:
            names = archive.getnames()
            self.assertIn(os.path.basename(self.prof_path), names)
            self.assertIn(f"{os.path.basename(self.prof_path)}/analyse.done", names)
            self.assertIn(
                f"{os.path.basename(self.prof_path)}/ASCEND_PROFILER_OUTPUT/trace_view.json",
                names,
            )
            self.assertIn(f"{os.path.basename(self.prof_path)}/FRAMEWORK/raw.bin", names)
        self.assertFalse(os.path.exists(self.prof_path))

    def test_async_mode_archives_after_all_analysis(self):
        first_future = Future()
        second_future = Future()
        self.prof_if.analyse.return_value = [first_future, second_future]
        tensorboard_trace_handler(
            self.temp_dir.name, async_mode=True, use_gzip=True
        )(self.prof_inst)

        archive_path = f"{self.prof_path}.tar.gz"
        self.assertFalse(os.path.exists(archive_path))
        with open(os.path.join(self.prof_path, "analyse.done"), "w", encoding="utf-8"):
            pass
        first_future.set_result(None)
        self.assertFalse(os.path.exists(archive_path))
        self.assertTrue(os.path.isdir(self.prof_path))
        second_future.set_result(None)
        with tarfile.open(archive_path, "r:gz") as archive:
            self.assertIn(f"{os.path.basename(self.prof_path)}/analyse.done", archive.getnames())
        self.assertFalse(os.path.exists(self.prof_path))

    def test_compression_failure_preserves_source(self):
        with mock.patch(
            "torch_npu.profiler.profiler.FileManager.compress_to_tar_gz",
            side_effect=RuntimeError("compression failed"),
        ):
            tensorboard_trace_handler(self.temp_dir.name, use_gzip=True)(self.prof_inst)

        self.assertTrue(os.path.isdir(self.prof_path))
        self.assertFalse(os.path.exists(f"{self.prof_path}.tar.gz"))

    def test_async_analysis_failure_preserves_source(self):
        future = Future()
        self.prof_if.analyse.return_value = [future]
        tensorboard_trace_handler(
            self.temp_dir.name, async_mode=True, use_gzip=True
        )(self.prof_inst)

        future.set_exception(RuntimeError("analysis failed"))
        self.assertTrue(os.path.isdir(self.prof_path))
        self.assertFalse(os.path.exists(f"{self.prof_path}.tar.gz"))


if __name__ == "__main__":
    unittest.main()
