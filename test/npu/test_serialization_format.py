import os
import tempfile

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu


# Due to compilation caching, we need to start a new process to load tensor,
# otherwise the cache will be reused without any errors.
# Because torch_npu.testing.testcase.TestCase will set device first and we set device
# in main process then subprocess will raise error, we need a new file without set device
# to test this case.


FORMAT_INFO = {
    "NCHW": 0,
    "NHWC": 1,
    "ND": 2,
    "NC1HWC0": 3,
    "FRACTAL_Z": 4,
    "FRACTAL_NZ": 29,
    }


def save_tensor(tensor, path, acl_format):
    x = torch_npu.npu_format_cast(tensor.npu(), acl_format)
    torch.save(x, path)


def load_tensor(tensor, path):
    y = torch.load(path)

    if not torch.allclose(y.cpu(), tensor):
        raise ValueError("load tensor not equal to save tensor.")


class TestSerializationFormat(TestCase):
    def test_save_load_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            tensor = torch.rand(64, 3, 7, 7)

            proc = torch.multiprocessing.get_context("spawn").Process

            for _, acl_format in FORMAT_INFO.items():
                process_save = proc(
                    target=save_tensor,
                    name="save",
                    args=(tensor, path, acl_format),
                )
                process_save.start()
                process_save.join()
                self.assertEqual(process_save.exitcode, 0)

                process_load = proc(
                    target=load_tensor,
                    name="load",
                    args=(tensor, path),
                )
                process_load.start()
                process_load.join()
                self.assertEqual(process_load.exitcode, 0)


if __name__ == "__main__":
    run_tests()
