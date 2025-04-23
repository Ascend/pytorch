import io
import os
import tempfile
import tarfile
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

torch_npu.npu.set_device("npu:0")


# acl format
FORMAT_NCHW = 0
FORMAT_ND = 2
FORMAT_NC1HWC0 = 3
FORMAT_NZ = 29


class NpuMNIST(nn.Module):

    def __init__(self):
        super(NpuMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """

    def __init__(self, n_in_channels, n_channels):
        super(WN, self).__init__()
        self.n_channels = n_channels
        start = torch.nn.Conv2d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

    def forward(self, forward_input):
        audio, spect = forward_input
        audio = self.start(torch.unsqueeze(audio, -1)).squeeze_(-1)
        return audio


class TestSerialization(TestCase):
    """
    The saved data is saved by using the PyTorch CPU storage structure, but
    following `torch.load()`  will load the corresponding NPU data.
    """

    def test_save(self):
        x = torch.randn(5).npu()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            torch.save(x, path)
            self.assertExpectedInline(f'{x.device.type}:{x.device.index}', 'npu:0')
            x_loaded = torch.load(path, map_location="npu:0")
            self.assertRtolEqual(x.cpu(), x_loaded.cpu())
            y_loaded = torch.load(path, map_location="npu")
            self.assertRtolEqual(x.cpu(), y_loaded.cpu())
            z_loaded = torch.load(path, map_location=torch.device("npu"))
            self.assertRtolEqual(x.cpu(), z_loaded.cpu())
            m_loaded = torch.load(path, map_location="cpu")
            self.assertRtolEqual(x.cpu(), m_loaded)
            n_loaded = torch.load(path, map_location=torch.device("cpu"))
            self.assertRtolEqual(x.cpu(), n_loaded)
            x_loaded = torch.load(path, map_location="npu:0", weights_only=True)
            self.assertRtolEqual(x.cpu(), x_loaded.cpu())

    def test_load_legacy_file(self):
        x = torch.randn(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            torch.serialization.save(x, path, _use_new_zipfile_serialization=False)
            x_loaded = torch.load(path, map_location="npu:0")
            self.assertExpectedInline(f'{x_loaded.device.type}:{x_loaded.device.index}', 'npu:0')
            self.assertRtolEqual(x, x_loaded.cpu())
            x_loaded = torch.load(path, map_location=torch.device("npu:0"))
            self.assertExpectedInline(f'{x_loaded.device.type}:{x_loaded.device.index}', 'npu:0')
            self.assertRtolEqual(x, x_loaded.cpu())

    def test_save_npu_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            x = torch_npu.npu_format_cast(torch.randn(2, 3, 224, 224).npu(), FORMAT_NCHW)
            torch.save(x, path)
            x_loaded = torch.load(path)
            self.assertRtolEqual(torch_npu.get_npu_format(x),
                                 torch_npu.get_npu_format(x_loaded))
            x = torch_npu.npu_format_cast(torch.randn(2, 3, 224, 224).npu(), FORMAT_ND)
            torch.save(x, path)
            x_loaded = torch.load(path)
            self.assertRtolEqual(torch_npu.get_npu_format(x),
                                 torch_npu.get_npu_format(x_loaded))
            x = torch_npu.npu_format_cast(torch.randn(2, 3, 224, 224).npu(), FORMAT_NC1HWC0)
            torch.save(x, path)
            x_loaded = torch.load(path)
            self.assertRtolEqual(torch_npu.get_npu_format(x),
                                 torch_npu.get_npu_format(x_loaded))
            x = torch_npu.npu_format_cast(torch.randn(2, 3, 224, 224).npu(), FORMAT_NZ)
            torch.save(x, path)
            x_loaded = torch.load(path)
            self.assertRtolEqual(torch_npu.get_npu_format(x),
                                 torch_npu.get_npu_format(x_loaded))

    def test_save_noncontiguous_tensor(self):
        x = torch.randn(2, 5, 6).npu()
        y = x[:, :, :2]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            torch.save(y, path)
            y_loaded = torch.load(path)
            self.assertRtolEqual(y.cpu(), y_loaded.cpu())

        y = x[:, :2, :]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            torch.save(y, path)
            y_loaded = torch.load(path)
            self.assertRtolEqual(y.cpu(), y_loaded.cpu())

        y = x[:1, :, :]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            torch.save(y, path)
            y_loaded = torch.load(path)
            self.assertRtolEqual(y.cpu(), y_loaded.cpu())

    def test_load_maplocation(self):
        x = torch.randn(2, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            torch.save(x, path)
            x_loaded = torch.load(path, map_location="npu:0")
            self.assertExpectedInline(f'{x_loaded.device.type}', 'npu')
            self.assertRtolEqual(x, x_loaded.cpu())
            x_loaded = torch.load(path, map_location=torch.device("npu:0"))
            self.assertExpectedInline(f'{x_loaded.device.type}', 'npu')
            self.assertRtolEqual(x, x_loaded.cpu())

    def test_save_string(self):
        x = dict(ds_version='0.6.0+0b40f54')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            torch.save(x, path)
            x_loaded = torch.load(path)
            self.assertExpectedInline(str(x), str(x_loaded))
            x_loaded = torch.load(path, weights_only=True)
            self.assertExpectedInline(str(x), str(x_loaded))

    def test_save_torch_size(self):
        x = torch.randn(2, 3).size()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            torch.save(x, path)
            x_loaded = torch.load(path)
            self.assertExpectedInline(str(x), str(x_loaded))
            x_loaded = torch.load(path, weights_only=True)
            self.assertExpectedInline(str(x), str(x_loaded))

    def test_save_tuple(self):
        x = torch.randn(5).npu()
        model = NpuMNIST().npu()
        number = 3
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            torch.save((x, model, number), path)
            x_loaded, model_loaded, number_loaded = torch.load(path)
            self.assertRtolEqual(x.cpu(), x_loaded.cpu())
            self.assertExpectedInline(str(model), str(model_loaded))
            self.assertTrue(number, number_loaded)

    def test_save_value(self):
        x = 44
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            torch.save(x, path)
            x_loaded = torch.load(path)
            self.assertTrue(x, x_loaded)
            x_loaded = torch.load(path, weights_only=True)
            self.assertTrue(x, x_loaded)

    def test_save_argparse_namespace(self):
        args = argparse.Namespace()
        args.foo = 1
        args.bar = [1, 2, 3]
        args.baz = 'yippee'
        args.tensor = torch.tensor([1., 2., 3.]).npu()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            torch.save(args, path)
            args_loaded = torch.load(path)
            self.assertTrue(args, args_loaded)

    def test_serialization_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            model = NpuMNIST().npu()
            torch.save(model, path)
            loaded_model = torch.load(path)
            self.assertExpectedInline(str(model), str(loaded_model))

    def test_serialization_weight_norm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            model = WN(2, 4).npu()
            torch.save(model, path)
            loaded_model = torch.load(path)
            self.assertExpectedInline(str(model), str(loaded_model))

    def test_model_storage_ptr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            model = NpuMNIST().npu()
            ptr1 = model.conv1.weight.data_ptr()
            torch.save(model, path)
            ptr2 = model.conv1.weight.data_ptr()
            self.assertEqual(ptr1, ptr2)

    def test_serialization_state_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            model = NpuMNIST().npu()
            torch.save(model.state_dict(), path)
            state_dict = torch.load(path)
            cpu_model = NpuMNIST()
            cpu_model.load_state_dict(state_dict)
            loaded_model = cpu_model.npu()
            before_save = model.state_dict()
            after_load = loaded_model.state_dict()

            self.assertRtolEqual(before_save['conv1.weight'].cpu(), after_load['conv1.weight'].cpu())
            self.assertRtolEqual(before_save['conv2.weight'].cpu(), after_load['conv2.weight'].cpu())
            self.assertRtolEqual(before_save['fc1.weight'].cpu(), after_load['fc1.weight'].cpu())
            self.assertRtolEqual(before_save['fc2.weight'].cpu(), after_load['fc2.weight'].cpu())
            self.assertRtolEqual(before_save['conv1.bias'].cpu(), after_load['conv1.bias'].cpu())
            self.assertRtolEqual(before_save['conv2.bias'].cpu(), after_load['conv2.bias'].cpu())
            self.assertRtolEqual(before_save['fc1.bias'].cpu(), after_load['fc1.bias'].cpu())
            self.assertRtolEqual(before_save['fc2.bias'].cpu(), after_load['fc2.bias'].cpu())

    def test_save_different_dtype_unallocated(self):

        def save_load_check():
            with io.BytesIO() as f:
                torch.save([a, b], f)
                f.seek(0)
                a_loaded, b_loaded = torch.load(f)
            self.assertEqual(a, a_loaded)
            self.assertEqual(b, b_loaded)

        dtypes = []
        for dtype in dtypes:
            a = torch.tensor([], dtype=dtype, device='npu')
            for other_dtype in dtypes:
                s = torch.TypedStorage(wrap_storage=a.storage().untyped(), dtype=other_dtype)
                save_load_check(a, s)
                save_load_check(a.storage(), s)
                b = torch.tensor([], dtype=other_dtype, device='npu')
                save_load_check(a, b)

    def test_tarfile_with_weights_only_unpickler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mock.tar")
            with tarfile.open(path, 'w') as tar:
                tar.add(os.path.devnull, arcname="empty_file")

            with self.assertRaisesRegex(
                Exception, "Cannot use ``weights_only=True`` with files saved in the legacy .tar format"
            ):
                torch.load(path, weights_only=True)

            with self.assertRaisesRegex(Exception, "Unsupported operand"):
                with open(path, "rb") as opened_file:
                    try:
                        with tarfile.open(fileobj=opened_file, mode="r:", format=tarfile.PAX_FORMAT):
                            pass
                    finally:
                        torch.load(opened_file, weights_only=True)


if __name__ == "__main__":
    run_tests()
