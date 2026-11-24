import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch_npu.utils._dynamo import register_inductor_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestNonzeroStaticLayout(TestCase):
    def test_fake_matches_eager_layout(self):
        register_inductor_npu()
        for shape, size in (((), 1), ((2, 2), 3), ((2, 3, 2), 5), ((0, 2), 0), ((2,), 4)):
            with self.subTest(shape=shape, size=size):
                value = torch.ones(shape, device="npu", dtype=torch.bool)
                actual = torch.nonzero_static(value, size=size)
                with FakeTensorMode() as mode:
                    expected = torch.nonzero_static(mode.from_tensor(value), size=size)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.dtype, expected.dtype)
                torch.testing.assert_close(
                    actual.cpu(), torch.nonzero_static(value.cpu(), size=size)
                )
                out = torch.empty_like(actual)
                torch.nonzero_static(value, size=size, out=out)
                with FakeTensorMode() as mode:
                    fake_out = mode.from_tensor(out)
                    expected_out = torch.nonzero_static(
                        mode.from_tensor(value), size=size, out=fake_out
                    )
                self.assertEqual(out.stride(), expected_out.stride())

    def test_compiled_coordinate_consumer(self):
        register_inductor_npu()

        def select(value, mask):
            return value[torch.nonzero_static(mask, size=3)]

        value = torch.tensor([3.0, 8.0], device="npu")
        mask = torch.tensor([[True, True], [False, True]], device="npu")
        actual = torch.compile(select, fullgraph=True)(value, mask)
        torch.testing.assert_close(actual, select(value, mask))


if __name__ == "__main__":
    run_tests()
