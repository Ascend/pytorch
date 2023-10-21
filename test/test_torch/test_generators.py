import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

device = 'npu:0'
torch.npu.set_device(device)


def get_npu_type(type_name):
    if isinstance(type_name, type):
        type_name = '{}.{}'.format(type_name.__module__, type_name.__name__)
    module, name = type_name.rsplit('.', 1)
    assert module == 'torch'
    return getattr(torch.npu, name)


class TestGenerators(TestCase):
    def test_generator(self):
        g_npu = torch.Generator(device=device)
        print(g_npu.device)
        self.assertExpectedInline(str(g_npu.device), '''npu:0''')

    def test_default_generator(self):
        output = torch.default_generator
        print(output)


if __name__ == "__main__":
    run_tests()
