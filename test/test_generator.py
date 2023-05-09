import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class GeneratorTest(TestCase):

    def test_state(self):
        gen = torch_npu._C.Generator()
        gen.set_state(torch.get_rng_state())
        self.assertEqual(gen.get_state(), torch.get_rng_state())

    def test_seed(self):
        gen = torch_npu._C.Generator(torch_npu.npu.native_device + ":" + str(torch_npu.npu.current_device()))
        gen.manual_seed(1234)
        self.assertEqual(gen.initial_seed(), 1234)


if __name__ == '__main__':
    run_tests()
