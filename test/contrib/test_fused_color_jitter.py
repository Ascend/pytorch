from PIL import Image
import numpy as np

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib.module import FusedColorJitter


class TestFusedColorJitter(TestCase):
    def test_fusedcolorjitter(self):
        image = Image.fromarray(torch.randint(0, 256, size=(224, 224, 3)).numpy().astype(np.uint8))
        fcj = FusedColorJitter(0.1, 0.1, 0.1, 0.1)
        output = fcj(image)
        self.assertEqual(output is not None, True)

if __name__ == "__main__":
    run_tests()
