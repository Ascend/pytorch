
import os
import re
import logging
import tempfile
from pathlib import Path
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from torch._inductor import config
from testutils import TestUtils
import torch_npu

os.environ["INDUCTOR_ASCEND_DUMP_FX_GRAPH"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "1"


class TestDebugMsg(TestUtils):    
    @parametrize('shape_x', [(32, 512, 64)])
    @parametrize('shape_y', [(32, 1, 64)])
    @parametrize('dtype', ['float32'])
    def test_case1(self, shape_x, shape_y, dtype):
        x = self._generate_tensor(shape_x, dtype)
        y = self._generate_tensor(shape_y, dtype)


        def run_case1(x, y):
            z = x + y
            return z

        run = torch.compile(run_case1, backend='inductor')
        with config.patch(
            {
                "trace.debug_dir": tempfile.mkdtemp(),
                "force_disable_caches": True,
            }
        ):
            with self.assertLogs(
                logging.getLogger("torch._inductor.debug"), level=logging.WARNING
            ) as cm:
                run(x, y)

        self.assertEqual(len(cm.output), 1)
        m = re.match(r"WARNING.* debug trace: (.*)", cm.output[0])
        self.assertTrue(m)
        filename = Path(m.group(1))
        self.assertTrue(filename.is_dir())
        content = open(filename / "output_code.py").read().rstrip()

        self.assertIn(
            "# SchedulerNodes: [SchedulerNode(name='op0')]",
            content
        )

        self.assertIn(
            """
# def forward(self, arg0_1, arg1_1):
#     expand = torch.ops.aten.expand.default(arg1_1, [32, 512, 64]);  arg1_1 = None
#     add = torch.ops.aten.add.Tensor(arg0_1, expand);  arg0_1 = expand = None
#     return (add,)""",
            content
        )

        self.assertIn(
            """
#  inputs: [FakeTensor(..., device='npu:0', size=(32, 512, 64), strides=(32768, 64, 1)), FakeTensor(..., device='npu:0', size=(32, 1, 64), strides=(64, 64, 1))]
#  outputs: [FakeTensor(..., device='npu:0', size=(32, 512, 64), strides=(32768, 64, 1))]""",
            content
        )


    @parametrize('shape_x', [(32, 512, 64)])
    @parametrize('shape_y', [(32, 1, 64)])
    @parametrize('dtype', ['float32'])
    def test_case2(self, shape_x, shape_y, dtype):
        x = self._generate_tensor(shape_x, dtype)
        y = self._generate_tensor(shape_y, dtype)


        def run_case2(x, y):
            z = x + y
            z = z.repeat([256, 1, 1])
            return z

        run = torch.compile(run_case2, backend='inductor')
        with config.patch(
            {
                "trace.debug_dir": tempfile.mkdtemp(),
                "force_disable_caches": True,
            }
        ):
            with self.assertLogs(
                logging.getLogger("torch._inductor.debug"), level=logging.WARNING
            ) as cm:
                run(x, y)

        self.assertEqual(len(cm.output), 1)
        m = re.match(r"WARNING.* debug trace: (.*)", cm.output[0])
        self.assertTrue(m)
        filename = Path(m.group(1))
        self.assertTrue(filename.is_dir())
        content = open(filename / "output_code.py").read().rstrip()

        self.assertIn(
            "# SchedulerNodes: [SchedulerNode(name='op0')]",
            content
        )

        self.assertIn(
            """
# def forward(self, arg0_1, arg1_1):
#     expand = torch.ops.aten.expand.default(arg1_1, [32, 512, 64]);  arg1_1 = None
#     add = torch.ops.aten.add.Tensor(arg0_1, expand);  arg0_1 = expand = None
#     repeat = torch.ops.aten.repeat.default(add, [256, 1, 1]);  add = None
#     return (repeat,)""",
            content
        )

        self.assertIn(
            """
#  inputs: [FakeTensor(..., device='npu:0', size=(32, 512, 64), strides=(32768, 64, 1)), FakeTensor(..., device='npu:0', size=(32, 1, 64), strides=(64, 64, 1))]
#  outputs: [FakeTensor(..., device='npu:0', size=(8192, 512, 64), strides=(32768, 64, 1))]""",
            content
        )


instantiate_parametrized_tests(TestDebugMsg)

if __name__ == "__main__":
    run_tests()
