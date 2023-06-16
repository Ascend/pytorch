import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestNormal(TestCase):

    @Dtypes(torch.float)
    def test_normal(self, dtype=torch.float16):
        q = torch.empty(100, 100, dtype=dtype, device="cpu").to("npu")
        q.normal_()
        self.assertEqual(q.mean(), 0, 0.2)
        self.assertEqual(q.to("cpu").to(torch.float).std(), 1, 0.2)

        q.normal_(2, 3)
        self.assertEqual(q.mean(), 2, 0.3)
        self.assertEqual(q.to("cpu").to(torch.float).std(), 3, 0.3)

        mean = torch.empty(100, 100, dtype=dtype, device="cpu").to("npu")
        std = torch.empty(100, 100, dtype=dtype, device="cpu").to("npu")
        mean.fill_(-2)
        std.fill_(3)

        r = torch.normal(mean)
        self.assertEqual(r.mean(), -2, 0.2)
        self.assertEqual(r.to("cpu").to(torch.float).std(), 1, 0.2)

        r = torch.normal(mean, 3)
        self.assertEqual(r.mean(), -2, 0.2)
        self.assertEqual(r.to("cpu").to(torch.float).std(), 3, 0.2)

        r = torch.normal(2, std)
        self.assertEqual(r.mean(), 2, 0.2)
        self.assertEqual(r.to("cpu").to(torch.float).std(), 3, 0.2)

        r = torch.normal(mean, std)
        self.assertEqual(r.mean(), -2, 0.2)
        self.assertEqual(r.to("cpu").to(torch.float).std(), 3, 0.3)

        r = torch.normal(2, 3, (100, 100))
        self.assertEqual(r.mean(), 2, 0.2)
        self.assertEqual(r.to("cpu").to(torch.float).std(), 3, 0.2)

    def test_normal_dtype_float16(self):
        size = [500]
        mean = torch.empty(size=size, dtype=torch.float16).to("npu")
        mean.fill_(2.0)
        std = 2.0
        out = torch.normal(mean=mean, std=std).to(torch.float32)
        self.assertEqual(out.to("cpu").mean(), 2.0, 0.2)
        self.assertEqual(out.to("cpu").std(), 2.0, 0.2)
        out = torch.empty(size=size, dtype=torch.float16).to("npu")
        torch.normal(mean=mean, std=std, out=out)
        self.assertEqual(out.to(torch.float32).to("cpu").mean(), 2.0, 0.2)
        self.assertEqual(out.to(torch.float32).to("cpu").std(), 2.0, 0.2)

        size = [27, 33]
        mean = 2.0
        std = torch.empty(size=size, dtype=torch.float16).to("npu")
        std.fill_(2.0)
        out = torch.normal(mean=mean, std=std).to(torch.float32)
        self.assertEqual(out.to(torch.float32).to("cpu").mean(), 2.0, 0.2)
        self.assertEqual(out.to(torch.float32).to("cpu").std(), 2.0, 0.2)
        out = torch.empty(size=size, dtype=torch.float16).to("npu")
        torch.normal(mean=mean, std=std, out=out)
        self.assertEqual(out.to(torch.float32).to("cpu").mean(), 2.0, 0.2)
        self.assertEqual(out.to(torch.float32).to("cpu").std(), 2.0, 0.2)

        size = [11, 11, 11]
        mean = torch.empty(size=size, dtype=torch.float16).to("npu")
        std = torch.empty(size=size, dtype=torch.float16).to("npu")
        mean.fill_(2.0)
        std.fill_(2.0)
        out = torch.normal(mean=mean, std=std).to(torch.float32)
        self.assertEqual(out.to(torch.float32).to("cpu").mean(), 2.0, 0.2)
        self.assertEqual(out.to(torch.float32).to("cpu").std(), 2.0, 0.2)
        out = torch.empty(size=size, dtype=torch.float16).to("npu")
        torch.normal(mean=mean, std=std, out=out)
        self.assertEqual(out.to(torch.float32).to("cpu").mean(), 2.0, 0.2)
        self.assertEqual(out.to(torch.float32).to("cpu").std(), 2.0, 0.2)

    def test_normal_dtype_float32(self):
        size = [25, 25]
        mean = 5.0
        std = 5.0
        out = torch.normal(mean=mean, std=std, size=size, device="npu")
        self.assertEqual(out.to("cpu").mean(), 5.0, 0.5)
        self.assertEqual(out.to("cpu").std(), 5.0, 0.5)

        size = [7, 9, 11]
        out = torch.empty(size=size, dtype=torch.float32).to("npu")
        torch.normal(mean=mean, std=std, out=out, size=size)
        self.assertEqual(out.to("cpu").mean(), 5.0, 0.5)
        self.assertEqual(out.to("cpu").std(), 5.0, 0.5)

        size = [4, 5, 8, 10]
        out = torch.empty(size=size, dtype=torch.float32).to("npu")
        out.normal_(mean=mean, std=std)
        self.assertEqual(out.to("cpu").mean(), 5.0, 0.5)
        self.assertEqual(out.to("cpu").std(), 5.0, 0.5)
        out = torch.empty(size=size, dtype=torch.float32).to("npu").transpose(0, 1)
        out.normal_(mean=mean, std=std)
        self.assertEqual(out.to("cpu").mean(), 5.0, 0.5)
        self.assertEqual(out.to("cpu").std(), 5.0, 0.5)

        size = [25, 35]
        mean = torch.empty(size=size, dtype=torch.float32).to("npu")
        mean.fill_(5.0)
        std = 5.0
        out = torch.normal(mean=mean, std=std)
        self.assertEqual(out.to("cpu").mean(), 5.0, 0.5)
        self.assertEqual(out.to("cpu").std(), 5.0, 0.5)
        out = torch.empty(size=size, dtype=torch.float32).to("npu")
        torch.normal(mean=mean, std=std, out=out)
        self.assertEqual(out.to("cpu").mean(), 5.0, 0.5)
        self.assertEqual(out.to("cpu").std(), 5.0, 0.5)

        size = [7, 8, 15]
        mean = 5.0
        std = torch.empty(size=size, dtype=torch.float32).to("npu")
        std.fill_(5.0)
        out = torch.normal(mean=mean, std=std)
        self.assertEqual(out.to("cpu").mean(), 5.0, 0.5)
        self.assertEqual(out.to("cpu").std(), 5.0, 0.5)
        out = torch.empty(size=size, dtype=torch.float32).to("npu")
        torch.normal(mean=mean, std=std, out=out)
        self.assertEqual(out.to("cpu").mean(), 5.0, 0.5)
        self.assertEqual(out.to("cpu").std(), 5.0, 0.5)

        size = [800]
        mean = torch.empty(size=size, dtype=torch.float32).to("npu")
        std = torch.empty(size=size, dtype=torch.float32).to("npu")
        mean.fill_(5.0)
        std.fill_(5.0)
        out = torch.normal(mean=mean, std=std)
        self.assertEqual(out.to("cpu").mean(), 5.0, 0.5)
        self.assertEqual(out.to("cpu").std(), 5.0, 0.5)
        out = torch.empty(size=size, dtype=torch.float32).to("npu")
        torch.normal(mean=mean, std=std, out=out)
        self.assertEqual(out.to("cpu").mean(), 5.0, 0.5)
        self.assertEqual(out.to("cpu").std(), 5.0, 0.5)

    def test_normal_format(self):
        size = [5, 6, 7, 8]
        mean = torch.empty(size=size, dtype=torch.float32).to("npu")
        std = torch.empty(size=size, dtype=torch.float32).to("npu")
        mean.fill_(5.0)
        std.fill_(5.0)

        mean = torch_npu.npu_format_cast(mean, 0)
        std = torch_npu.npu_format_cast(std, 0)
        out = torch.normal(mean=mean, std=std)
        self.assertEqual(out.to("cpu").mean(), 5.0, 0.5)
        self.assertEqual(out.to("cpu").std(), 5.0, 0.5)

        mean = torch_npu.npu_format_cast(mean, 2)
        std = torch_npu.npu_format_cast(std, 2)
        out = torch.normal(mean=mean, std=std)
        self.assertEqual(out.to("cpu").mean(), 5.0, 0.5)
        self.assertEqual(out.to("cpu").std(), 5.0, 0.5)

    def test_normal_seed(self):
        torch.manual_seed(123)
        mean = torch.rand(2, 3).npu()
        std = torch.rand(2, 1).npu()
        output1 = torch.normal(mean, std)
        torch.manual_seed(123)
        output2 = torch.normal(mean, std)
        self.assertRtolEqual(output1.cpu(), output2.cpu())

    def test_normal_seed_fp16(self):
        torch.manual_seed(23)
        mean = torch.rand(2, 3).half().npu()
        std = torch.rand(2, 1).half().npu()
        output1 = torch.normal(mean, std)
        torch.manual_seed(23)
        output2 = torch.normal(mean, std)
        self.assertRtolEqual(output1.cpu(), output2.cpu())

    def test_normal_broadcast(self):
        torch.manual_seed(23)
        mean = torch.randn(2, 1, 4, 5).npu()
        std = torch.randn(2, 3, 4, 5).npu()
        output1 = torch.normal(mean, std)
        torch.manual_seed(23)
        output2 = torch.normal(mean, std)
        self.assertRtolEqual(output1.cpu(), output2.cpu())


if __name__ == "__main__":
    run_tests()
