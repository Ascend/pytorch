import multiprocessing

import torch
from torch_npu.testing.testcase import TestCase, run_tests


class TorchDeterministicApiTestCase(TestCase):

    def npu_op_exec(self, data, target, reduction):
        loss = torch.nn.NLLLoss2d(reduction=reduction)
        output = loss(data, target)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output

    def test_npu_set_deterministic(self):
        torch.use_deterministic_algorithms(True)
        res = torch.are_deterministic_algorithms_enabled()
        self.assertEqual(res, True)

        torch.use_deterministic_algorithms(False)
        res = torch.are_deterministic_algorithms_enabled()
        self.assertEqual(res, False)


def test_npu_set_deterministic_true():
    torch.use_deterministic_algorithms(True)
    tensora = torch.randn(64, 3, 64, 64, dtype=torch.float, device="npu:0")
    tensorsum = torch.sum(tensora)
    index = 0
    for i in range(100):
        res = torch.sum(tensora)
        if (res == tensorsum):
            index = index + 1
    if (index != 100):
        raise AssertionError("failed to test_npu_set_deterministic_true!")


def test_deterministic_newprocessing():
    Process_jobs = []
    multiprocessing.set_start_method("spawn", force=True)
    p = multiprocessing.Process(target=test_npu_set_deterministic_true)
    Process_jobs.append(p)
    p.start()
    p.join()


if __name__ == "__main__":
    test_deterministic_newprocessing()
    option = {}
    option["ACL_OP_COMPILER_CACHE_MODE"] = "disable"
    torch.npu.set_option(option)
    run_tests()
