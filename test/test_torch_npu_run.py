import os
import shutil
import tempfile
import subprocess
import unittest


class TestTorchNpuRun(unittest.TestCase):
    def setUp(self):
        self._filename = os.path.realpath(__file__)
        self._dirname = os.path.dirname(self._filename)

    def _get_command(self):
        return ['torch_npu_run', '--rdzv_backend=parallel', '--nproc_per_node=2', '--nnodes=1', '--node_rank=0',
                '--master_addr=127.0.0.1', '--master_port=29513', f'{self._dirname}/sample_torch_npu_run_store.py']

    def test_simple_torch_npu_run(self):
        outputs = list()
        command = self._get_command()
        with subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            for line in p.stdout.readlines():
                outputs.append(line.decode('utf-8'))
        self.assertEqual(0, p.returncode)
        print('++++++++++++++ output begin ++++++++++++++++++++++++++++')
        print(''.join(outputs))
        print('++++++++++++++ output end   ++++++++++++++++++++++++++++')


if __name__ == "__main__":
    unittest.main()
